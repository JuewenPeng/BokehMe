#!/user/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy
import re

kernel_Render_updateOutput = '''

    extern "C" __global__ void kernel_Render_updateOutput(
        const int n,
        const float* image,          // original image
        const float* defocus,        // signed defocus map
        int* defocusDilate,          // signed defocus map after dilating
        float* bokehCum,             // cumulative bokeh image
        float* weightCum             // cumulative weight map
    )
    {
        for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int intN = ( intIndex / SIZE_3(weightCum) / SIZE_2(weightCum) / SIZE_1(weightCum) ) % SIZE_0(weightCum);
            // const int intC = ( intIndex / SIZE_3(weightCum) / SIZE_2(weightCum)                     ) % SIZE_1(weightCum);
            const int intY = ( intIndex / SIZE_3(weightCum)                                         ) % SIZE_2(weightCum);
            const int intX = ( intIndex                                                             ) % SIZE_3(weightCum);

            float fltDefocus = VALUE_4(defocus, intN, 0, intY, intX);
            float fltRadius = fabsf(fltDefocus);

            for (int intDeltaY = -(int)(fltRadius)-1; intDeltaY <= (int)(fltRadius)+1; ++intDeltaY) {
                for (int intDeltaX = -(int)(fltRadius)-1; intDeltaX <= (int)(fltRadius)+1; ++intDeltaX) {

                    int intNeighborY = intY + intDeltaY;
                    int intNeighborX = intX + intDeltaX;

                    if ((intNeighborY >= 0) && (intNeighborY < SIZE_2(bokehCum)) && (intNeighborX >= 0) && (intNeighborX < SIZE_3(bokehCum))) {
                        float fltDist = sqrtf((float)(intDeltaY)*(float)(intDeltaY) + (float)(intDeltaX)*(float)(intDeltaX));
                        float fltWeight = (0.5 + 0.5 * tanhf(4 * (fltRadius - fltDist))) / (fltRadius * fltRadius + 0.2);
                        if (fltRadius >= fltDist) {
                            atomicMax(&defocusDilate[OFFSET_4(defocusDilate, intN, 0, intNeighborY, intNeighborX)], int(fltDefocus));
                        }
                        atomicAdd(&weightCum[OFFSET_4(weightCum, intN, 0, intNeighborY, intNeighborX)], fltWeight);
                        atomicAdd(&bokehCum[OFFSET_4(bokehCum, intN, 0, intNeighborY, intNeighborX)], fltWeight * VALUE_4(image, intN, 0, intY, intX));
                        atomicAdd(&bokehCum[OFFSET_4(bokehCum, intN, 1, intNeighborY, intNeighborX)], fltWeight * VALUE_4(image, intN, 1, intY, intX));
                        atomicAdd(&bokehCum[OFFSET_4(bokehCum, intN, 2, intNeighborY, intNeighborX)], fltWeight * VALUE_4(image, intN, 2, intY, intX));
                    }
                }
            }
        }
    }

'''


def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction]

    while True:
        objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArg = int(objMatch.group(2))

        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()

        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')
    # end

    while True:
        objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    return strKernel
# end


# @cupy.util.memoize(for_each_device=True)
@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end


class _FunctionRender(torch.autograd.Function):
    @staticmethod
    def forward(self, image, defocus):
        # self.save_for_backward(image, defocus)

        defocus_dilate = defocus.int()
        bokeh_cum = torch.zeros_like(image)
        weight_cum = torch.zeros_like(defocus)

        if defocus.is_cuda == True:
            n = weight_cum.nelement()
            cupy_launch('kernel_Render_updateOutput', cupy_kernel('kernel_Render_updateOutput', {
                'image': image,
                'defocus': defocus,
                'defocusDilate': defocus_dilate,
                'bokehCum': bokeh_cum,
                'weightCum': weight_cum
            }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cupy.int(n),
                    image.data_ptr(),
                    defocus.data_ptr(),
                    defocus_dilate.data_ptr(),
                    bokeh_cum.data_ptr(),
                    weight_cum.data_ptr()
                ]
            )

        elif defocus.is_cuda == False:
            raise NotImplementedError()

        # end

        return defocus_dilate.float(), bokeh_cum, weight_cum
    # end

    # @staticmethod
    # def backward(self, gradBokehCum, gradWeightCum):
    # end

# end


def FunctionRender(image, defocus):
    defocus_dilate, bokeh_cum, weight_cum = _FunctionRender.apply(image, defocus)

    return defocus_dilate, bokeh_cum, weight_cum
# end


class ModuleRenderScatter(torch.nn.Module):
    def __init__(self):
        super(ModuleRenderScatter, self).__init__()
    # end

    def forward(self, image, defocus):
        defocus_dilate, bokeh_cum, weight_cum = FunctionRender(image, defocus)
        bokeh = bokeh_cum / weight_cum
        return bokeh, defocus_dilate
    # end
# end
