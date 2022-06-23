#!/usr/bin/env python
# encoding: utf-8
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
import torch.nn as nn
import torch.nn.functional as F


class Space2Depth(nn.Module):
    def __init__(self, down_factor):
        super(Space2Depth, self).__init__()
        self.down_factor = down_factor

    def forward(self, x):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, self.down_factor, stride=self.down_factor)
        return unfolded_x.view(n, c * self.down_factor ** 2, h // self.down_factor, w // self.down_factor)


def conv_bn_activation(in_channels, out_channels, kernel_size, stride, padding, use_bn, activation):
    module = nn.Sequential()
    # module.add_module('pad', nn.ReflectionPad2d(padding))
    module.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    module.add_module('bn', nn.BatchNorm2d(out_channels)) if use_bn else None
    module.add_module('activation', activation) if activation else None

    return module


class BlockStack(nn.Module):
    def __init__(self, channels, num_block, share_weight, connect_mode, use_bn, activation):
        # connect_mode: refer to "Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks"
        super(BlockStack, self).__init__()

        self.num_block = num_block
        self.connect_mode = connect_mode

        self.blocks = nn.ModuleList()

        if share_weight is True:
            block = nn.Sequential(
                conv_bn_activation(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3, stride=1, padding=1,
                    use_bn=use_bn, activation=activation
                ),
                conv_bn_activation(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3, stride=1, padding=1,
                    use_bn=use_bn, activation=activation
                )
            )
            for i in range(num_block):
                self.blocks.append(block)

        else:
            for i in range(num_block):
                block = nn.Sequential(
                    conv_bn_activation(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=3, stride=1, padding=1,
                        use_bn=use_bn, activation=activation
                    ),
                    conv_bn_activation(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=3, stride=1, padding=1,
                        use_bn=use_bn, activation=activation
                    )
                )
                self.blocks.append(block)

    def forward(self, x):
        if self.connect_mode == 'no':
            for i in range(self.num_block):
                x = self.blocks[i](x)
        elif self.connect_mode == 'distinct_source':
            for i in range(self.num_block):
                x = self.blocks[i](x) + x
        elif self.connect_mode == 'shared_source':
            x0 = x
            for i in range(self.num_block):
                x = self.blocks[i](x) + x0
        else:
            print('"connect_mode" error!')
            exit(0)
        return x


class ARNet(nn.Module):  # Adaptive Rendering Network
    def __init__(self, shuffle_rate=2, in_channels=5, out_channels=4, middle_channels=128, num_block=3, share_weight=False, connect_mode='distinct_source', use_bn=False, activation='elu'):
        super(ARNet, self).__init__()

        self.shuffle_rate = shuffle_rate
        self.connect_mode = connect_mode

        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            activation = nn.LeakyReLU(inplace=True)
        elif activation == 'elu':
            activation = nn.ELU(inplace=True)
        else:
            print('"activation" error!')
            exit(0)

        self.downsample = Space2Depth(shuffle_rate)
        self.conv0 = conv_bn_activation(
            in_channels=(in_channels - 1) * shuffle_rate ** 2 + 1,
            out_channels=middle_channels,
            kernel_size=3, stride=1, padding=1,
            use_bn=use_bn, activation=activation
        )
        self.block_stack = BlockStack(
            channels=middle_channels,
            num_block=num_block, share_weight=share_weight, connect_mode=connect_mode,
            use_bn=use_bn, activation=activation
        )
        self.conv1 = conv_bn_activation(
            in_channels=middle_channels,
            out_channels=out_channels * shuffle_rate ** 2,
            kernel_size=3, stride=1, padding=1,
            use_bn=False, activation=None
        )
        self.upsample = nn.PixelShuffle(shuffle_rate)

    def forward(self, image, defocus, gamma):
        _, _, h, w = image.shape
        h_re = int(h // self.shuffle_rate * self.shuffle_rate)
        w_re = int(w // self.shuffle_rate * self.shuffle_rate)
        x = torch.cat((image, defocus), dim=1)
        x = F.interpolate(x, size=(h_re, w_re), mode='bilinear', align_corners=True)
        x = self.downsample(x)
        gamma = torch.ones_like(x[:, :1]) * gamma
        x = torch.cat((x, gamma), dim=1)
        x = self.conv0(x)
        x = self.block_stack(x)
        x = self.conv1(x)
        x = self.upsample(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        bokeh = x[:, :-1]
        mask = torch.sigmoid(x[:, -1:])

        return bokeh, mask


class IUNet(nn.Module):  # Iterative Upsampling Network
    def __init__(self, shuffle_rate=2, in_channels=8, out_channels=3, middle_channels=64, num_block=3, share_weight=False, connect_mode='distinct_source', use_bn=False, activation='elu'):
        super(IUNet, self).__init__()

        self.shuffle_rate = shuffle_rate
        self.connect_mode = connect_mode

        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            activation = nn.LeakyReLU(inplace=True)
        elif activation == 'elu':
            activation = nn.ELU(inplace=True)
        else:
            print('"activation" error!')
            exit(0)

        self.downsample = Space2Depth(shuffle_rate)
        self.conv0 = conv_bn_activation(
            in_channels=(in_channels - 4) * shuffle_rate ** 2 + 4,
            out_channels=middle_channels,
            kernel_size=3, stride=1, padding=1,
            use_bn=use_bn, activation=activation
        )
        self.block_stack = BlockStack(
            channels=middle_channels,
            num_block=num_block, share_weight=share_weight, connect_mode=connect_mode,
            use_bn=use_bn, activation=activation
        )
        self.conv1 = conv_bn_activation(
            in_channels=middle_channels,
            out_channels=out_channels * shuffle_rate ** 2,
            kernel_size=3, stride=1, padding=1,
            use_bn=False, activation=None
        )
        self.upsample = nn.PixelShuffle(shuffle_rate)

    def forward(self, image, defocus, bokeh_coarse, gamma):
        _, _, h, w = image.shape
        h_re = int(h // self.shuffle_rate * self.shuffle_rate)
        w_re = int(w // self.shuffle_rate * self.shuffle_rate)
        x = torch.cat((image, defocus), dim=1)
        x = F.interpolate(x, size=(h_re, w_re), mode='bilinear', align_corners=True)
        x = self.downsample(x)
        if bokeh_coarse.shape[2] != x.shape[2] or bokeh_coarse.shape[3] != x.shape[3]:
            bokeh_coarse = F.interpolate(bokeh_coarse, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        gamma = torch.ones_like(x[:, :1]) * gamma
        x = torch.cat((x, bokeh_coarse, gamma), dim=1)
        x = self.conv0(x)
        x = self.block_stack(x)
        x = self.conv1(x)
        x = self.upsample(x)
        bokeh_refine = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return bokeh_refine
