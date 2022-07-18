# BokehMe: When Neural Rendering Meets Classical Rendering (CVPR 2022 Oral)

[Juewen Peng](https://scholar.google.com/citations?hl=en&user=fYC6lCUAAAAJ)<sup>1</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>,
[Xianrui Luo](https://scholar.google.com/citations?hl=en&user=tUeWQ5AAAAAJ)<sup>1</sup>,
[Hao Lu](http://faculty.hust.edu.cn/LUHAO/en/index.htm)<sup>1</sup>,
[Ke Xian](https://sites.google.com/site/kexian1991/)<sup>1*</sup>,
[Jianming Zhang](https://jimmie33.github.io/)<sup>2</sup>

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Adobe Research

<p align="center">
<img src=https://user-images.githubusercontent.com/38718148/171405815-b3cc8799-27cd-457e-89df-686695187554.jpg />
</p>

### [Project](https://juewenpeng.github.io/BokehMe/) | [Paper](https://github.com/JuewenPeng/BokehMe/blob/main/pdf/BokehMe.pdf) | [Supp](https://github.com/JuewenPeng/BokehMe/blob/main/pdf/BokehMe-supp.pdf) | [Poster](https://github.com/JuewenPeng/BokehMe/blob/main/pdf/BokehMe-poster.pdf) | [Video](https://www.youtube.com/watch?v=e-zr_wCxNc8) | [Data](#blb-dataset)

This repository is the official PyTorch implementation of the CVPR 2022 paper "BokehMe: When Neural Rendering Meets Classical Rendering".


**NOTE**: There is a citation mistake in the paper of the conference version. In section 4.1, the disparity maps of the EBB400 dataset are predicted by MiDaS [1] instead of DPT [2]. <!-- We have corrected it in the arXiv version. We apologize for this oversight and for any confusion that it may have caused.  --><br/>
> [1] Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer <br/>
> [2] Vision Transformers for Dense Prediction



## Installation
```
git clone https://github.com/JuewenPeng/BokehMe.git
cd BokehMe
pip install -r requirements.txt
```


## Usage
```
python demo.py --image_path 'inputs/21.jpg' --disp_path 'inputs/21.png' --save_dir 'outputs' --K 60 --disp_focus 90/255 --gamma 4 --highlight
```
- `image_path`:  path of the input all-in-focus image
- `disp_path`: path of the input disparity map (predicted by [DPT](https://github.com/isl-org/DPT) in this example)
- `save_dir`: directory to save the results
- `K`: blur parameter
- `disp_focus`: refocused disparity (range from 0 to 1)
- `gamma`: gamma value (range from 1 to 5)
- `highlight`: enhance RGB values of highlights before rendering for stunning bokeh balls

See `demo.py` for more details.




## BLB Dataset
The BLB dataset is synthesized by Blender 2.93. It contains 10 scenes, each consisting of an all-in-focus image, a disparity map, a stack of bokeh images with 5 blur amounts and 10 refocused disparities, and a parameter file. We additionally provide 15 corrupted disparity maps (through gaussian blur, dilation, erosion) for each scene. Our BLB dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1URpab6AXQsNTqcBcighF73w5pFlvM0Ej?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1U0XlFM_84-vVgnXGYz0ncQ?pwd=re8q).

**Instructions**: 
- EXR images can be loaded by `image = cv2.imread(IMAGE_PATH, -1)[..., :3].astype(np.float32) ** (1/2.2)` . The loaded images are in BGR, so you can convert them to RGB by `image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` if necessary.
- EXR depth maps can be loaded by `depth = cv2.imread(DEPTH_PATH, -1)[..., 0].astype(np.float32)`. You can convert them to disparity maps by `disp = 1 / depth`. Note that it is **unnecesary** to normalize the disparity maps since we have pre-processed them to ensure that the signed defocus maps calculated by `K * (disp - disp_focus)` are in line with the experimental settings of the paper.
- NOTE: Some pixel values of images may be larger than 1 for highlights (but mostly smaller than 1). Considering the fact that some rendering methods can only output values between 0 and 1, we clip the numerical ranges of the predicted bokeh images and the real ones to [0, 1] before evaluation. The main reason for this phenomenon (image values exceeding 1) is that the EXR images exported from Blender are in linear space, and we only process them with gamma 2.2 correction without tone mapping. We will improve it in the future.

## Citation
If you find our work useful in your research, please cite our paper.

```
@inproceedings{Peng2022BokehMe,
  title = {BokehMe: When Neural Rendering Meets Classical Rendering},
  author = {Peng, Juewen and Cao, Zhiguo and Luo, Xianrui and Lu, Hao and Xian, Ke and Zhang, Jianming},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}
```
