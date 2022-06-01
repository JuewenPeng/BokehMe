# BokehMe: When Neural Rendering Meets Classical Rendering (CVPR 2022 Oral)

[Juewen Peng](https://scholar.google.com/citations?hl=en&user=fYC6lCUAAAAJ)<sup>1</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>,
[Xianrui Luo](https://scholar.google.com/citations?hl=en&user=tUeWQ5AAAAAJ)<sup>1</sup>,
[Hao Lu](http://faculty.hust.edu.cn/LUHAO/en/index.htm)<sup>1</sup>,
[Ke Xian](https://sites.google.com/site/kexian1991/)<sup>1*</sup>,
[Jianming Zhang](https://jimmie33.github.io/)<sup>2</sup>

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Adobe Research

---

### [Project](https://juewenpeng.github.io/BokehMe/) | [Paper](https://github.com/JuewenPeng/BokehMe/blob/main/pdf/BokehMe.pdf) | [Supp](https://github.com/JuewenPeng/BokehMe/blob/main/pdf/BokehMe-supp.pdf) | [Data](#blb-dataset)

This repository is the official PyTorch implementation of the paper "BokehMe: When Neural Rendering Meets Classical Rendering".


**NOTE**: There is a citation mistake in the paper of the conference version. In section 4.1, the disparity maps of the EBB400 dataset are predicted by MiDaS [1] instead of DPT [2]. <!-- We have corrected it in the arXiv version. We apologize for this oversight and for any confusion that it may have caused.  --><br/>
> [1] Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer <br/>
> [2] Vision Transformers for Dense Prediction

<!-- 

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
- `image_path`:  path of the input all-in-focus image (of arbitrary resolution)
- `disp_path`: path of the input disparity map (predicted by [DPT](https://github.com/isl-org/DPT) in this example)
- `save_dir`: directory to save the results
- `K`: blur parameter
- `disp_focus`: refocused disparity (range from 0 to 1)
- `gamma`: gamma value (range from 1 to 5)
- `highlight`: forcibly enhance RGB values of highlights before rendering for more obvious bokeh balls

See `demo.py` for more details. Note that it may take some time to run if the image resolution and `K` are very large. 

-->


## BLB Dataset
The BLB dataset is synthesized by Blender 2.93. It contains 10 scenes, each consisting of an all-in-focus image, a disparity map, a stack of bokeh images with 5 blur amounts and 10 refocused disparities, and a parameter file. We additionally provide 15 corrupted disparity maps (through gaussian blur, dilation, erosion) for each scene. Our BLB dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1URpab6AXQsNTqcBcighF73w5pFlvM0Ej?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1U0XlFM_84-vVgnXGYz0ncQ?pwd=re8q).


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
