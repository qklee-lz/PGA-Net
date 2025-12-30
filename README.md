# PGA-Net: Polynomial Global Attention Network with Mean Curvature Loss for Lane Detection

- ### Related works
<img src=./figures/related_work.png width="400x">

- ### Overview of our PGA-Net
<img src=./figures/overflow.png>

---
## 1.HARDWARE & SOFTWARE

Ubuntu 20.04

CPU: 12700k

GPU: 1 * 3090, 24G

Python: 3.8.0

Pytorch: 1.13.1+cu113


## 2.Installation
### Clone this repository

```Shell
git clone https://github.com/qklee-lz/PGA-Net.git
```


### Create a conda virtual environment
```Shell
conda create -n PGA-Net python=3.8 -y
conda activate PGA-Net
```
### Install dependencies
```Shell
# install pytorch
pip install torch==1.13.0 torchvision==0.12.0
# install python packages
python setup.py build develop
```

## 3.Data preparation
### CULane

Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `$CULANEROOT`. Create link to `data` directory.

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, you should have structure like this:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```

### Tusimple
Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `$TUSIMPLEROOT`. Create link to `data` directory.

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```

For Tusimple, you should have structure like this:
```
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x4
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
$TUSIMPLEROOT/test_label.json # test label json file

```

For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

```Shell
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
# this will generate seg_label directory
```

### LLAMAS
Download [LLAMAS](https://unsupervised-llamas.com/llamas/)
[Official Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVRSUAD/Behrendt_Unsupervised_Labeled_Lane_Markers_Using_Maps_ICCVW_2019_paper.pdf) 


## 4.Start
### Train
For training, run
```Shell
python train.py [configs/config_file] --gpus [gpu_ids]
```
For example, run
```Shell
python train.py configs/PGA_Net_ResNet34_culane.py --gpus 0
```

### Test
For testing, run
```Shell
python test.py [configs/config_file]  --load_from [pretrained_file --gpus [gpu_ids]
```
For example, run
```Shell
python test.py configs/PGA_Net_ResNet34_culane.py --load_from checkpoint/PGA_Net_ResNet34_best.pth --gpus 0
```


### Inference
To run inference on example images in ./images and save the visualization images in vis folder:
```Shell
python inference.py [configs/config_file] --img images --load_from [pretrained_file] --savedir ./vis --gpus [gpu_ids]
```


## 5.Demo
- ### Result on three benchmark
    - Our method achieves **state-of-the-art** performance on two popular benchmarks (TuSimple and LLAMAS) and a most challenging benchmark (CULane) with faster speed (**>140fps**) and lightweight model size (**<3M**).
    - Visualization results on TuSimple (the first row), CULane (the middle row), and LLAMAS (the last row) datasets. Different lane instances are represented by different colors.
    <img src=./demo/results.png>

- ### Grad-CAM with our PGA-Net
    - Grad-CAM visualization on the last layer of the ResNet backbone and the transformer encoder. (a) Visualization on Tusimple. (b) Visualization on CULane. (c) Visualization on LLAMAS. The encoder aggregates a lot of contextual information and constrains the attention features to the lane road even the lane with occluded.
    <img src=./demo/gramcam.png>

- ### PGANet infers lane lines from **real vehicle cameras**
The PGA-Net used in the following inference is only trained on the CULane dataset.

- Complex road conditions
    - <img src=./demo/complex_road.GIF width="360px">

- Simple road conditions
    - <img src=./demo/simple_road.GIF width="360px">


## 6.Codes 
This repository is built upon several open-source projects, including 
[DETR](https://github.com/facebookresearch/detr), 
[LSTR](https://github.com/liuruijin17/LSTR), 
and [PolyLaneNet](https://github.com/lucastabelini/PolyLaneNet).

The full system was developed on industrial and internship servers, and thus this repository does not contain a fully consolidated version of the entire project. Nevertheless, we add the proposed core **Mean Curvature Loss** is included in `./code/mean_curvature_loss.py`.

## 7. Citation
```

@article{li2023pga,
  title={PGA-Net: Polynomial global attention network with mean curvature loss for lane detection},
  author={Li, Qiankun and Yu, Xianwang and Chen, Junxin and He, Ben-Guo and Wang, Wei and Rawat, Danda B and Lyu, Zhihan},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={25},
  number={1},
  pages={417--429},
  year={2023},
  publisher={IEEE}
}

```