# 《DMM: Disparity-guided Multispectral Mamba for Oriented Object Detection in Remote Sensing》

The official implementation of the [paper](https://arxiv.org/abs/2407.08132).

## Dataset
1. Download the dataset from the repository https://github.com/VisDrone/DroneVehicle, then run the following code to crop the white borders:
```shell
python tools/data_process.py
```

2. Run the following code to process the labels (since the original labels for the "freight-car" category are inconsistent and contain errors such as "*", we have unified them to "freight-car" in the code):
```shell
python tools/VOC2DOTA.py
```


pretrained weights: [BaiduYun](https://pan.baidu.com/s/1XdKKjrGseeM5_JSKfxfA1A?pwd=jwqx) \[code: jwqx\]


## Envirenment
CUDA==11.8

Pytorch==2.1.2

mmcv==2.1.0

mmdet==3.3.0

mmengine==0.10.5

numpy==1.26.4

You can follow the steps below to create an virtual environment:

1. install all dependencies:
```
conda create -n dmm python=3.10
conda activate dmm

conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -U openmim
mim install mmdet
pip install numpy==1.26.4
```

- You might encounter the following, Downgrade the pip version to 24.0 (pip install pip==24.0)
```
Ignoring mmcv: markers 'extra == "mim"' don't match your environment
Ignoring mmengine: markers 'extra == "mim"' don't match your environment
```


2. Follow the https://github.com/MzeroMiko/VMamba Getting Started Step 2, install selective_scan==0.0.2


3. Clone the code and install:
```
git clone https://github.com/Another-0/DMM
cd DMM
pip install -v -e .
```

## Run

1. train
```
python ./tools/train.py ${CONFIG_FILE} 
```

2. test
```
python ./tools/test.py ${CONFIG_FILE} ${CHECKPOINT}
```


For more command-line arguments, please refer to the code details.

## Acknowledgment
Our codes are mainly based on [MMRotate](https://github.com/open-mmlab/mmrotate) and [VMamba](https://github.com/MzeroMiko/VMamba). Many thanks to the authors!

## Citation
Please cite our work if you find our work and codes helpful for your research.
```
@article{zhou2024dmm,
  title={DMM: Disparity-guided Multispectral Mamba for Oriented Object Detection in Remote Sensing},
  author={Zhou, Minghang and Li, Tianyu and Qiao, Chaofan and Xie, Dongyu and Wang, Guoqing and Ruan, Ningjuan and Mei, Lin and Yang, Yang},
  journal={arXiv preprint arXiv:2407.08132},
  year={2024}
}
```