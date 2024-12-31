# DMM
The code will be released soon.


## Dataset



## Envirenment
CUDA==11.6

Pytorch==1.13.1

mmcv==2.0.1

mmdet==3.3.0

mmengine==0.10.4

numpy==1.26.4

You can follow the steps below to create an virtual environment:

1. install all dependencies:
```
conda create -n dmm python=3.10
conda activate dmm

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

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

## Train
```
python ./tools/train.py ${CONFIG_FILE} 
```

## test
```
python ./tools/test.py ${CONFIG_FILE} ${CHECKPOINT}
```

## Result

### Result on DroneVehicle test


## Acknowledgment
Our codes are mainly based on [MMRotate](https://github.com/open-mmlab/mmrotate) and [VMamba](https://github.com/MzeroMiko/VMamba). Many thanks to the authors!

## Citation
Please cite our work if you find our work and codes helpful for your research.