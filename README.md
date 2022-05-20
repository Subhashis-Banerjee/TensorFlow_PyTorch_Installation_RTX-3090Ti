# TensorFlow_PyTorch_Installation_RTX-3090Ti
## Install CUDA, cuDNN, TensorFlow, PyTorch on RTX 30xx using Ubuntu 20.04  

## Supported versions for CUDA 11.0
TensorFlow-2.4.0 (https://www.tensorflow.org/install/source#gpu)
PyTorch v1.7.1 (https://pytorch.org/get-started/previous-versions/)

##  cuDNN v8.0.5 (November 9th, 2020), for CUDA 11.0 (provided)

## 1) Install Nvidia Driver 
Software & Updates ==> AdditionalDrivers ==> nvidia-driver-510 (proprietary, tested) 

## 2) Install gcc
sudo apt update\
sudo apt install build-essential\
sudo apt-get install manpages-dev\
sudo apt install gcc

## 3) Install CUDA Toolkit 11.0 Update 3

wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run \
sudo sh cuda_11.0.3_450.51.06_linux.run

### Only check CUDA toolKit 11.0 during the installation

## 4) Add CUDA path
nano ~/.bashrc \ 
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}$ \ 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

## 5) Install CUDNN
tar -xvf cudnn-11.0-linux-x64-v8.0.5.39.tg\
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include\
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64\
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*\

### Check installation
nvcc --version

### Install pytorch and tensorflow
#### Install virtual environment 
1) sudo apt install python3.8-venv
2) python3 -m venv venv
### Activate the virtual environment
source venv/bin/activate

check pytorch and tensorflow is picking gpu or not

import tensorflow as tf
tf.config.list_physical_devices('GPU') 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.test.gpu_device_name()

import torch
torch.cuda.is_available()
torch.cuda.get_device_name(0)
