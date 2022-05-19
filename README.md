# TensorFlow_PyTorch_Installation_RTX-3090Ti
## Install CUDA, cuDNN, TensorFlow, PyTorch on RTX 30xx using Ubuntu 20.04  

### Install gcc
1) sudo apt update
2) sudo apt install build-essential
3) sudo apt-get install manpages-dev
4) sudo apt install gcc

### install CUDA
1) wget https://developer.download.nvidia.com... 
2) sh cuda_11.0.3_450.51.06_linux.run

### Add CUDA path
1) nano ~/.bashrc 
2) export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}$ 
3) export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

### Install CUDNN
1) tar -xvf cudnn-11.0-linux-x64-v8.0.5.39.tg
2) sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
3) sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
4) sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

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
