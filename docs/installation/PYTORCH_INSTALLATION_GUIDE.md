# Installation Guide for PyTorch with MPI support

*Note : To enable MPI support, it is required to install PyTorch from source.*</br>

## Install PyTorch from source
### Install Miniconda and activate conda environment on Linux

```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
conda create -n PyTorch_env python=3.9.16
conda activate PyTorch_env
export PYTHONNOUSERSITE=true
```

### Clone PyTorch repository
```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
git checkout v1.12.1
```

### Add cuda-aware MPI support
Modify pytorch/caffe2/mpi/mpi_ops_gpu.cc:
```bash
#define CAFFE2_HAS_CUDA_MPI_BASICS 1
#define CAFFE2_HAS_CUDA_MPI_ALLREDUCE 1
```

Modify pytorch/torch/csrc/distributed/c10d/ProcessGroupMPI.cpp
```bash
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (MPIX_Query_cuda_support() == 1) {
    return true;
  } else {
    return true;
  }
#else // !defined(MPIX_CUDA_AWARE_SUPPORT)
  return true;
#endif // MPIX_CUDA_AWARE_SUPPORT
}
```

#### Create a different branch with MPI support changes and commit the changes

```bash
git checkout -b v1.12.1-cudaMPI
git add .
git commit -m "Support for CUDA-aware MPI"
```

### Install Dependencies
```bash
conda install pytorch dependencies 
conda install astunparse numpy ninja pyyaml setuptools cmake typing_extensions six requests dataclasses
conda install mkl mkl-include

```
### Set environment variable 
```bash
export CUDA_HOME=/opt/cuda/$CUDA_VERSION
export CPATH=$CUDA_HOME/include:$CPATH
export CUDNN_LIB_DIR=/home/gulhane.2/cuda/lib64
export CUDNN_INCLUDE_DIR=/home/gulhane.2/cuda/include
```

### Install PyTorch
```bash
git submodule sync
git submodule update --init --recursive
python setup.py develop
```
### For more information refer PyTorch installation guide 
- https://github.com/pytorch/pytorch


## Install Torchvision from source
When we install PyTorch from source, torchvision package doesn't come up with PyTorch. Thus, we need to install torchvision seperately.

### Clone repo
https://github.com/pytorch/vision.

### Clone Torchvision repository
```bash
git clone https://github.com/pytorch/vision
```

### Checkout appropriate branch

*Note : torchvision versioin should be compatible with PyTorch version. Refer https://github.com/pytorch/vision#installation to get torchvision version corresponding to PyTorch version*
```bash
cd pytorch
git checkout v0.13.0
```

### Install Torchvisioin

```bash
python setup.py install
```
### For more information refer Torchvision installation guide 
- https://github.com/pytorch/vision
