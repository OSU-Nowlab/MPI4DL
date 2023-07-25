# Installation Guide for PyTorch with MPI support

*Note : To enable MPI support, it is required to install PyTorch from source.*</br>

### Install Miniconda and activate conda environment on Linux

```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
conda create -n PyTorch_env python=3.6.5
conda activate PyTorch_env
export PYTHONNOUSERSITE=true
```

### Clone PyTorch repository
```bash
git clone --recursive https://github.com/pytorch/pytorch
git checkout v1.12.1
```

### Add cuda-aware MPI support
Modify pytorch/caffe2/mpi/mpi_ops_gpu.cc:
```bash
#define CAFFE2_HAS_CUDA_MPI_BASICS 1
#define CAFFE2_HAS_CUDA_MPI_ALLREDUCE 1
```

Modify torch/csrc/distributed/c10d/ProcessGroupMPI.cpp
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
### For more information refer PyTorch installation guide 
- https://github.com/pytorch/pytorch
