# MPI4DL

There are several approaches that have been proposed to address some of the limitations of layer parallelism. However, most studies are performed for low-resolution images that exhibit different characteristics. Compared to low-resolution images, high-resolution images (e.g. Digital pathology images) result in higher activation memory and larger tensors, which in turn lead to a larger communication overhead.

<div align="center">
 <img src="docs/assets/images/DP_MP_SP_Vs_Memory.png" width="600px">
 <br>
 <figcaption>Figure 1. Capabilities of each parallelism scheme for low-resolution, high-resoution and very high resolutioin images. 
</figcaption>

<br>
</div>
<br>

Figure 1. shows capabilities of each parallelism scheme with respective to diferent image sizes. Data parallelism has a memory limitation and cannot be performed for out-of-core models. Layer parallelism overcomes the limitation of data parallelism by distributing the model across different GPUs. However, it causes GPU underutilization as only one GPU is utilized. Pipeline parallelism accelerates the performance of layer parallelism by training the model in a pipeline fashion. However, pipeline parallelism is only possible when the model is trainable with a batch size > 1, which is typically impossible with high-resolution images due to memory constraints. To train high-resolution images, spatial parallelism can be used, which distributes images across multiple GPUs. On the other hand, it has performance issues due to high communication overhead and the inability to accelerate low-resolution images that are common in the latter half of DNNs.



**Our objective is efficiently utilizing distributed training for very high-resolution images that appear in real-world applications. Integrating spatial and layer parallelism can solve the aforementioned limitations of spatial parallelism and layer parallelism. Spatial parallelism enables training high-resolution images efficiently even when the model size is large, and layer parallelism accelerates low-resolution images in the latter half of DNNs. This schema enables training high-resolution images efficiently. This project is a PyTorch implementation of this technique and is based on [Hy-Fi: Hybrid Five-Dimensional Parallel DNN Training on High-Performance GPU Clusters](https://dl.acm.org/doi/abs/10.1007/978-3-031-07312-0_6).**

# Background

## Layer Parallelism: 
Layer parallelism distributes the DNN model on separate GPUs before applying distributed forward and backward passes. These distributed forward and backward passes are implemented with simple Send and Recv operations. Thus, layer parallelism suffers from under-utilization of resources and scalability, as only a single GPU can operate at once.

## Pipeline Parallelism
Pipelining divides the input batch into smaller batches called micro-batches, the number of which we call parts. The goal of pipeline parallelism is to reduce underutilization by overlapping micro-batches, which allows multiple GPUs to proceed with computation within the forward and backward passes.
## Spatial Parallelism:

In spatial parallelism, the convolution layer is replicated across multiple GPUs, and image parts are partitioned across replicas. Convolution and Pooling layers can be distributed across multiple GPUs to work on different regions of the image. Hence, unlike layer parallelism, this approach enables simultaneous computation on multiple GPUs while facilitating the training of the out-of-core convolution layer, but it requires extra communication to receive border pixels from neighboring partitions, also called halo-exchange. Refer [Halo exchnage](benchmarks/communication/halo) for more information.

## Spatial Parallelism + Layer Parallelism
<div align="center">
 <img src="docs/assets/images/Spatial_Parallelism.jpg" width="600px">
 </br>
 <figcaption>Figure 2. Combination of spatial and layer parallelism. </figcaption>
    </br>
</div>

Above figure shows combination of spatial and layer parallelism. In this approach, the model is divided into 4 partitions, and spatial parallelism is used for the first partition to perform convolution operations on the input image. The second layer aggregates the output from the first layer and then sends it, while lateral layers use layer parallelism.

Due to the increased communication overhead, spatial parallelism is more suitable for large images, which makes this approach inappropriate for the latter half of CNNs where the image input size usually consists of few pixels. Layer parallelism can be used to compute this latter half. Figure 2 shows a combination of spatial parallelism and layer parallelism for a CNN partitioned into four partitions at the layer granularity. Spatial parallelism is applied to the first model partition, and layer parallelism is applied to the other three model partitions.

Refer [Spatial Parallelism](benchmarks/spatial_parallelism) for more details.

## Installation:

### Prerequisite:
- Python 3.8 or later (for Linux, Python 3.8.1+ is needed).
- MVAPICH2
Refer [MVAPICH2 installation guide](docs/installation/MVAPICH_INSTALLATION_GUIDE.md) to install MVAPICH2.
- PyTorch :  1.12.1 or 1.13.1
Refer [PyTorch installation guide](/docs/installation/PYTORCH_INSTALLATION_GUIDE.md) to install PyTorch from source and configure MVAPICH2 support. 

*Note:
We used the following versions during implementation and testing.
Python=3.9.16, cuda=11.6, gcc=10.3.0, cmake=3.22.2, PyTorch=1.12.1, MVAPICH2-GDR=2.3.7*

### Install mpi4dl
```bash
cd mpi4dl
python setup.py install
```
### Run model benchmark:
Example to run AmoebaNet model with partition size for model as two, spatial partition as four and spatial size (i.e. number of model partition which will use spatial partition) as 1
```bash
$MV2_HOME/bin/mpirun_rsh --export-all -np 5 --hostfile {$HOSTFILE} MV2_USE_CUDA=1 MV2_HYBRID_BINDING_POLICY=spread MV2_CPU_BINDING_POLICY=hybrid MV2_USE_GDRCOPY=0 PYTHONNOUSERSITE=true LD_PRELOAD=$MV2_HOME/lib/libmpi.so python benchmarks/spatial_parallelism/benchmark_amoebanet_sp.py --image-size 512 --num-spatial-parts 4 --slice-method "vertical" --split-size 2 --spatial-size 1
```

Refer [Spatial Parallelism](benchmarks/spatial_parallelism) and [Halo Exchange](benchmarks/communication/halo) for more spatial benchmarks.

## References:
1. Arpan Jain, Ammar Ahmad Awan, Asmaa M. Aljuhani, Jahanzeb Maqbool Hashmi, Quentin G. Anthony, Hari Subramoni, Dhableswar K. Panda, Raghu Machiraju, and Anil Parwani. 2020. GEMS: <u>G</u>PU-<u>e</u>nabled <u>m</u>emory-aware model-parallelism <u>s</u>ystem for distributed DNN training. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '20). IEEE Press, Article 45, 1–15.
2. Arpan Jain, Aamir Shafi, Quentin Anthony, Pouya Kousha, Hari Subramoni, and Dhableswar K. Panda. 2022. Hy-Fi: Hybrid Five-Dimensional Parallel DNN Training on High-Performance GPU Clusters. In High Performance Computing: 37th International Conference, ISC High Performance 2022, Hamburg, Germany, May 29 – June 2, 2022, Proceedings. Springer-Verlag, Berlin, Heidelberg, 109–130. https://doi.org/10.1007/978-3-031-07312-0_6

