# Spatial Parallelism Benchmarks

Model benchmarks for spatial parallelism also require performing model parallelism. To configure the number of model partitions and the number of model partitions that will use spatial parallelism, you can use the --split-size and --spatial-size arguments respectively.

## Run spatial parallelism:

#### Generic command:
```bash

$MV2_HOME/bin/mpirun_rsh --export-all -np $np --hostfile  {$HOSTFILE} MV2_USE_GDRCOPY=0 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 LD_PRELOAD=$MV2_HOME/lib/libmpi.so python ${model_type} --halo-D2 --num-spatial-parts ${num_spatial_parts}  --image-size ${image_size} --batch-size ${batch_size} --slice-method ${partition}

```
#### Examples

- With 5 GPUs [split size: 2, num_spatial_parts: 4, spatial_size: 1]

Example to run AmoebaNet model with 2 model split size(i.e. # of partitions for MP), spatial partition (# of image partitions) as 4 and 1 as spatial size (i.e. number of model partition which will use spatial partition). In this configuration, we split model into two parts where first part will use spatial parallelism. 

```bash
$MV2_HOME/bin/mpirun_rsh --export-all -np 5 --hostfile {$HOSTFILE} MV2_USE_GDRCOPY=0 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 LD_PRELOAD=$MV2_HOME/lib/libmpi.so python now-dl/benchmarks/spatial_parallelism/benchmark_amoebanet_sp.py --image-size 512 --num-spatial-parts 4 --slice-method "vertical" --split-size 2 --spatial-size 1
```
- With 9 GPUs [split size: 3, num_spatial_parts: 4, spatial_size: 2]
In this configuration, we split model int three parts where first two part will use spatial parallelism. 

```bash
$MV2_HOME/bin/mpirun_rsh --export-all -np 9 --hostfile {$HOSTFILE} MV2_USE_GDRCOPY=0 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 LD_PRELOAD=$MV2_HOME/lib/libmpi.so python now-dl/benchmarks/spatial_parallelism/benchmark_amoebanet_sp.py --image-size 512 --num-spatial-parts 4 --slice-method "vertical" --split-size 3 --spatial-size 2
```

- Similarly, we can run benchmark for ResNet model.
Find the example to run ResNet with halo-D2 enabled to reduce communication opertaions. To learn more about halo-D2, refer [Hy-Fi: Hybrid Five-Dimensional Parallel DNN Training on High-Performance GPU Clusters](https://dl.acm.org/doi/abs/10.1007/978-3-031-07312-0_6)
```bash
$MV2_HOME/bin/mpirun_rsh --export-all -np 5 --hostfile {$HOSTFILE} MV2_USE_GDRCOPY=0 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 LD_PRELOAD=$MV2_HOME/lib/libmpi.so now-dl/benchmarks/spatial_parallelism/benchmark_resnet_sp.py --halo-D2 --num-spatial-parts 4 --image-size 1024 --batch-size 2 --slice-method "square"
``` 

Below are the available configuration options :

<pre>
usage: amoebanet_run.py [-h] [-v] [--batch-size BATCH_SIZE] [--parts PARTS] [--split-size SPLIT_SIZE] [--num-spatial-parts NUM_SPATIAL_PARTS]
                        [--spatial-size SPATIAL_SIZE] [--times TIMES] [--image-size IMAGE_SIZE] [--num-epochs NUM_EPOCHS] [--num-layers NUM_LAYERS]
                        [--num-filters NUM_FILTERS] [--balance BALANCE] [--halo-D2] [--fused-layers FUSED_LAYERS] [--local-DP LOCAL_DP] [--slice-method SLICE_METHOD]
                        [--app APP] [--datapath DATAPATH]

SP-MP-DP Configuration Script

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Prints performance numbers or logs (default: False)
  --batch-size BATCH_SIZE
                        input batch size (default: 32)
  --parts PARTS         Number of parts for MP (default: 1)
  --split-size SPLIT_SIZE
                        Number of process for MP (default: 2)
  --num-spatial-parts NUM_SPATIAL_PARTS
                        Number of partitions in spatial parallelism (default: 4)
  --spatial-size SPATIAL_SIZE
                        Number splits for spatial parallelism (default: 1)
  --times TIMES         Number of times to repeat MASTER 1: 2 repications, 2: 4 replications (default: 1)
  --image-size IMAGE_SIZE
                        Image size for synthetic benchmark (default: 32)
  --num-epochs NUM_EPOCHS
                        Number of epochs (default: 1)
  --num-layers NUM_LAYERS
                        Number of layers in amoebanet (default: 18)
  --num-filters NUM_FILTERS
                        Number of layers in amoebanet (default: 416)
  --balance BALANCE     length of list equals to number of partitions and sum should be equal to num layers (default: None)
  --halo-D2             Enable design2 (do halo exhange on few convs) for spatial conv. (default: False)
  --fused-layers FUSED_LAYERS
                        When D2 design is enables for halo exchange, number of blocks to fuse in ResNet model (default: 1)
  --local-DP LOCAL_DP   LBANN intergration of SP with MP. MP can apply data parallelism. 1: only one GPU for a given split, 2: two gpus for a given split (uses DP)
                        (default: 1)
  --slice-method SLICE_METHOD
                        Slice method (square, vertical, and horizontal) in Spatial parallelism (default: square)
  --app APP             Application type (1.medical, 2.cifar, and synthetic) in Spatial parallelism (default: 3)
  --datapath DATAPATH   local Dataset path (default: ./train)
  </pre>
