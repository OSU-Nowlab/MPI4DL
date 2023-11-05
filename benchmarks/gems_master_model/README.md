# GEMS: <u>G</u>PU-<u>E</u>nabled <u>M</u>emory-Aware Model-Parallelism <u>S</u>ystem for Distributed DNN Training
Model Parallelism is necessary for training out-of-core models; however, it can lead to the underutilization of resources. To address this limitation, Pipeline Parallelism is employed, where the batch size is set to greater than 1. But, when dealing with very high-resolution images, certain state-of-the-art models can only work with a unit batch size. GEMS is a memory-efficient design for model parallelism that enables training models with any batch size while utilizing the same resources. For more details, please refer to the original paper: [GEMS: <u>G</u>PU-<u>E</u>nabled <u>M</u>emory-Aware Model-Parallelism <u>S</u>ystem for Distributed DNN Training](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9355254).

## Run GEMS-MASTER:

#### Generic command:
```bash
$MV2_HOME/bin/mpirun_rsh --export-all -np $np --hostfile ${HOSTFILE} MV2_USE_GDRCOPY=0 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 LD_PRELOAD=$MV2_HOME/lib/libmpi.so python ${gems_model_script} --split-size ${split_size} --image-size ${image_size} --batch-size ${batch_size} --times ${times}
```
#### Examples

- Example to run AmoebaNet MASTER model for 1024 * 1024 image size with 4 model split size(i.e. # of partitions for MP), model replication factor (η = 2) and batch size for each model replica as 1 (i.e. effective batch size (EBS) = η × BS = 2).

```bash
$MV2_HOME/bin/mpirun_rsh --export-all -np $np --hostfile ${HOSTFILE} MV2_USE_GDRCOPY=0 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 LD_PRELOAD=$MV2_HOME/lib/libmpi.so python benchmarks/gems_master_model/benchmark_amoebanet_gems_master.py --split-size 4 --image-size 1024 --batch-size 1 --times 2
```
- Similarly, we can run benchmark for ResNet MASTER model.
Below is example to run ResNet MASTER model for 2048 * 2048 image size with 4 model split size(i.e. # of partitions for MP), model replication factor (η = 4) and batch size for each model replica as 1 (i.e. effective batch size (EBS) = η × BS = 4).
```bash
$MV2_HOME/bin/mpirun_rsh --export-all -np $np --hostfile ${HOSTFILE} MV2_USE_GDRCOPY=0 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 LD_PRELOAD=$MV2_HOME/lib/libmpi.so python benchmarks/gems_master_model/benchmark_resnet_gems_master.py --split-size 4 --image-size 2048 --batch-size 1 --times 4 &>> $OUTFILE 2>&1

```

Below are the available configuration options :

<pre>
usage: benchmark_amoebanet_sp.py [-h] [-v] [--batch-size BATCH_SIZE] [--parts PARTS] [--split-size SPLIT_SIZE] [--num-spatial-parts NUM_SPATIAL_PARTS]
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

  *Note:"--times" is GEMS specific parameter and certain parameters such as "--num-spatial-parts", "--slice-method", "--halo-D2" would not be required by GEMS.*
