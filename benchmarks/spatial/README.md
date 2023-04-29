# Spatial Parallelism Benchmarks

Spatial parallelism benchmarks include halo exchange and model benchmarks. These benchmarks will test the working of spatial parallelism.


##  Halo exchnage benchmark:
- While performing convolutional operations on each partition of the image, halo exchange will be performed to receive input from neighboring partitions.
- Halo exchange can also be performed in parallel, while convolution operations on available input are done in parallel while performing halo exchange.
- spatial_halo_exchange_bench.py and spatial_halo_exchange_with_compute_bench.py are used to test the proper functioning of send and receive operations for halo regions.
- spatial_halo_exchange_with_compute_val_bench.py is utilized to validate the received inputs, in addition to testing the halo region send and receive operations.


**Run halo-exchange benchmarks:**

- Load Required model:
```bash
cd torch-gems
python setup.py install
```

- Example to run halo exchange benchmark for four vertical partition : 
```bash
cd benchmarks/spatial/model/
$MV2_HOME/bin/mpirun_rsh --export-all -np 4 --hostfile {$HOSTFILE} MV2_USE_GDRCOPY=0 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 LD_PRELOAD=$MV2_HOME/lib/libmpi.so python spatial_halo_exchange_bench.py --image-size 32 --batch-size 32 --num-spatial-parts 4 --slice-method "vertical"
```

Halo exchange benchmarks can also be configured for different num-spatial-parts, slice-method, etc. Find all available options below:
<pre>
usage: spatial_halo_exchange_bench.py [-h] [--fp16-allreduce] [--image-size IMAGE_SIZE] [--batch-size BATCH_SIZE] [--halo-len HALO_LEN] [--in-channels IN_CHANNELS]
                                      [--warmup WARMUP] [--iterations ITERATIONS] [--out-channels OUT_CHANNELS]

Halo exchange benchmark

optional arguments:
  -h, --help            show this help message and exit
  --fp16-allreduce      use fp16 compression during allreduce (default: False)
  --image-size IMAGE_SIZE
                        Full image size (default: 8)
  --batch-size BATCH_SIZE
                        input batch size (default: 1)
  --halo-len HALO_LEN   halo length (default: 1)
  --in-channels IN_CHANNELS
                        Number of channels in the input (default: 1)
  --warmup WARMUP       warmups (default: 10)
  --iterations ITERATIONS
                        Iterations (default: 100)
  --out-channels OUT_CHANNELS
                        number of output channels (default: 256)
</pre>

## Model benchmarks

Model benchmarks for spatial parallelism also require performing model parallelism. To configure the number of model partitions and the number of model partitions that will use spatial parallelism, you can use the --split-size and --spatial-size arguments respectively.

1. Amoebanet benchmark

Run spatial parallelism for Amoebanet model:

Example to run Amoebanet model with partition size for model as two, spatial partition as four and spatial size (i.e. number of model partition which will use spatial partition) as 1
```bash
$MV2_HOME/bin/mpirun_rsh --export-all -np 5 --hostfile {$HOSTFILE} MV2_USE_GDRCOPY=0 MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 LD_PRELOAD=$MV2_HOME/lib/libmpi.so python amoebanet_run.py --image-size 512 --num-spatial-parts 4 --slice-method "vertical" --split-size 2 --spatial-size 1
```

Below are the available configuration options :

<pre>
usage: amoebanet_run.py [-h] [--fp16-allreduce] [--model MODEL] [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE] [--num-gpus-mp NUM_GPUS_MP]
                        [--mem-per-process MEM_PER_PROCESS] [--parts PARTS] [--split-size SPLIT_SIZE] [--num-spatial-parts NUM_SPATIAL_PARTS]
                        [--spatial-size SPATIAL_SIZE] [--times TIMES] [--image-size IMAGE_SIZE] [--dp-per-node DP_PER_NODE] [--enable-dp] [--enable-master-comm-opt]
                        [--num-gpu-per-node NUM_GPU_PER_NODE] [--num-epochs NUM_EPOCHS] [--num-layers NUM_LAYERS] [--num-filters NUM_FILTERS] [--unet-b UNET_B]
                        [--unet-c UNET_C] [--balance BALANCE] [--halo-D2] [--fused-layers FUSED_LAYERS] [--local-DP LOCAL_DP] [--slice-method SLICE_METHOD]

MP-DP ResNet Script

optional arguments:
  -h, --help            show this help message and exit
  --fp16-allreduce      use fp16 compression during allreduce (default: False)
  --model MODEL         model to benchmark (default: resnet50)
  --batch-size BATCH_SIZE
                        input batch size (default: 32)
  --learning-rate LEARNING_RATE
                        learning rate for the optimizer (default: 0.001)
  --num-gpus-mp NUM_GPUS_MP
                        number of GPUS per node for MP (default: 1)
  --mem-per-process MEM_PER_PROCESS
                        TF GPU memory per GPU (default: 1)
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
  --dp-per-node DP_PER_NODE
                        Number of DP modes per node (default: 1)
  --enable-dp           Enable DP for pytorch scripts (default: False)
  --enable-master-comm-opt
                        Enable communication optimization for MASTER in Spatial (default: False)
  --num-gpu-per-node NUM_GPU_PER_NODE
                        Number of GPUs per node (default: 4)
  --num-epochs NUM_EPOCHS
                        Number of epochs (default: 1)
  --num-layers NUM_LAYERS
                        Number of layers in amoebanet (default: 18)
  --num-filters NUM_FILTERS
                        Number of layers in amoebanet (default: 416)
  --unet-b UNET_B       B hyperparamter in unet (default: 6)
  --unet-c UNET_C       C hyperparamter in unet (default: 72)
  --balance BALANCE     length of list equals to number of partitions and sum should be equal to num layers (default: None)
  --halo-D2             Enable design2 (do halo exhange on few convs) for spatial conv. (default: False)
  --fused-layers FUSED_LAYERS
                        When D2 design is enables for halo exchange, number of blocks to fuse in ResNet model (default: 1)
  --local-DP LOCAL_DP   LBANN intergration of SP with MP. MP can apply data parallelism. 1: only one GPU for a given split, 2: two gpus for a given split (uses DP)
                        (default: 1)
  --slice-method SLICE_METHOD
                        Slice method (square, vertical, and horizontal) in Spatial parallelism (default: square)

</pre>
