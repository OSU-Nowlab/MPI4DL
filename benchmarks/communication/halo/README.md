#  Halo exchnage:
In spatial parallelism, Convolution and Pooling layers can be distributed across multiple GPUs to work on the different regions of the image. Thus, spatial parallelism requires a halo exchange (shown in Figure. 1.) at every convolution and pooling layer to compute the result for the pixels present on the boundary of image parts. Halo exchange can also be performed in parallel with convolution operations on available input pixels.
<div align="center">
 <img src="../../../docs/assets/images/Halo_Exchange.jpg" width="600px">
 </br>
  <figcaption>Figure 1. Halo exchange in spatial parallelism. The input image is partitioned into four regions, and each region is given to the different processes. To calculate the convolution operation at X location, the value of nearby pixels is required. 
  </figcaption>
    </br>
</div>


## halo-exchange benchmarks:
- *benchmark_sp_halo_exchange.py* and *benchmark_sp_halo_exchange_with_compute.py* are used to test the proper functioning of send and receive operations for halo regions.
- *benchmark_sp_halo_exchange_with_compute_val.py* is utilized to validate the received inputs, in addition to testing the halo region send and receive operations.
### Run halo-exchange benchmark

#### Generic command:
```bash

$MV2_HOME/bin/mpirun_rsh --export-all -np $np --hostfile  {$HOSTFILE} MV2_USE_CUDA=1 MV2_HYBRID_BINDING_POLICY=spread MV2_CPU_BINDING_POLICY=hybrid MV2_USE_GDRCOPY=0 PYTHONNOUSERSITE=true LD_PRELOAD=$MV2_HOME/lib/libmpi.so python ${halo_benchmark} --image-size ${image_size} --batch-size ${batch_size} --num-spatial-parts ${num_spatial_parts} --slice-method ${partition}

```
#### Example:
Example to run halo exchange benchmark for 4 vertical partition of 1024 * 1024 image with halo-len and batch size of 3 and 1 respectively: 
```bash
$MV2_HOME/bin/mpirun_rsh --export-all -np 4 --hostfile {$HOSTFILE} MV2_USE_CUDA=1 MV2_HYBRID_BINDING_POLICY=spread MV2_CPU_BINDING_POLICY=hybrid MV2_USE_GDRCOPY=0 PYTHONNOUSERSITE=true LD_PRELOAD=$MV2_HOME/lib/libmpi.so python benchmarks/communication/halo/benchmark_sp_halo_exchange.py --image-size 1024 --batch-size 1 --halo-len 3 --num-spatial-parts 4 --slice-method "vertical"
```

Expected output:
```
rank : 0 size:  4
Rank:0 Time taken (ms):0.3337113571166992
Validation passed Rank:0
rank : 3 size:  4
Rank:3 Time taken (ms):0.3339980697631836
Validation passed Rank:3
rank : 2 size:  4
Rank:2 Time taken (ms):0.33376255035400393
Validation passed Rank:2
rank : 1 size:  4
Rank:1 Time taken (ms):0.33356800079345705
Validation passed Rank:1
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
