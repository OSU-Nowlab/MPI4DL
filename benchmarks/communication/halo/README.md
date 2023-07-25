##  Halo exchnage benchmark:
- While performing convolutional operations on each partition of the image, halo exchange will be performed to receive input from neighboring partitions.
- Halo exchange can also be performed in parallel, while convolution operations on available input are done in parallel while performing halo exchange.
- spatial_halo_exchange_bench.py and spatial_halo_exchange_with_compute_bench.py are used to test the proper functioning of send and receive operations for halo regions.
- spatial_halo_exchange_with_compute_val_bench.py is utilized to validate the received inputs, in addition to testing the halo region send and receive operations.


**Run halo-exchange benchmarks:**

- Load Required model:
```bash
cd now-dl
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