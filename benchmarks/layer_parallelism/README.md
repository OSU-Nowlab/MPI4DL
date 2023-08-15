# Layer Parallelism Benchmarks

## Run Layer parallelism:

#### Generic command:
```bash

$MV2_HOME/bin/mpirun_rsh --export-all -np $np --hostfile  {$HOSTFILE} MV2_USE_CUDA=1 MV2_HYBRID_BINDING_POLICY=spread MV2_CPU_BINDING_POLICY=hybrid MV2_USE_GDRCOPY=0 PYTHONNOUSERSITE=true LD_PRELOAD=$MV2_HOME/lib/libmpi.so python ${lp_model_script} --image-size ${image_size} --batch-size ${batch_size} --split-size ${split_size} --parts ${parts}

```
#### Examples

- With 4 GPUs [split size: 4]

Example to run AmoebaNet LP model with 4 model split size(i.e. # of partitions for LP) for 1024 * 1024 image size. 

```bash
$MV2_HOME/bin/mpirun_rsh --export-all -np $np --hostfile ${hostfile} MV2_USE_CUDA=1 MV2_HYBRID_BINDING_POLICY=spread MV2_CPU_BINDING_POLICY=hybrid MV2_USE_GDRCOPY=0 PYTHONNOUSERSITE=true LD_PRELOAD=$MV2_HOME/lib/libmpi.so python benchmarks/layer_parallelism/benchmark_amoebanet_lp.py --batch-size 1 --image-size 1024 --split-size 4 
```

Similarly, we can run benchmark for ResNet LP model.

```bash
$MV2_HOME/bin/mpirun_rsh --export-all -np $np --hostfile ${hostfile} MV2_USE_CUDA=1 MV2_HYBRID_BINDING_POLICY=spread MV2_CPU_BINDING_POLICY=hybrid MV2_USE_GDRCOPY=0 PYTHONNOUSERSITE=true LD_PRELOAD=$MV2_HOME/lib/libmpi.so python benchmarks/layer_parallelism/benchmark_resnet_lp.py --batch-size 1 --image-size 1024 --split-size 4 
``` 

Below are the available configuration options :

<pre>
optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Prints performance numbers or logs (default: False)
  --batch-size BATCH_SIZE
                        input batch size (default: 32)
  --parts PARTS         Number of parts for MP (default: 1)
  --split-size SPLIT_SIZE
                        Number of process for MP (default: 2)
  --image-size IMAGE_SIZE
                        Image size for synthetic benchmark (default: 32)
  --num-epochs NUM_EPOCHS
                        Number of epochs (default: 1)
  --num-layers NUM_LAYERS
                        Number of layers in amoebanet (default: 18)
  --num-filters NUM_FILTERS
                        Number of layers in amoebanet (default: 416)
  --balance BALANCE     length of list equals to number of partitions and sum should be equal to num layers (default: None)
  --fused-layers FUSED_LAYERS
                        When D2 design is enables for halo exchange, number of blocks to fuse in ResNet model (default: 1)
  --local-DP LOCAL_DP   LBANN intergration of SP with MP. MP can apply data parallelism. 1: only one GPU for a given split, 2: two gpus for a given split (uses DP)
                        (default: 1)
  --app APP             Application type (1.medical, 2.cifar, and synthetic) in Spatial parallelism (default: 3)
  --datapath DATAPATH   local Dataset path (default: ./train)
  </pre>
