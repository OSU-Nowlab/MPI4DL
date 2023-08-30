## Developer Guide

### Project structure
```
├── benchmarks
│   ├── communication
│   │   └── halo
│   ├── layer_parallelism
│   └── spatial_parallelism
└── src
    ├── models
    └── torchgems
        ├── comm.py
        ├── gems_master.py
        ├── __init__.py
        ├── mp_pipeline.py
        ├── parser.py
        ├── spatial.py
        ├── train_spatial_master.py
        └── train_spatial.py
```

- `benchmarks` directory contains the main file for training models.
    - `halo-exchange`: provides benchmarks to estimate halo-exchange latency for different configurations (such as spatial_type, halo length, etc.). Refer to 
    - `layer_parallelism`: includes support for AmeobaNet and ResNet models.
    - `spatial_parallelism`: integrates support for Spatial and Layer or Pipeline parallelism. Model support include AmeobaNet and ResNet models.
- `src` directory contains all the packages required to train models.
    - `models`: contains supported model definitions and their spatial versions, e.g., `resnet.py` and `resnet_spatial.py`.
    - `torchgems`: contains packages which include definitions for both forward and backward passes, send-receive group information and managing communication between different ranks depending on paralllelism schema.
        - Example: In spatial parallelism, the convolutional layers needs to communicate with different GPUs undergoing spatial parallelism for halo exchange. Subsequent layers are required to receive input from GPUs that underwent SP in the previous layer.
        - `mp_pipeline` is specific to pipeline parallelism, while `train_spatial` is used for spatial parallelism.

## Bring your own model support
- If you would like to add support for layer and pipeline parallelism, you need to add a definition for that model in `models/`.
- For spatial parallelism support, define a model definition specific to spatial parallelism, which involves defining operations for SP. For example, ResNet spatial uses `torchgems.spatial.conv_spatial` specifically for SP. In spatial parallelism, convolution operations involve adding padding to store halo-exchange tensors received from neighboring ranks.
- For more details, refer the original paper https://doi.org/10.1007/978-3-031-07312-0_6 

## How to split the model
- Layer/Pipeline Parallism specific parameters
    - `split_size` : number of splits for model paralliem
    - `parts`: defines size for pipeline
    - `balance`: defines number of layers in each split

- Spatial Parallism specific parameters
    - `num_spatial_parts` defines number of partition for SP
    - `slice_method` defines type of SP split [square, vertical, horizontal]
    - `spatial_size` defines number of splits that will use spatial parallelism

If you encounter a 'CUDA_OUT_OF_MEMORY' error, you may want to increase the 'split_size', 'spatial_size', and/or 'num_spatial_parts'."

## Contact
For further questions, please contact The High-Performance Deep Learning (HiDL) Team
http://hidl.cse.ohio-state.edu 