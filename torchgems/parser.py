import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="MP-DP ResNet Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="Prints performance numbers or logs",
        action="store_true",
    )

    parser.add_argument(
        "--fp16-allreduce",
        action="store_true",
        default=False,
        help="use fp16 compression during allreduce",
    )

    parser.add_argument(
        "--model", type=str, default="resnet50", help="model to benchmark"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="input batch size")

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="learning rate for the optimizer",
    )
    parser.add_argument(
        "--num-gpus-mp", type=int, default=1, help="number of GPUS per node for MP"
    )
    parser.add_argument(
        "--mem-per-process", type=float, default=1, help="TF GPU memory per GPU"
    )

    parser.add_argument("--parts", type=int, default=1, help="Number of parts for MP")

    parser.add_argument(
        "--split-size", type=int, default=2, help="Number of process for MP"
    )
    parser.add_argument(
        "--num-spatial-parts",
        type=str,
        default="4",
        help="Number of partitions in spatial parallelism",
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=1,
        help="Number splits for spatial parallelism",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=1,
        help="Number of times to repeat MASTER 1: 2 repications, 2: 4 replications",
    )
    parser.add_argument(
        "--image-size", type=int, default=32, help="Image size for synthetic benchmark"
    )

    parser.add_argument(
        "--dp-per-node", type=int, default=1, help="Number of DP modes per node"
    )

    parser.add_argument(
        "--enable-dp",
        dest="enable_dp",
        action="store_true",
        help="Enable DP for pytorch scripts",
    )

    parser.add_argument(
        "--enable-master-comm-opt",
        dest="enable_master_comm_opt",
        action="store_true",
        default=False,
        help="Enable communication optimization for MASTER in Spatial",
    )

    parser.add_argument(
        "--num-gpu-per-node", type=int, default=4, help="Number of GPUs per node"
    )

    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")

    parser.add_argument(
        "--num-layers", type=int, default=18, help="Number of layers in amoebanet"
    )
    parser.add_argument(
        "--num-filters", type=int, default=416, help="Number of layers in amoebanet"
    )
    parser.add_argument("--unet-b", type=int, default=6, help="B hyperparamter in unet")
    parser.add_argument(
        "--unet-c", type=int, default=72, help="C hyperparamter in unet"
    )
    parser.add_argument(
        "--balance",
        type=str,
        default=None,
        help="length of list equals to number of partitions and sum should be equal to num layers",
    )
    parser.add_argument(
        "--halo-D2",
        dest="halo_d2",
        action="store_true",
        default=False,
        help="Enable design2 (do halo exhange on few convs) for spatial conv. ",
    )
    parser.add_argument(
        "--fused-layers",
        type=int,
        default=1,
        help="When D2 design is enables for halo exchange, number of blocks to fuse in ResNet model ",
    )
    parser.add_argument(
        "--local-DP",
        type=int,
        default=1,
        help="LBANN intergration of SP with MP. MP can apply data parallelism. 1: only one GPU for a given split, 2: two gpus for a given split (uses DP)",
    )
    parser.add_argument(
        "--slice-method",
        type=str,
        default="square",
        help="Slice method (square, vertical, and horizontal) in Spatial parallelism",
    )
    parser.add_argument(
        "--app",
        type=int,
        default=3,
        help="Application type (1.medical, 2.cifar, and synthetic) in Spatial parallelism",
    )

    parser.set_defaults(enable_dp=False)
    return parser
