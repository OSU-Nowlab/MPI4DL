import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="SP-MP-DP Configuration Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="Prints performance numbers or logs",
        action="store_true",
    )

    parser.add_argument("--batch-size", type=int, default=32, help="input batch size")

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

    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")

    parser.add_argument(
        "--num-layers", type=int, default=18, help="Number of layers in amoebanet"
    )
    parser.add_argument(
        "--num-filters", type=int, default=416, help="Number of layers in amoebanet"
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
    parser.add_argument(
        "--datapath",
        type=str,
        default="./train",
        help="local Dataset path",
    )
    return parser
