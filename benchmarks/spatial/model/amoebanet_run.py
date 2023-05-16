import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import numpy as np
import time
import sys
import os
import math
import logging
from collections import OrderedDict
from torchgems import parser
import time
from torchgems.mp_pipeline import model_generator, train_model
from torchgems.gems_master import train_model_master
from torchgems.train_spatial import train_model_spatial
import torchgems.comm as gems_comm
from torchgems.spatial_new import conv_spatial

parser_obj = parser.get_parser()
args = parser_obj.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG)

if args.halo_d2:
    from models import amoebanet
    from models import amoebanet_d2

else:
    from models import amoebanet

gems_comm.initialize_cuda()

from torch.nn.parallel import DistributedDataParallel as DDP


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def init_processes(backend="tcp"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend)
    size = dist.get_world_size()
    rank = dist.get_rank()
    return size, rank


def get_depth(version, n):
    if version == 1:
        return n * 6 + 2
    elif version == 2:
        return n * 9 + 2


sys.stdout = Unbuffered(sys.stdout)

np.random.seed(seed=1405)
parts = args.parts
batch_size = args.batch_size
resnet_n = 12
epoch = args.num_epochs
ENABLE_ASYNC = True

# APP
# 1: Medical
# 2: Cifar
# 3: synthetic
APP = args.app
amoebanet_test = False
image_size = int(args.image_size)
print("image size", image_size)
steps = 100
num_layers = args.num_layers
num_filters = args.num_filters
balance = args.balance
split_size = args.split_size
spatial_size = args.spatial_size

temp_num_spatial_parts = args.num_spatial_parts.split(",")

if len(temp_num_spatial_parts) == 1:
    num_spatial_parts_list = [int(temp_num_spatial_parts[0])]
    num_spatial_parts = int(temp_num_spatial_parts[0])
else:
    num_spatial_parts = [int(i) for i in temp_num_spatial_parts]
    num_spatial_parts_list = num_spatial_parts

times = 1
num_classes = 1000
LOCAL_DP_LP = args.local_DP

# DDP support


def isPowerTwo(num):
    return not (num & (num - 1))


"""
For Amoebanet model, image size and image size after partitioning should be power of two. 
As, Amoebanet performs summation of results of two convolution layers during training, 
odd input size(i.e. image size which is not power of 2) will give different output sizes 
for convolution operations present at same layer, thus it will throw error as addition 
operation can not be performed with diffent size outputs. 
"""


def verify_config():
    assert args.slice_method in [
        "square",
        "vertical",
        "horizontal",
    ], "Possible slice methods are ['square', 'vertical', 'horizontal']"

    assert args.app in range(
        1, 4
    ), "Possible Application values should be 1, 2, or 3 i.e. 1.medical, 2.cifar, and 3.synthetic"

    assert isPowerTwo(int(image_size)), "Image size should be power of Two"

    if args.slice_method == "square":
        assert isPowerTwo(
            int(image_size / math.sqrt(num_spatial_parts))
        ), "Image size of each partition should be power of Two"
    else:
        assert isPowerTwo(
            int(image_size / num_spatial_parts)
        ), "Image size of each partition should be power of Two"

verify_config()


mpi_comm = gems_comm.MPIComm(
    split_size=split_size,
    ENABLE_MASTER=False,
    ENABLE_SPATIAL=True,
    num_spatial_parts=num_spatial_parts,
    spatial_size=spatial_size,
    LOCAL_DP_LP=LOCAL_DP_LP,
)
sync_allreduce = gems_comm.SyncAllreduce(mpi_comm)

rank = mpi_comm.rank
comm_size = mpi_comm.size
local_rank = rank

split_rank = mpi_comm.split_rank


if args.balance != None:
    balance = args.balance.split(",")
    balance = [int(j) for j in balance]
else:
    balance = None

"""
"image_size_seq" is required to determine the output shape after spatial partitioning of images. 
The shape of the output will be determined for each model partition based on the values in "image_size_seq."
These values will then be used to calculate the output shape for a given input size and spatial partition.
"""
image_size_seq = 512

model_seq = amoebanet.amoebanetd(
    num_layers=num_layers, num_filters=num_filters, num_classes=num_classes
)
print("length", len(model_seq), balance)
model_gen_seq = model_generator(
    model=model_seq,
    split_size=split_size,
    input_size=(int(batch_size / parts), 3, image_size_seq, image_size_seq),
    balance=balance,
)
model_gen_seq.ready_model(split_rank=split_rank, GET_SHAPES_ON_CUDA=True)

image_size_times = int(image_size / image_size_seq)


temp_count = 0
if args.slice_method == "square":
    resnet_shapes_list = []
    for output_shape in model_gen_seq.shape_list:
        if isinstance(output_shape, list):
            temp_shape = []
            for shape_tuple in output_shape:
                if temp_count < spatial_size:
                    # reduce shape only when it is smaller than spatial size
                    x = (
                        int(shape_tuple[0]),
                        shape_tuple[1],
                        int(shape_tuple[2] * image_size_times / 2),
                        int(shape_tuple[3] * image_size_times / 2),
                    )
                    temp_shape.append(x)
                else:
                    x = (
                        int(shape_tuple[0]),
                        shape_tuple[1],
                        int(shape_tuple[2] * image_size_times),
                        int(shape_tuple[3] * image_size_times),
                    )
                    temp_shape.append(x)
            resnet_shapes_list.append(temp_shape)
        else:
            if len(output_shape) == 2:
                x = (int(output_shape[0]), output_shape[1])
                resnet_shapes_list.append(x)
            else:
                if temp_count < spatial_size:
                    x = (
                        int(output_shape[0]),
                        output_shape[1],
                        int(output_shape[2] * image_size_times / 2),
                        int(output_shape[3] * image_size_times / 2),
                    )
                    resnet_shapes_list.append(x)
                else:
                    x = (
                        int(output_shape[0]),
                        output_shape[1],
                        int(output_shape[2] * image_size_times),
                        int(output_shape[3] * image_size_times),
                    )
                    resnet_shapes_list.append(x)
        temp_count += 1

elif args.slice_method == "vertical":
    resnet_shapes_list = []
    for output_shape in model_gen_seq.shape_list:
        if isinstance(output_shape, list):
            temp_shape = []
            for shape_tuple in output_shape:
                if temp_count < spatial_size:
                    x = (
                        int(shape_tuple[0]),
                        shape_tuple[1],
                        int(shape_tuple[2] * image_size_times / 1),
                        int(
                            shape_tuple[3]
                            * image_size_times
                            / num_spatial_parts_list[temp_count]
                        ),
                    )
                    temp_shape.append(x)
                else:
                    x = (
                        int(shape_tuple[0]),
                        shape_tuple[1],
                        int(shape_tuple[2] * image_size_times),
                        int(shape_tuple[3] * image_size_times),
                    )
                    temp_shape.append(x)
            resnet_shapes_list.append(temp_shape)
        else:
            if len(output_shape) == 2:
                x = (int(output_shape[0]), output_shape[1])
                resnet_shapes_list.append(x)
            else:
                if temp_count < spatial_size:
                    x = (
                        int(output_shape[0]),
                        output_shape[1],
                        int(output_shape[2] * image_size_times / 1),
                        int(
                            output_shape[3]
                            * image_size_times
                            / num_spatial_parts_list[temp_count]
                        ),
                    )
                    resnet_shapes_list.append(x)
                else:
                    x = (
                        int(output_shape[0]),
                        output_shape[1],
                        int(output_shape[2] * image_size_times),
                        int(output_shape[3] * image_size_times),
                    )
                    resnet_shapes_list.append(x)
        temp_count += 1


elif args.slice_method == "horizontal":
    resnet_shapes_list = []
    for output_shape in model_gen_seq.shape_list:
        if isinstance(output_shape, list):
            temp_shape = []
            for shape_tuple in output_shape:
                if temp_count < spatial_size:
                    x = (
                        int(shape_tuple[0]),
                        shape_tuple[1],
                        int(
                            shape_tuple[2]
                            * image_size_times
                            / num_spatial_parts_list[temp_count]
                        ),
                        int(shape_tuple[3] * image_size_times / 1),
                    )
                    temp_shape.append(x)
                else:
                    x = (
                        int(shape_tuple[0]),
                        shape_tuple[1],
                        int(shape_tuple[2] * image_size_times),
                        int(shape_tuple[3] * image_size_times),
                    )
                    temp_shape.append(x)
            resnet_shapes_list.append(temp_shape)
        else:
            if len(output_shape) == 2:
                x = (int(output_shape[0]), output_shape[1])
                resnet_shapes_list.append(x)
            else:
                if temp_count < spatial_size:
                    x = (
                        int(output_shape[0]),
                        output_shape[1],
                        int(
                            output_shape[2]
                            * image_size_times
                            / num_spatial_parts_list[temp_count]
                        ),
                        int(output_shape[3] * image_size_times / 1),
                    )
                    resnet_shapes_list.append(x)
                else:
                    x = (
                        int(output_shape[0]),
                        output_shape[1],
                        int(output_shape[2] * image_size_times),
                        int(output_shape[3] * image_size_times),
                    )
                    resnet_shapes_list.append(x)
        temp_count += 1


print(model_gen_seq.shape_list, resnet_shapes_list)

del model_seq
del model_gen_seq
torch.cuda.ipc_collect()

if args.halo_d2:
    model = amoebanet_d2.amoebanetd_spatial(
        local_rank=local_rank % mpi_comm.total_spatial_processes,
        spatial_size=spatial_size,
        num_spatial_parts=num_spatial_parts,
        mp_size=split_size,
        balance=balance,
        slice_method="square",
        num_classes=num_classes,
        num_layers=num_layers,
        num_filters=num_filters,
    )
else:
    model = amoebanet.amoebanetd_spatial(
        local_rank=local_rank % mpi_comm.total_spatial_processes,
        spatial_size=spatial_size,
        num_spatial_parts=num_spatial_parts,
        mp_size=split_size,
        balance=balance,
        slice_method="square",
        num_classes=num_classes,
        num_layers=num_layers,
        num_filters=num_filters,
    )


model_gen = model_generator(
    model=model,
    split_size=split_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=balance,
    shape_list=resnet_shapes_list,
)


model_gen.ready_model(split_rank=split_rank)
model_gen.DDP_model(mpi_comm, num_spatial_parts, spatial_size, bucket_size=0)


print("Shape list", resnet_shapes_list)


t_s = train_model_spatial(
    model_gen,
    local_rank,
    batch_size,
    epochs=1,
    spatial_size=spatial_size,
    num_spatial_parts=num_spatial_parts,
    criterion=None,
    optimizer=None,
    parts=parts,
    ASYNC=True,
    GEMS_INVERSE=False,
    slice_method=args.slice_method,
    LOCAL_DP_LP=LOCAL_DP_LP,
    mpi_comm=mpi_comm,
)

x = torch.zeros(
    (batch_size, 3, int(image_size / 2), int(image_size / 2)), device="cuda"
)
y = torch.zeros((batch_size,), dtype=torch.long, device="cuda")


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
torch.manual_seed(0)

if APP == 1:
    trainset = torchvision.datasets.ImageFolder(
        "/usr/workspace/jain8/project/cancer/1024_1024_5/train",
        transform=transform,
        target_transform=None,
    )
    my_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=times * batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    size_dataset = 1030
elif APP == 2:
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    my_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=times * batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    size_dataset = 50000
else:
    my_dataset = torchvision.datasets.FakeData(
        size=10 * batch_size,
        image_size=(3, image_size, image_size),
        num_classes=num_classes,
        transform=transform,
        target_transform=None,
        random_offset=0,
    )
    my_dataloader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=batch_size * times,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    size_dataset = 10 * batch_size


# sync_allreduce.sync_model_spatial(model_gen)
perf = []


def split_input(inputs):
    if args.slice_method == "square":
        image_height_local = int(image_size / math.sqrt(num_spatial_parts))
        image_width_local = int(image_size / math.sqrt(num_spatial_parts))

        total_rows = int(math.sqrt(num_spatial_parts))
        total_cols = int(math.sqrt(num_spatial_parts))

        # current position of rank in matrix of math.sqrt(num_spatial_parts) * math.sqrt(num_spatial_parts)
        row = int(local_rank / total_cols)
        col = int(local_rank % total_cols)

        start_left = col * image_width_local
        end_right = (col + 1) * image_width_local

        start_top = row * image_height_local
        end_bottom = (row + 1) * image_height_local

        return inputs[:, :, start_top:end_bottom, start_left:end_right]

    elif args.slice_method == "vertical":
        image_height_local = int(image_size / num_spatial_parts)
        image_width_local = int(image_size / num_spatial_parts)

        start_left = local_rank * image_width_local
        end_right = (local_rank + 1) * image_width_local

        if local_rank == num_spatial_parts - 1:
            # In case of GPU count, partition size will be uneven and last
            # rank will receive remaining image
            return inputs[:, :, :, start_left:]
        else:
            return inputs[:, :, :, start_left:end_right]

    elif args.slice_method == "horizontal":
        image_height_local = int(image_size / num_spatial_parts)
        image_width_local = int(image_size / num_spatial_parts)

        start_top = local_rank * image_height_local
        end_bottom = (local_rank + 1) * image_height_local

        if local_rank == num_spatial_parts - 1:
            # In case of odd GPU count, partition size will be uneven and last
            # rank will receive remaining image
            return inputs[:, :, start_top:, :]
        else:
            return inputs[:, :, start_top:end_bottom, :]


def run_epoch():
    for i_e in range(epoch):
        loss = 0
        correct = 0
        t = time.time()
        for i, data in enumerate(my_dataloader, 0):
            start_event = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event.record()
            if i > math.floor(size_dataset / (times * batch_size)) - 1:
                break
            inputs, labels = data

            if local_rank < num_spatial_parts_list[0]:
                x = split_input(inputs)
            else:
                x = inputs

            temp_loss, temp_correct = t_s.run_step(x, labels)
            loss += temp_loss
            correct += temp_correct

            torch.cuda.synchronize()

            t_s.update()
            if local_rank == comm_size - 1:
                logging.info(
                    f"Step :{i}, LOSS: {temp_loss}, Global loss: {loss/(i+1)} Acc: {temp_correct}"
                )

            end_event.record()
            torch.cuda.synchronize()
            t = start_event.elapsed_time(end_event) / 1000
            if local_rank == 0:
                None
                print("images per sec:", batch_size / t)
                perf.append(batch_size / t)

            t = time.time()
        if local_rank == comm_size - 1:
            print("epoch", i_e, " Global loss:", loss, " acc", correct / i)


run_epoch()

if local_rank == 0:
    print("Mean {} Median {}".format(sum(perf) / len(perf), np.median(perf)))
exit()
