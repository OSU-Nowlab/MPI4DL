# Copyright 2023, The Ohio State University. All rights reserved.
# The MPI4DL software package is developed by the team members of
# The Ohio State University's Network-Based Computing Laboratory (NBCL),
# headed by Professor Dhabaleswar K. (DK) Panda.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from models import resnet
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision
import numpy as np
import time
import sys
import math
import logging
from torchgems import parser
from torchgems.mp_pipeline import model_generator
from torchgems.train_spatial import train_model_spatial, split_input, get_shapes_spatial
import torchgems.comm as gems_comm

parser_obj = parser.get_parser()
args = parser_obj.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG)

if args.halo_d2:
    # from models import resnet
    from models import resnet_spatial_d2 as resnet_spatial
else:
    from models import resnet_spatial

gems_comm.initialize_cuda()


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


def init_processes(backend="mpi"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend)
    size = dist.get_world_size()
    rank = dist.get_rank()
    return size, rank


sys.stdout = Unbuffered(sys.stdout)

np.random.seed(seed=1405)

ENABLE_ASYNC = True
parts = args.parts
batch_size = args.batch_size
epochs = args.num_epochs
image_size = int(args.image_size)
balance = args.balance
split_size = args.split_size
spatial_size = args.spatial_size
times = args.times
datapath = args.datapath
num_workers = args.num_workers

# APP
# 1: Medical
# 2: Cifar
# 3: synthetic
APP = args.app
num_classes = args.num_classes

temp_num_spatial_parts = args.num_spatial_parts.split(",")

if len(temp_num_spatial_parts) == 1:
    num_spatial_parts_list = [int(temp_num_spatial_parts[0])]
    num_spatial_parts = int(temp_num_spatial_parts[0])
else:
    num_spatial_parts = [int(i) for i in temp_num_spatial_parts]
    num_spatial_parts_list = num_spatial_parts

spatial_part_size = num_spatial_parts_list[0]  # Partition size for spatial parallelism

################## ResNet model specific parameters/functions ##################

"""
"image_size_seq" is required to determine the output shape after spatial partitioning of images.
The shape of the output will be determined for each model partition based on the values in "image_size_seq."
These values will then be used to calculate the output shape for a given input size and spatial partition.
"""
image_size_seq = 32
resnet_n = 12


def get_depth(version, n):
    if version == 1:
        return n * 6 + 2
    elif version == 2:
        return n * 9 + 2


###############################################################################


def isPowerTwo(num):
    return not (num & (num - 1))


"""
For ResNet model, image size and image size after partitioning should be power of two.
As, ResNet performs convolution operations at different layers, odd input size
(i.e. image size which is not power of 2) will lead to truncation of input. Thus,
other GPU devices will receive truncated input with unexpected input size.
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
            int(image_size / math.sqrt(spatial_part_size))
        ), "Image size of each partition should be power of Two"
    else:
        assert isPowerTwo(
            int(image_size / spatial_part_size)
        ), "Image size of each partition should be power of Two"

    for each_part_size in num_spatial_parts_list:
        assert (
            each_part_size == spatial_part_size
        ), "Size of each SP partition should be same"


verify_config()

mpi_comm = gems_comm.MPIComm(
    split_size=split_size,
    ENABLE_MASTER=False,
    ENABLE_SPATIAL=True,
    num_spatial_parts=num_spatial_parts,
    spatial_size=spatial_size,
)
sync_allreduce = gems_comm.SyncAllreduce(mpi_comm)
rank = mpi_comm.rank
comm_size = mpi_comm.size
local_rank = rank
split_rank = mpi_comm.split_rank


if balance != None:
    balance = balance.split(",")
    balance = [int(j) for j in balance]
else:
    balance = None

# Initialize ResNet model
model_seq = resnet.get_resnet_v2(
    (int(batch_size / parts), 3, image_size_seq, image_size_seq), depth=get_depth(2, 12)
)

model_gen_seq = model_generator(
    model=model_seq,
    split_size=split_size,
    input_size=(int(batch_size / parts), 3, image_size_seq, image_size_seq),
    balance=balance,
)

# Get the shape of model on each split rank for image_size_seq and move it to device
# Note : we take shape w.r.t image_size_seq as model w.r.t image_size may not be
# able to fit in memory
model_gen_seq.ready_model(split_rank=split_rank, GET_SHAPES_ON_CUDA=True)


# Get the shape of model on each split rank for image_size and number of spatial parts
image_size_times = int(image_size / image_size_seq)
resnet_shapes_list = get_shapes_spatial(
    model_gen_seq.shape_list,
    args.slice_method,
    spatial_size,
    num_spatial_parts_list,
    image_size_times,
)

del model_seq
del model_gen_seq
torch.cuda.ipc_collect()

# Initialize ResNet model with Spatial and Model Parallelism support
if args.halo_d2:
    model, balance = resnet_spatial.get_resnet_v2(
        input_shape=(batch_size / parts, 3, image_size, image_size),
        depth=get_depth(2, 12),
        local_rank=local_rank % spatial_part_size,
        mp_size=split_size,
        balance=balance,
        spatial_size=spatial_size,
        num_spatial_parts=num_spatial_parts,
        num_classes=num_classes,
        fused_layers=args.fused_layers,
        slice_method=args.slice_method,
    )
else:
    model = resnet_spatial.get_resnet_v2(
        input_shape=(batch_size / parts, 3, image_size, image_size),
        depth=get_depth(2, 12),
        local_rank=local_rank % spatial_part_size,
        mp_size=split_size,
        balance=balance,
        spatial_size=spatial_size,
        num_spatial_parts=num_spatial_parts,
        num_classes=num_classes,
        fused_layers=args.fused_layers,
        slice_method=args.slice_method,
    )


model_gen = model_generator(
    model=model,
    split_size=split_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=balance,
    shape_list=resnet_shapes_list,
)

# Move model it it's repective devices
model_gen.ready_model(split_rank=split_rank)

logging.info(f"Shape of model on local_rank {local_rank} : {model_gen.shape_list}")

# Initialize parameters require for training the model with Spatial and Model
# Parallelism support
t_s = train_model_spatial(
    model_gen,
    local_rank,
    batch_size,
    epochs=1,
    spatial_size=spatial_size,
    num_spatial_parts=num_spatial_parts,
    criterion=None,
    optimizer=None,
    parts=1,
    ASYNC=True,
    GEMS_INVERSE=False,
    slice_method=args.slice_method,
    mpi_comm=mpi_comm,
)

x = torch.zeros(
    (
        batch_size,
        3,
        int(image_size / math.sqrt(spatial_part_size)),
        int(image_size / math.sqrt(spatial_part_size)),
    ),
    device="cuda",
)
y = torch.zeros((batch_size,), dtype=torch.long, device="cuda")


############################## Dataset Definition ##############################
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
torch.manual_seed(0)

if APP == 1:
    trainset = torchvision.datasets.ImageFolder(
        datapath, transform=transform, target_transform=None
    )
    my_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=times * batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    size_dataset = len(my_dataloader.dataset)
elif APP == 2:
    trainset = torchvision.datasets.CIFAR10(
        root=datapath, train=True, download=True, transform=transform
    )
    my_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=times * batch_size,
        shuffle=False,
        num_workers=num_workers,
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
        num_workers=num_workers,
        pin_memory=True,
    )
    size_dataset = 10 * batch_size

################################################################################

sync_allreduce.sync_model_spatial(model_gen)

################################# Train Model ##################################

perf = []


def run_epoch():
    for i_e in range(epochs):
        loss = 0
        correct = 0
        size = len(my_dataloader.dataset)
        t = time.time()
        for batch, data in enumerate(my_dataloader, 0):
            start_event = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event.record()
            if batch > math.floor(size_dataset / (times * batch_size)) - 1:
                break
            inputs, labels = data

            if local_rank < spatial_part_size:
                x = split_input(
                    inputs, args.slice_method, image_size, spatial_part_size, local_rank
                )
            else:
                x = inputs

            local_loss, local_correct = t_s.run_step(x, labels)
            loss += local_loss
            correct += local_correct
            if local_rank < spatial_size * spatial_part_size:
                sync_allreduce.apply_allreduce(
                    model_gen, mpi_comm.spatial_allreduce_grp
                )
            torch.cuda.synchronize()

            t_s.update()
            if local_rank == spatial_part_size:
                logging.info(
                    f"Step :{batch}, LOSS: {local_loss}, Global loss: {loss/(batch+1)} Acc: {local_correct} [{batch * len(inputs):>5d}/{size:>5d}]"
                )

            end_event.record()
            torch.cuda.synchronize()
            t = start_event.elapsed_time(end_event) / 1000
            if local_rank == 0:
                print(f"Epoch: {i_e} images per sec:{batch_size / t}")
                perf.append(batch_size / t)

            t = time.time()
        if local_rank == comm_size - 1:
            print(f"Epoch {i_e} Global loss: {loss / batch} Acc {correct / batch}")


run_epoch()

if local_rank == 0:
    print(f"Mean {sum(perf) / len(perf)} Median {np.median(perf)}")

################################################################################

exit()
