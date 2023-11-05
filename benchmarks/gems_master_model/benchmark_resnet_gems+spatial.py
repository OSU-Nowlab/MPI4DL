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
from torchgems.train_spatial import get_shapes_spatial, split_input
from torchgems.train_spatial_master import (
    train_spatial_model_master,
    verify_spatial_master_config,
)
import torchgems.comm as gems_comm
from models import resnet

# Example of GEMS + SPATIAL split_size = 2, spatial_size = 1, num_spatial_parts = 4
#
# split_size = 5, spatial_size = 1, num_spatial_parts = 4 are not valid as ranks 1, 2, 3 are used by spatial parts from both the model.
#  Model 1:
#  _______________        ____        ____        ____        ____
# |  0(0) |  1(1) |      |    |      |    |      |    |      |    |
# |-------|-------|----->|4(4)|----->|5(5)|----->|6(6)|----->|7(7)|
# |  2(2) |  3(3) |      |    |      |    |      |    |      |    |
# |_______|_______|      |____|      |____|      |____|      |____|
#
# Model 2 (INVERSE GEMS):
#  _______________        ____        ____        ____        ____
# |  0(7) |  1(6) |      |    |      |    |      |    |      |    |
# |-------|-------|----->|4(3)|----->|5(2)|----->|6(1)|----->|7(0)|
# |  2(5) |  3(4) |      |    |      |    |      |    |      |    |
# |_______|_______|      |____|      |____|      |____|      |____|
#
# Numbers inside the brackets () refer to World Rank
# whereas outside numbers refer to local rank for each model

parser_obj = parser.get_parser()
args = parser_obj.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG)

if args.halo_d2:
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


def init_processes(backend="tcp"):
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
slice_method = args.slice_method
times = args.times
datapath = args.datapath
num_classes = args.num_classes
LOCAL_DP_LP = args.local_DP
ENABLE_MASTER_OPT = args.enable_master_comm_opt
# APP
# 1: Medical
# 2: Cifar
# 3: synthetic
APP = args.app

temp_num_spatial_parts = args.num_spatial_parts.split(",")

if len(temp_num_spatial_parts) == 1:
    num_spatial_parts_list = [int(temp_num_spatial_parts[0])]
    num_spatial_parts = int(temp_num_spatial_parts[0])
else:
    num_spatial_parts = [int(i) for i in temp_num_spatial_parts]
    num_spatial_parts_list = num_spatial_parts

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

mpi_comm_first = gems_comm.MPIComm(
    split_size=split_size,
    ENABLE_MASTER=False,
    ENABLE_SPATIAL=True,
    num_spatial_parts=num_spatial_parts,
    spatial_size=spatial_size,
    LOCAL_DP_LP=LOCAL_DP_LP,
)

verify_spatial_master_config(
    slice_method,
    image_size,
    num_spatial_parts_list,
    spatial_size,
    mpi_comm_first.mp_size,
)

mpi_comm_second = gems_comm.MPIComm(
    split_size=split_size,
    ENABLE_MASTER=True,
    ENABLE_SPATIAL=True,
    num_spatial_parts=num_spatial_parts,
    spatial_size=spatial_size,
    LOCAL_DP_LP=LOCAL_DP_LP,
    DISABLE_INIT=True,
)

gems_comm.sync_comms_for_master(mpi_comm_first, mpi_comm_second)
comm_size = mpi_comm_first.size

if args.balance != None:
    balance = args.balance.split(",")
    balance = [int(j) for j in balance]
else:
    balance = None

# Initialize ResNet model
model_seq = resnet.get_resnet_v2(
    (int(batch_size / parts), 3, image_size_seq, image_size_seq),
    depth=get_depth(2, resnet_n),
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
model_gen_seq.ready_model(
    split_rank=mpi_comm_second.split_rank, GET_SHAPES_ON_CUDA=True
)

image_size_times = int(image_size / image_size_seq)

# Get the shape of model on each split rank for image_size and number of spatial parts
resnet_shapes_list = get_shapes_spatial(
    shape_list=model_gen_seq.shape_list,
    slice_method=slice_method,
    spatial_size=spatial_size,
    num_spatial_parts_list=num_spatial_parts_list,
    image_size_times=image_size_times,
)

del model_seq
del model_gen_seq
torch.cuda.ipc_collect()

# Initialize ResNet model with Spatial and Model Parallelism support
if args.halo_d2:
    model1, balance = resnet_spatial.get_resnet_v2(
        input_shape=(batch_size / parts, 3, image_size, image_size),
        depth=get_depth(2, 12),
        local_rank=mpi_comm_first.local_rank % mpi_comm_first.total_spatial_processes,
        mp_size=split_size,
        balance=balance,
        spatial_size=spatial_size,
        num_spatial_parts=num_spatial_parts,
        num_classes=num_classes,
        fused_layers=args.fused_layers,
        slice_method=slice_method,
    )

    model2, balance = resnet_spatial.get_resnet_v2(
        input_shape=(batch_size / parts, 3, image_size, image_size),
        depth=get_depth(2, 12),
        local_rank=mpi_comm_second.local_rank % mpi_comm_second.total_spatial_processes,
        mp_size=split_size,
        balance=balance,
        spatial_size=spatial_size,
        num_spatial_parts=num_spatial_parts,
        num_classes=num_classes,
        fused_layers=args.fused_layers,
        slice_method=slice_method,
    )
else:
    model1 = resnet_spatial.get_resnet_v2(
        input_shape=(batch_size / parts, 3, image_size, image_size),
        depth=get_depth(2, 12),
        local_rank=mpi_comm_first.local_rank % mpi_comm_first.total_spatial_processes,
        mp_size=split_size,
        balance=balance,
        spatial_size=spatial_size,
        num_spatial_parts=num_spatial_parts,
        num_classes=num_classes,
        fused_layers=args.fused_layers,
        slice_method=slice_method,
    )

    model2 = resnet_spatial.get_resnet_v2(
        input_shape=(batch_size / parts, 3, image_size, image_size),
        depth=get_depth(2, 12),
        local_rank=mpi_comm_second.local_rank % mpi_comm_second.total_spatial_processes,
        mp_size=split_size,
        balance=balance,
        spatial_size=spatial_size,
        num_spatial_parts=num_spatial_parts,
        num_classes=num_classes,
        fused_layers=args.fused_layers,
        slice_method=slice_method,
    )


model_gen1 = model_generator(
    model=model1,
    split_size=split_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=balance,
    shape_list=resnet_shapes_list,
)

# Move model it it's repective devices
model_gen1.ready_model(split_rank=mpi_comm_first.split_rank)
logging.info(
    f"Shape of model1 on local_rank {mpi_comm_first.local_rank } : {model_gen1.shape_list}"
)

model_gen2 = model_generator(
    model=model2,
    split_size=split_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=balance,
    shape_list=resnet_shapes_list,
)

# Move model it it's repective devices
model_gen2.ready_model(split_rank=mpi_comm_second.split_rank)
logging.info(
    f"Shape of model2 on local_rank {mpi_comm_first.local_rank } : {model_gen2.shape_list}"
)

t_s_master = train_spatial_model_master(
    model_gen1,
    model_gen2,
    batch_size,
    spatial_size,
    num_spatial_parts,
    slice_method,
    mpi_comm_first,
    mpi_comm_second,
    LOCAL_DP_LP=LOCAL_DP_LP,
    criterion=None,
    optimizer=None,
    parts=parts,
    ASYNC=True,
    replications=int(args.times / 2),
)

############################## Dataset Definition ##############################

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
elif APP == 3:
    my_dataset = torchvision.datasets.FakeData(
        size=10 * batch_size * args.times,
        image_size=(3, image_size, image_size),
        num_classes=num_classes,
        transform=transform,
        target_transform=None,
        random_offset=0,
    )
    my_dataloader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=batch_size * args.times,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    size_dataset = 10 * batch_size
else:
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.ImageFolder(
        datapath,
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
    size_dataset = len(my_dataloader.dataset)

################################################################################

sync_comm = gems_comm.SyncAllreduce(mpi_comm_first)

################################# Train Model ##################################

perf = []


def run_epoch():
    for i_e in range(epochs):
        loss = 0
        correct = 0
        t = time.time()
        size = len(my_dataloader.dataset)
        for batch, data in enumerate(my_dataloader, 0):
            start_event = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event.record()
            if batch > math.floor(size_dataset / (times * batch_size)) - 1:
                break

            inputs, labels = data

            if mpi_comm_first.local_rank < num_spatial_parts_list[0]:
                x = split_input(
                    inputs=inputs,
                    image_size=image_size,
                    slice_method=slice_method,
                    local_rank=mpi_comm_first.local_rank,
                    num_spatial_parts_list=num_spatial_parts_list,
                )
            elif mpi_comm_second.local_rank < num_spatial_parts_list[0]:
                x = split_input(
                    inputs=inputs,
                    image_size=image_size,
                    slice_method=slice_method,
                    local_rank=mpi_comm_second.local_rank,
                    num_spatial_parts_list=num_spatial_parts_list,
                )
            else:
                x = inputs

            if ENABLE_MASTER_OPT:
                local_loss, local_correct = t_s_master.run_step_allreduce(
                    x, labels, batch % 2 == 1
                )
            else:
                local_loss, local_correct = t_s_master.run_step(x, labels)

            loss += local_loss
            correct += local_correct

            start_event_allreduce = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event_allreduce = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event_allreduce.record()
            t_allreduce_temp = time.time()

            if ENABLE_MASTER_OPT == False:
                sync_comm.apply_allreduce_master_master(
                    model_gen1, model_gen2, mpi_comm_first, mpi_comm_second
                )
            torch.cuda.synchronize()

            if ENABLE_MASTER_OPT:
                if batch % 2 == 1:
                    t_s_master.train_model1.update()
                else:
                    t_s_master.train_model2.update()
            else:
                t_s_master.train_model1.update()
                t_s_master.train_model2.update()

            end_event_allreduce.record()
            torch.cuda.synchronize()
            t_allreduce = start_event_allreduce.elapsed_time(end_event_allreduce) / 1000
            t_allreduce = time.time() - t_allreduce_temp

            if mpi_comm_second.local_rank == comm_size - 1:
                logging.info(
                    f"Step :{batch}, LOSS: {local_loss}, Global loss: {loss/(batch+1)} Acc: {local_correct} [{batch * len(inputs):>5d}/{size:>5d}]"
                )

            if ENABLE_MASTER_OPT:
                torch.distributed.barrier()

            end_event.record()
            torch.cuda.synchronize()
            t = start_event.elapsed_time(end_event) / 1000
            if mpi_comm_second.local_rank == 0:
                print(
                    f"Epoch: {i_e} images per sec:{batch_size / t} Time:{t} Time Allreduce:{t_allreduce}"
                )
                perf.append(batch_size / t)

            t = time.time()
        if mpi_comm_second.local_rank == comm_size - 1:
            print(f"Epoch {i_e} Global loss: {loss / batch} Acc {correct / batch}")


run_epoch()

################################################################################

if mpi_comm_second.local_rank == 0:
    print("Mean {} Median {}".format(sum(perf) / len(perf), np.median(perf)))

exit()
