# Copyright 2023, The Ohio State University. All rights reserved.
# The MVAPICH software package is developed by the team members of
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
import torchvision.transforms as transforms
import torchvision
import numpy as np
import sys
import math
import logging
from torchgems import parser
import time
from torchgems.mp_pipeline import model_generator, train_model
from models import amoebanet
import torchgems.comm as gems_comm


parser_obj = parser.get_parser()
args = parser_obj.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG)

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


sys.stdout = Unbuffered(sys.stdout)

np.random.seed(seed=1405)
ENABLE_ASYNC = True

parts = args.parts
batch_size = args.batch_size
epochs = args.num_epochs
image_size = int(args.image_size)
num_layers = args.num_layers
num_filters = args.num_filters
balance = args.balance
mp_size = args.split_size
times = args.times
datapath = args.datapath
# APP
# 1: Medical
# 2: Cifar
# 3: synthetic
APP = args.app

##################### AmoebaNet model specific parameters #####################

image_size_seq = 512
num_classes = 1000

###############################################################################

mpi_comm = gems_comm.MPIComm(split_size=mp_size, ENABLE_MASTER=False)
rank = mpi_comm.rank
local_rank = rank % mp_size
if balance is not None:
    balance = [int(i) for i in balance.split(",")]

# Initialize AmoebaNet model
model = amoebanet.amoebanetd(
    num_classes=num_classes, num_layers=args.num_layers, num_filters=args.num_filters
)

# Initialize parameters for Model Parallelism
model_gen = model_generator(
    model=model,
    split_size=mp_size,
    input_size=(int(batch_size / parts), 3, image_size_seq, image_size_seq),
    balance=balance,
)

# Get the shape of model on each split rank for image_size_seq and move it to device
# Note : we take shape w.r.t image_size_seq as model w.r.t image_size may not be
# able to fit in memory

model_gen.ready_model(split_rank=local_rank, GET_SHAPES_ON_CUDA=True)

# Get the shape of model on each split rank for image_size
image_size_times = int(image_size / image_size_seq)
amoebanet_shapes_list = []
for output_shape in model_gen.shape_list:
    if isinstance(output_shape, list):
        temp_shape = []
        for shape_tuple in output_shape:
            x = (
                shape_tuple[0],
                shape_tuple[1],
                int(shape_tuple[2] * image_size_times),
                int(shape_tuple[3] * image_size_times),
            )
            temp_shape.append(x)
        amoebanet_shapes_list.append(temp_shape)
    else:
        if len(output_shape) == 2:
            amoebanet_shapes_list.append(output_shape)
        else:
            x = (
                output_shape[0],
                output_shape[1],
                int(output_shape[2] * image_size_times),
                int(output_shape[3] * image_size_times),
            )
            amoebanet_shapes_list.append(x)

model_gen.shape_list = amoebanet_shapes_list

logging.info(f"Shape of model on local_rank {local_rank} : {model_gen.shape_list}")

del model_gen
del model
torch.cuda.ipc_collect()

model = amoebanet.amoebanetd(
    num_classes=num_classes, num_layers=args.num_layers, num_filters=args.num_filters
)

model_gen = model_generator(
    model=model,
    split_size=mp_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=balance,
    shape_list=amoebanet_shapes_list,
)

# Move model it it's repective devices
model_gen.ready_model(split_rank=local_rank, GET_SHAPES_ON_CUDA=True)

tm = train_model(
    model_gen,
    local_rank,
    batch_size,
    epochs,
    criterion=None,
    optimizer=None,
    parts=parts,
    ASYNC=ENABLE_ASYNC,
)

############################## Dataset Definition ##############################

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
torch.manual_seed(0)

if APP == 1:
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
elif APP == 2:
    trainset = torchvision.datasets.CIFAR10(
        root=datapath, train=True, download=True, transform=transform
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


################################################################################


################################# Train Model ##################################

perf = []


def run_epoch():
    for i_e in range(epochs):
        loss = 0
        correct = 0
        t = time.time()
        for batch, data in enumerate(my_dataloader, 0):
            start_event = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event.record()

            if batch > math.floor(size_dataset / (times * batch_size)) - 1:
                break
            inputs, labels = data

            local_loss, local_correct = tm.run_step(inputs, labels)
            loss += local_loss
            correct += local_correct

            tm.update()

            end_event.record()
            torch.cuda.synchronize()
            t = start_event.elapsed_time(end_event) / 1000

            if local_rank == mp_size - 1:
                logging.info(
                    f"Step :{batch}, LOSS: {local_loss}, Global loss: {loss/(batch+1)} Acc: {local_correct}"
                )

            if local_rank == 0:
                print(f"Epoch: {i_e} images per sec:{batch_size / t}")
                perf.append(batch_size / t)

            t = time.time()
    if local_rank == mp_size - 1:
        print(f"Epoch {i_e} Global loss: {loss} Acc {correct / batch}")


run_epoch()

if local_rank == 0:
    print(f"Mean {sum(perf) / len(perf)} Median {np.median(perf)}")

################################################################################
