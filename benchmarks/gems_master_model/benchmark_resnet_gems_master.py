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
from models import resnet
from torchgems import parser
from torchgems.mp_pipeline import model_generator
from torchgems.gems_master import train_model_master
import torchgems.comm as gems_comm
from torchgems.utils import get_depth

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


def init_processes(backend="mpi"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend)
    size = dist.get_world_size()
    rank = dist.get_rank()
    return size, rank


sys.stdout = Unbuffered(sys.stdout)

np.random.seed(seed=1405)
parts = args.parts
batch_size = args.batch_size
epoch = args.num_epochs

# APP
# 1: Medical
# 2: Cifar
# 3: synthetic
APP = args.app
times = args.times
image_size = int(args.image_size)
num_layers = args.num_layers
num_filters = args.num_filters
balance = args.balance
mp_size = args.split_size
datapath = args.datapath
num_workers = args.num_workers
num_classes = args.num_classes
precision = str(args.precision)
backend = args.backend


if precision == "bf_16":
    assert torch.cuda.is_bf16_supported() == True, "Native System doen't support bf16"

EVAL_MODE = args.enable_evaluation
CHECKPOINT = None
if EVAL_MODE and APP != 3:
    # Note MPI4DL_ImageNeteee.pth is with image_size 256 and 10 num_classes
    CHECKPOINT = "/users/PAS2312/rgulhane/nowlab/checkpoints/imagenetee_img_size_64/MPI4DL_ImageNeteee_TensorRT_model_temp.pth"


################## ResNet model specific parameters/functions ##################

image_size_seq = 32
ENABLE_ASYNC = True
resnet_n = 12

###############################################################################
mpi_comm = gems_comm.MPIComm(split_size=mp_size, ENABLE_MASTER=True, backend=backend)
rank = mpi_comm.rank

local_rank = rank % mp_size

if balance is not None:
    balance = [int(i) for i in balance.split(",")]

# Initialize ResNet model
model = resnet.get_resnet_v2(
    (int(batch_size / parts), 3, image_size_seq, image_size_seq),
    depth=get_depth(2, resnet_n),
    num_classes=num_classes,
)


mul_shape = int(args.image_size / image_size_seq)

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
resnet_shapes_list = []
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
        resnet_shapes_list.append(temp_shape)
    else:
        if len(output_shape) == 2:
            resnet_shapes_list.append(output_shape)
        else:
            x = (
                output_shape[0],
                output_shape[1],
                int(output_shape[2] * image_size_times),
                int(output_shape[3] * image_size_times),
            )
            resnet_shapes_list.append(x)

model_gen.shape_list = resnet_shapes_list
logging.info(f"Shape of model on local_rank {local_rank} : {model_gen.shape_list}")


del model_gen
del model
torch.cuda.ipc_collect()

model = resnet.get_resnet_v2(
    (int(batch_size / parts), 3, image_size, image_size),
    get_depth(2, resnet_n),
    num_classes=num_classes,
)

# GEMS Model 1
model_gen1 = model_generator(
    model=model,
    split_size=mp_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=None,
    shape_list=resnet_shapes_list,
)
model_gen1.ready_model(
    split_rank=local_rank,
    eval_mode=EVAL_MODE,
    checkpoint_path=CHECKPOINT,
    precision=precision,
)

model = resnet.get_resnet_v2(
    (int(batch_size / parts), 3, image_size, image_size),
    get_depth(2, resnet_n),
    num_classes=num_classes,
)

# GEMS Model 2
model_gen2 = model_generator(
    model=model,
    split_size=mp_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=None,
    shape_list=model_gen1.shape_list,
)
model_gen2.ready_model(
    split_rank=mp_size - local_rank - 1,
    eval_mode=EVAL_MODE,
    checkpoint_path=CHECKPOINT,
    precision=precision,
)

tm_master = train_model_master(
    model_gen1,
    model_gen2,
    local_rank,
    batch_size,
    epoch,
    precision,
    eval_mode=EVAL_MODE,
    criterion=None,
    optimizer=None,
    parts=parts,
    ASYNC=ENABLE_ASYNC,
)


sync_allreduce = gems_comm.SyncAllreduce(mpi_comm)

############################## Dataset Definition ##############################

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
torch.manual_seed(0)

if APP == 1:
    if EVAL_MODE:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        torch.manual_seed(0)

        # testset = torchvision.datasets.ImageNet(
        #         root="/home/gulhane.2/GEMS_Inference/datasets/ImageNet/", split='val', transform=transform
        # )
        testset = torchvision.datasets.ImageFolder(
            root=datapath,
            transform=transform,
            target_transform=None,
        )

        my_dataloader = torch.utils.data.DataLoader(
            testset,
            batch_size=times * batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        size_dataset = len(my_dataloader.dataset)
    else:
        trainset = torchvision.datasets.ImageFolder(
            datapath,
            transform=transform,
            target_transform=None,
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
    size_dataset = len(my_dataloader.dataset)
elif APP == 3:
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
else:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # root="/home/gulhane.2/GEMS_Inference/datasets/ImageNet/",
    my_dataset = torchvision.datasets.ImageNet(
        root=datapath, split="train", transform=transform
    )

    my_dataloader = torch.utils.data.DataLoader(
        my_dataset, batch_size=batch_size, shuffle=False
    )
    size_dataset = len(my_dataloader.dataset)


################################################################################

if EVAL_MODE == False:
    sync_allreduce.sync_model(model_gen1, model_gen2)

perf = []


def run_epoch():
    for i_e in range(epoch):
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

            local_loss, local_correct = tm_master.run_step(
                inputs, labels, eval_mode=EVAL_MODE
            )
            loss += local_loss
            correct += local_correct
            sync_allreduce.apply_allreduce_master_and_update(
                tm_master, model_gen1, model_gen2
            )

            end_event.record()
            torch.cuda.synchronize()
            t = start_event.elapsed_time(end_event) / 1000

            if local_rank == mp_size - 1:
                logging.info(
                    f"Step :{batch}, LOSS: {local_loss}, Global loss: {loss/(batch+1)} Acc: {local_correct} [{batch * len(inputs):>5d}/{size:>5d}]"
                )

            if local_rank == 0:
                print(f"Epoch: {i_e} images per sec:{batch_size / t}")
                perf.append(batch_size / t)

            t = time.time()
        if local_rank == mp_size - 1:
            print(f"Epoch {i_e} Global loss: {loss / batch} Acc {correct / batch}")


def run_eval():
    # ImageNettee:
    # Global loss: 1.7099052721085777 Acc 0.753822629969419 batch = 1
    # images per sec:32.311230273782776
    # Global loss: 1.7051572809413988 Acc 0.761734693877551 batch = 4

    loss = 0
    correct = 0
    size = len(my_dataloader.dataset)
    t = time.time()
    with torch.no_grad():
        for batch, data in enumerate(my_dataloader, 0):
            inputs, labels = data

            start_event = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event.record()
            if precision == "fp_16":
                inputs = inputs.half()
            elif precision == "bfp_16":
                inputs = inputs.to(torch.bfloat16)
                # labels = labels.to(torch.float16)

            if batch > math.floor(size_dataset / (times * batch_size)) - 1:
                break
            before_step = torch.cuda.max_memory_allocated(device="cuda")
            # print(
            #     f"Max Memory before step {batch} on rank {local_rank} Using PyTorch CUDA: {before_step / (1024 ** 2):.2f} MB"
            # )

            local_loss, local_correct = tm_master.run_step(
                inputs, labels, eval_mode=EVAL_MODE
            )
            loss += local_loss
            correct += local_correct

            end_event.record()
            torch.cuda.synchronize()
            t = start_event.elapsed_time(end_event) / 1000

            if local_rank == mp_size - 1:
                logging.info(
                    f"Step :{batch}, LOSS: {local_loss}, Global loss: {loss/(batch+1)} Acc: {local_correct} [{batch * len(inputs):>5d}/{size:>5d}]"
                )

            if local_rank == 0:
                print(f"images per sec:{batch_size / t}")
                perf.append(batch_size / t)
            t = time.time()
        if local_rank == mp_size - 1:
            print(f"Global loss: {loss / batch} Acc {correct / batch}")


if EVAL_MODE == True:
    run_eval()
else:
    run_epoch()

if local_rank == 0:
    print(f"Mean {sum(perf) / len(perf)} Median {np.median(perf)}")

################################################################################
