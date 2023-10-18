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

# from torchgems.gems_master import train_model_master
# from torchgems.train_spatial import get_shapes_spatial
from torchgems.train_spatial import split_input
from torchgems.train_spatial_master import train_spatial_model_master
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


def init_processes(backend="tcp"):
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
image_size = int(args.image_size)
num_layers = args.num_layers
num_filters = args.num_filters
balance = args.balance
mp_size = args.split_size
datapath = args.datapath
num_classes = args.num_classes
split_size = args.split_size
spatial_size = args.spatial_size
ENABLE_MASTER_OPT = args.enable_master_comm_opt

temp_num_spatial_parts = args.num_spatial_parts.split(",")

if len(temp_num_spatial_parts) == 1:
    num_spatial_parts_list = [int(temp_num_spatial_parts[0])]
    num_spatial_parts = int(temp_num_spatial_parts[0])
else:
    num_spatial_parts = [int(i) for i in temp_num_spatial_parts]
    num_spatial_parts_list = num_spatial_parts


times = args.times
num_classes = 1000
LOCAL_DP_LP = args.local_DP

################## ResNet model specific parameters/functions ##################

image_size_seq = 32
ENABLE_ASYNC = True
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
# local_rank = rank % mp_size

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
model_gen.ready_model(split_rank=mpi_comm_second.split_rank, GET_SHAPES_ON_CUDA=True)

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
# logging.info(f"Shape of model on local_rank {local_rank} : {model_gen.shape_list}")


del model_gen
del model
torch.cuda.ipc_collect()

model = resnet.get_resnet_v2(
    (int(batch_size / parts), 3, image_size, image_size), get_depth(2, resnet_n)
)

# GEMS Model 1
model_gen1 = model_generator(
    model=model,
    split_size=mp_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=None,
    shape_list=resnet_shapes_list,
)
model_gen1.ready_model(split_rank=mpi_comm_first.split_rank)


model = resnet.get_resnet_v2(
    (int(batch_size / parts), 3, image_size, image_size), get_depth(2, resnet_n)
)

# GEMS Model 2
model_gen2 = model_generator(
    model=model,
    split_size=mp_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=None,
    shape_list=model_gen1.shape_list,
)
model_gen2.ready_model(split_rank=mpi_comm_second.split_rank)
print("Shape list", resnet_shapes_list)

# tm_master = train_model_master(
#     model_gen1,
#     model_gen2,
#     local_rank,
#     batch_size,
#     epoch,
#     criterion=None,
#     optimizer=None,
#     parts=parts,
#     ASYNC=ENABLE_ASYNC,
# )
t_s_master = train_spatial_model_master(
    model_gen1,
    model_gen2,
    batch_size,
    spatial_size,
    num_spatial_parts,
    args.slice_method,
    mpi_comm_first,
    mpi_comm_second,
    LOCAL_DP_LP=LOCAL_DP_LP,
    criterion=None,
    optimizer=None,
    parts=parts,
    ASYNC=True,
    replications=int(args.times / 2),
)

# sync_allreduce = gems_comm.SyncAllreduce(mpi_comm)


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
    size_dataset = len(my_dataloader.dataset)
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

# sync_allreduce.sync_model(model_gen1, model_gen2)

perf = []

sync_comm = gems_comm.SyncAllreduce(mpi_comm_first)


MASTER = args.times

print("ENABLE_MASTER_OPT", ENABLE_MASTER_OPT)

perf = []


def run_epoch():
    for i_e in range(epoch):
        loss = 0
        correct = 0
        size = len(my_dataloader.dataset)
        t = time.time()
        for i, data in enumerate(my_dataloader, 0):
            start_event = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event.record()

            if i > math.floor(size_dataset / (times * batch_size)) - 1:
                break

            inputs, labels = data

            if mpi_comm_first.local_rank < num_spatial_parts_list[0]:
                x = split_input(
                    inputs=inputs,
                    image_size=image_size,
                    slice_method=args.slice_method,
                    local_rank=mpi_comm_first.local_rank,
                    num_spatial_parts_list=num_spatial_parts_list,
                )
            elif mpi_comm_second.local_rank < num_spatial_parts_list[0]:
                x = split_input(
                    inputs=inputs,
                    image_size=image_size,
                    slice_method=args.slice_method,
                    local_rank=mpi_comm_second.local_rank,
                    num_spatial_parts_list=num_spatial_parts_list,
                )
            else:
                x = inputs

            # for j in range(MASTER):

            # 	temp_loss,temp_correct = t_s1.run_step(x,labels)
            # 	temp_loss,temp_correct = t_s2.run_step(x,labels)

            if ENABLE_MASTER_OPT:
                temp_loss, temp_correct = t_s_master.run_step_allreduce(
                    x, labels, i % 2 == 1
                )
            else:
                temp_loss, temp_correct = t_s_master.run_step(x, labels)

            loss += temp_loss
            correct += temp_correct

            start_event_allreduce = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event_allreduce = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event_allreduce.record()
            t_allreduce_temp = time.time()

            if ENABLE_MASTER_OPT == False:
                print("benchmark_amoebanet_gems+spatial : START ALL REDUCE OPERATION")
                sync_comm.apply_allreduce_master_master(
                    model_gen1, model_gen2, mpi_comm_first, mpi_comm_second
                )

            """
			if(local_rank < spatial_size * num_spatial_parts):
				None
				#No need for this as, DDP is now used
				# sync_allreduce.apply_allreduce(model_gen,mpi_comm.spatial_allreduce_grp)
			"""
            torch.cuda.synchronize()

            if ENABLE_MASTER_OPT:
                if i % 2 == 1:
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
                None
                # print("Step",i," LOSS",temp_loss, " Global loss:",loss/(i+1), " Acc:",temp_correct)

            if ENABLE_MASTER_OPT:
                torch.distributed.barrier()

            end_event.record()
            torch.cuda.synchronize()
            t = start_event.elapsed_time(end_event) / 1000
            if mpi_comm_second.local_rank == 0:
                None
                print(
                    "images per sec:",
                    batch_size / t,
                    "Time:",
                    t,
                    " Time Allreduce:",
                    t_allreduce,
                )
                perf.append(batch_size / t)

            t = time.time()
        if mpi_comm_second.local_rank == comm_size - 1:
            print("epoch", i_e, " Global loss:", loss, " acc", correct / i)


run_epoch()

if mpi_comm_second.local_rank == 0:
    print(f"Mean {sum(perf) / len(perf)} Median {np.median(perf)}")

exit()
################################################################################
