import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision
import numpy as np
import time
import sys
import math
from torchgems import parser
from torchgems.mp_pipeline import model_generator
from torchgems.train_spatial import get_shapes_spatial, split_input
from torchgems.train_spatial_master import train_spatial_model_master
import torchgems.comm as gems_comm

parser_obj = parser.get_parser()
args = parser_obj.parse_args()

if args.halo_d2:
    from models import amoebanet
    from models import amoebanet_d2

else:
    from models import amoebanet

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


def get_depth(version, n):
    if version == 1:
        return n * 6 + 2
    elif version == 2:
        return n * 9 + 2


sys.stdout = Unbuffered(sys.stdout)

# torch.set_num_threads(1)
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
APP = 3
amoebanet_test = False
image_size = int(args.image_size)
print("image size", image_size)
steps = 100
num_layers = args.num_layers
num_filters = args.num_filters
balance = args.balance
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
# rank = mpi_comm.local_rank
# comm_size = mpi_comm.size
# local_rank = rank

# split_rank = mpi_comm.split_rank


if args.balance != None:
    balance = args.balance.split(",")
    balance = [int(j) for j in balance]
else:
    balance = None


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
model_gen_seq.ready_model(
    split_rank=mpi_comm_second.split_rank, GET_SHAPES_ON_CUDA=True
)

image_size_times = int(image_size / image_size_seq)

resnet_shapes_list = get_shapes_spatial(
    shape_list=model_gen_seq.shape_list,
    slice_method=args.slice_method,
    spatial_size=spatial_size,
    num_spatial_parts_list=num_spatial_parts_list,
    image_size_times=image_size_times,
)

print(model_gen_seq.shape_list, resnet_shapes_list)

del model_seq
del model_gen_seq
torch.cuda.ipc_collect()


if args.halo_d2:
    model1 = amoebanet_d2.amoebanetd_spatial(
        local_rank=mpi_comm_first.local_rank % mpi_comm_first.total_spatial_processes,
        spatial_size=spatial_size,
        num_spatial_parts=num_spatial_parts,
        mp_size=split_size,
        balance=balance,
        slice_method="square",
        num_classes=num_classes,
        num_layers=num_layers,
        num_filters=num_filters,
    )

    model2 = amoebanet_d2.amoebanetd_spatial(
        local_rank=mpi_comm_second.local_rank % mpi_comm_second.total_spatial_processes,
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
    model1 = amoebanet.amoebanetd_spatial(
        local_rank=mpi_comm_first.local_rank % mpi_comm_first.total_spatial_processes,
        spatial_size=spatial_size,
        num_spatial_parts=num_spatial_parts,
        mp_size=split_size,
        balance=balance,
        slice_method="square",
        num_classes=num_classes,
        num_layers=num_layers,
        num_filters=num_filters,
    )

    model2 = amoebanet.amoebanetd_spatial(
        local_rank=mpi_comm_second.local_rank % mpi_comm_second.total_spatial_processes,
        spatial_size=spatial_size,
        num_spatial_parts=num_spatial_parts,
        mp_size=split_size,
        balance=balance,
        slice_method="square",
        num_classes=num_classes,
        num_layers=num_layers,
        num_filters=num_filters,
    )


model_gen1 = model_generator(
    model=model1,
    split_size=split_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=balance,
    shape_list=resnet_shapes_list,
)
model_gen1.ready_model(split_rank=mpi_comm_first.split_rank)
# model_gen1.DDP_model(mpi_comm_first, num_spatial_parts, spatial_size, bucket_size=25, local_rank = mpi_comm_first.local_rank)


model_gen2 = model_generator(
    model=model2,
    split_size=split_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=balance,
    shape_list=resnet_shapes_list,
)
model_gen2.ready_model(split_rank=mpi_comm_second.split_rank)
# model_gen2.DDP_model(mpi_comm_second, num_spatial_parts, spatial_size, bucket_size=25, local_rank = mpi_comm_second.local_rank)


# model_gen.mp_size = 5
print("Shape list", resnet_shapes_list)


# t_s1 = train_model_spatial(model_gen1, mpi_comm_first.local_rank,batch_size,epochs=1, spatial_size=spatial_size, num_spatial_parts=num_spatial_parts ,criterion=None,optimizer=None,parts=parts,ASYNC=True,GEMS_INVERSE=False, slice_method = args.slice_method,
# 							LOCAL_DP_LP=LOCAL_DP_LP,
# 							mpi_comm = mpi_comm_first)


# t_s2 = train_model_spatial(model_gen2, mpi_comm_second.local_rank,batch_size,epochs=1, spatial_size=spatial_size, num_spatial_parts=num_spatial_parts ,criterion=None,optimizer=None,parts=parts,ASYNC=True,GEMS_INVERSE=True, slice_method = args.slice_method,
# 							LOCAL_DP_LP=LOCAL_DP_LP,
# 							mpi_comm = mpi_comm_second)

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


# sync_allreduce.sync_model_spatial(model_gen)
perf = []

sync_comm = gems_comm.SyncAllreduce(mpi_comm_first)


MASTER = args.times

print("ENABLE_MASTER_OPT", ENABLE_MASTER_OPT)


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
            # inputs=data_x
            # labels = data_y
            inputs, labels = data

            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # t= time.time()
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
    print("Mean {} Median {}".format(sum(perf) / len(perf), np.median(perf)))

exit()
