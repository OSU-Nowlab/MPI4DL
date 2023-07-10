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
parts = args.parts
batch_size = args.batch_size
resnet_n = 18
epoch = args.num_epochs
ENABLE_ASYNC = True
ENABLE_APP = False
amoebanet_test = False
image_size = int(args.image_size)
print("image size", image_size)
steps = 100
num_layers = args.num_layers
num_filters = args.num_filters
balance = args.balance
mp_size = args.split_size
image_size_seq = 512
times = 1
num_classes = 1000

mpi_comm = gems_comm.MPIComm(split_size=mp_size, ENABLE_MASTER=False)
rank = mpi_comm.rank

local_rank = rank % mp_size

if balance is not None:
    balance = [int(i) for i in balance.split(",")]

model = amoebanet.amoebanetd(
    num_classes=1000, num_layers=args.num_layers, num_filters=args.num_filters
)

model_gen = model_generator(
    model=model,
    split_size=mp_size,
    input_size=(int(batch_size / parts), 3, image_size_seq, image_size_seq),
    balance=balance,
)
model_gen.ready_model(split_rank=local_rank, GET_SHAPES_ON_CUDA=True)


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
print("local_ran:", local_rank, " Shapes:", model_gen.shape_list)


del model_gen
del model
torch.cuda.ipc_collect()

model = amoebanet.amoebanetd(
    num_classes=1000, num_layers=args.num_layers, num_filters=args.num_filters
)


model_gen = model_generator(
    model=model,
    split_size=mp_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=balance,
    shape_list=resnet_shapes_list,
)
model_gen.ready_model(split_rank=local_rank, GET_SHAPES_ON_CUDA=True)

tm = train_model(
    model_gen,
    local_rank,
    batch_size,
    epoch,
    criterion=None,
    optimizer=None,
    parts=parts,
    ASYNC=ENABLE_ASYNC,
)

# Dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
torch.manual_seed(0)
if ENABLE_APP == True:
    trainset = torchvision.datasets.ImageFolder(
        "/train", transform=transform, target_transform=None
    )
    my_dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
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

perf = []


def run_epoch():
    for i_e in range(epoch):
        loss = 0
        t = time.time()
        for i, data in enumerate(my_dataloader, 0):
            start_event = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event.record()

            if i > math.floor(size_dataset / (times * batch_size)) - 1:
                break
            inputs, labels = data

            temp_loss = tm.run_step(inputs, labels)
            loss += temp_loss
            tm.update()

            end_event.record()
            torch.cuda.synchronize()
            t = start_event.elapsed_time(end_event) / 1000

            if local_rank == mp_size - 1:
                logging.info(f"Step :{i}, LOSS: {temp_loss}, Global loss: {loss/(i+1)}")

            if local_rank == 0:
                print("Epoch: {} images per sec:{}".format(i_e, batch_size / t))
                perf.append(batch_size / t)

            t = time.time()


run_epoch()

if local_rank == 0:
    print("Mean {} Median {}".format(sum(perf) / len(perf), np.median(perf)))
