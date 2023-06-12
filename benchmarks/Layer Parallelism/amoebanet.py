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
from collections import OrderedDict
from torchgems import parser
import time
from torchgems.mp_pipeline import model_generator, train_model

from models import resnet_cifar_torch

from models import amoebanet 
import torchgems.comm as gems_comm 



parser_obj = parser.get_parser()
args = parser_obj.parse_args()


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

#torch.set_num_threads(1)
np.random.seed(seed=1405)
parts =args.parts
batch_size = args.batch_size
resnet_n = 18
epoch=args.num_epochs
ENABLE_ASYNC=True
ENABLE_APP=False
amoebanet_test = False
image_size = int(args.image_size) #1024
print("image size", image_size)
steps = 100
num_layers = args.num_layers
num_filters = args.num_filters
balance = args.balance
mp_size = args.split_size
image_size_seq = 512

MODEL_TYPE=2
num_classes=1000

mpi_comm = gems_comm.MPIComm(split_size=mp_size ,ENABLE_MASTER=False)
rank = mpi_comm.rank



local_rank = rank % mp_size

if(balance is not None):
	balance = [int(i) for i in balance.split(',')]





def get_depth(version,n):
	if version == 1:
		return n * 6 + 2
	elif version == 2:
		return n * 9 + 2

if(MODEL_TYPE==1):
	model = resnet_cifar_torch.get_resnet_v1((int(batch_size/parts),3,image_size_seq,image_size_seq),get_depth(1,resnet_n))
	# model = resnet_cifar_torch.get_resnet_v2(
    #     input_shape=(batch_size / parts, 3, image_size, image_size),
    #     depth=get_depth(2, 12))
elif(MODEL_TYPE==2):
	
	model = amoebanet.amoebanetd(num_classes=1000, num_layers=args.num_layers, num_filters=args.num_filters)


mul_shape = int(args.image_size/32)
#model_gen = model_generator(model=model, split_size=mp_size,input_size=(int(batch_size/parts),3,image_size,image_size),balance=None,shape_list=[(2, 16, 32*mul_shape, 32*mul_shape), (2, 32, 16*mul_shape, 16*mul_shape), (2, 64, 8*mul_shape, 8*mul_shape), (2, 10)])
#model_gen = model_generator(model=model, split_size=mp_size,input_size=(int(batch_size/parts),3,image_size_seq,image_size_seq),balance=balance)
model_gen = model_generator(
    model=model,
    split_size=mp_size,
    input_size=(int(batch_size / parts), 3, image_size, image_size),
    balance=balance
)
model_gen.ready_model(split_rank=local_rank, GET_SHAPES_ON_CUDA=True)


image_size_times = int(image_size / image_size_seq)
resnet_shapes_list = []
for output_shape in model_gen.shape_list:
	if(isinstance(output_shape, list)):
		temp_shape = []
		for shape_tuple in output_shape:


			x = (shape_tuple[0], shape_tuple[1], int(shape_tuple[2]* image_size_times), int(shape_tuple[3]* image_size_times) )
			temp_shape.append(x)
		resnet_shapes_list.append(temp_shape)
	else:

		if(len(output_shape) == 2):
			resnet_shapes_list.append(output_shape)
		else:
			
			x = (output_shape[0], output_shape[1], int(output_shape[2]* image_size_times), int(output_shape[3]* image_size_times) )
			resnet_shapes_list.append(x)

model_gen.shape_list = resnet_shapes_list
print("local_ran:",local_rank," Shapes:",model_gen.shape_list)


del model_gen
del model
torch.cuda.ipc_collect()

# remove to save memory 

if(MODEL_TYPE==1):
	model = resnet_cifar_torch.get_resnet_v1((int(batch_size/parts),3,image_size_seq,image_size_seq),get_depth(1,resnet_n))
elif(MODEL_TYPE==2):
	
	model = amoebanet.amoebanetd(num_classes=1000, num_layers=args.num_layers, num_filters=args.num_filters)


model_gen = model_generator(model=model, split_size=mp_size,input_size=(int(batch_size/parts),3,image_size,image_size),balance=balance, shape_list=resnet_shapes_list)
model_gen.ready_model(split_rank=local_rank, GET_SHAPES_ON_CUDA=True)


tm = train_model(model_gen, local_rank,batch_size,epoch,criterion=None,optimizer=None,parts=parts,ASYNC=ENABLE_ASYNC)



#Dataset 
transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
torch.manual_seed(0)
if(ENABLE_APP==True):
	trainset = torchvision.datasets.ImageFolder("/usr/workspace/jain8/project/cancer/1024_1024_5/train", transform=transform, target_transform=None)
	my_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											  shuffle=True, num_workers=0,pin_memory=True)
else:

	my_dataset=torchvision.datasets.FakeData(size=10*batch_size, image_size=(3, image_size, image_size), num_classes=num_classes, transform=transform, target_transform=None, random_offset=0)
	my_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
											  shuffle=False, num_workers=0,pin_memory=True)

perf = []
def run_epoch():

	for i_e in range(epoch):
		loss =0
		t = time.time()
		for i, data in enumerate(my_dataloader, 0):
			if(i>math.floor(50000/batch_size)-1):
				break
			#inputs=data_x
			#labels = data_y
			start_event = torch.cuda.Event(enable_timing=True, blocking=True)
			end_event = torch.cuda.Event(enable_timing=True, blocking=True)
			start_event.record()


			inputs, labels = data
			#inputs = inputs.to(device)
			#labels = labels.to(device)

			#t= time.time()

			temp_loss = tm.run_step(inputs, labels)
			tm.update()

			end_event.record()
			torch.cuda.synchronize()
			t = start_event.elapsed_time(end_event) / 1000


			if(local_rank==mp_size-1):
				None
				#print("LOSS",temp_loss,)

			if(local_rank==0):
				print("Epoch: {} images per sec:{}".format(i_e,batch_size/t))
				perf.append(batch_size/t)

			t = time.time()


run_epoch()

if(local_rank==0):
	print("Mean {} Median {}".format(sum(perf)/len(perf), np.median(perf)))
