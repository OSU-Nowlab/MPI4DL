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

import torch.nn as nn
import torch.optim as optim
import torch
import torch.distributed as dist
from collections import OrderedDict
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from .utils import print_model_size


class model_generator:
    def __init__(self, model, split_size, input_size, balance=None, shape_list=None):
        self.model = model
        self.input_size = input_size
        self.split_size = split_size
        self.balance = balance
        self.shape_list = shape_list

        if balance is not None:
            assert (
                len(balance) == split_size
            ), "Length of balance should be equal to split size "

    def get_start_end_layer_index(self, split_rank):
        # return the index of start and end layer for the model
        # based on the size of model parallelism and local rank of the process
        # it can be modified by giving balance parameter
        if self.balance == None:
            num_layers = len(self.model)
            part_layer = int(num_layers / self.split_size)
            start_layer = split_rank * part_layer

            if split_rank != self.split_size - 1:
                end_layer = (split_rank + 1) * part_layer
            else:
                end_layer = len(self.model)
        else:
            num_layers = len(self.model)
            assert sum(self.balance) == len(
                self.model
            ), "balance and number of layers differs"

            if split_rank == 0:
                start_layer = 0
                end_layer = self.balance[0]
                part_layer = self.balance[0]
            else:
                start_layer = sum(self.balance[:split_rank])
                end_layer = sum(self.balance[: split_rank + 1])
                part_layer = end_layer - start_layer

        return start_layer, end_layer

    def get_model(self, split_rank):
        layers = OrderedDict()

        start_layer, end_layer = self.get_start_end_layer_index(split_rank)

        # Dividing model
        i = 0
        for name, layer in self.model.named_children():
            if i >= start_layer and i < end_layer:
                layers[name] = layer
            i += 1

        return nn.Sequential(layers)

    def ModelQuantizaton(self, model, split_rank, model_no):
        import torch_tensorrt
        from .utils import load_fake_dataset

        # import os

        print("In ModelQunarization....")

        quantized_model_path = f"/home/gulhane.2/MPI4DL_Copy/MPI4DL/benchmarks/MPI4DL_Checkpoints/gems/split_size{self.split_size}_rank{split_rank}_model{model_no}"
        print(quantized_model_path)
        # if os.path.exists(quantized_model_path):
        if False:
            model = torch.jit.load(quantized_model_path)
            model.eval()
            model.to("cuda")
            return model

        # Create and save quantized model
        checkpoint_path = "/home/gulhane.2/github_torch_gems/MPI4DL/benchmarks/MPI4DL_Checkpoints/MPI4DL_ImageNeteee.pth"
        checkpoint = torch.load(checkpoint_path)
        model_state_dist_split_layer = {}

        # for name, _ in model.named_parameters():
        #     print(name)
        # model_state_dist_split_layer[name] = checkpoint["model_state_dict"][name]
        # if ".bias" in name and ".batch" in name:
        #     l_name = ".".join(name.split(".")[:-1])
        #     running_mean = l_name + ".running_mean"
        #     running_var = l_name + ".running_var"
        #     model_state_dist_split_layer[running_mean] = checkpoint[
        #         "model_state_dict"
        #     ][running_mean]
        #     model_state_dist_split_layer[running_var] = checkpoint[
        #         "model_state_dict"
        #     ][running_var]

        # model.load_state_dict(model_state_dist_split_layer)
        model.eval()
        if split_rank == 0:
            batch_size = self.input_size[0]
            channel = self.input_size[1]
            image_size1 = self.input_size[2]
            image_size2 = self.input_size[3]
        else:
            batch_size = self.shape_list[split_rank - 1][0]
            channel = self.shape_list[split_rank - 1][1]
            image_size1 = self.shape_list[split_rank - 1][2]
            image_size2 = self.shape_list[split_rank - 1][3]
        # batch_size = 1
        # channel = 3
        # image_size = 64

        testing_dataloader = load_fake_dataset(
            batch_size, channel, image_size1, image_size2
        )
        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            # cache_file="./calibration.cache",
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda"),
        )
        trt_mod = torch_tensorrt.compile(
            model,
            inputs=[
                torch_tensorrt.Input((batch_size, channel, image_size1, image_size2))
            ],
            enabled_precisions={torch.int8},
            calibrator=calibrator,
            device={
                "device_type": torch_tensorrt.DeviceType.GPU,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                #  "disable_tf32": False
            },
        )
        # torch.jit.save(torch.jit.script(trt_mod), quantized_model_path)
        print("Completed tensort model {}")
        return trt_mod

    def ready_model(
        self,
        split_rank,
        GET_SHAPES_ON_CUDA=False,
        eval_mode=False,
        checkpoint_path=None,
        precision=None,
    ):
        if self.shape_list == None:
            self.get_output_shapes(GET_SHAPES_ON_CUDA)
        temp_model = self.get_model(split_rank=split_rank)
        t = time.time()
        if eval_mode == False:
            self.models = temp_model.to("cuda")
            return
        assert precision is not None, "precision req for evaluatoin mode"
        if (
            checkpoint_path is None
        ):  # TBD: for testing purpose, we will be using evaluation on random weights for collecting throughput
            self.models = temp_model
            self.models.eval()
            print(f"precision : {precision}")
            if precision == "fp_16":
                self.models.half()
            elif precision == "int_8":
                self.models = self.ModelQuantizaton(temp_model, split_rank, model_no=1)
            self.models.to("cuda")
            print_model_size(self.models, split_rank, True)

            for param in self.models.parameters():
                param.requires_grad = False
            return

        # eval_mode is True
        assert checkpoint_path is not None, "No checkpoints found"

        checkpoint = torch.load(checkpoint_path)
        model_state_dist_split_layer = {}
        self.models = temp_model

        for name, _ in self.models.named_parameters():
            model_state_dist_split_layer[name] = checkpoint["model_state_dict"][name]
            if ".bias" in name and ".batch" in name:
                l_name = ".".join(name.split(".")[:-1])
                running_mean = l_name + ".running_mean"
                running_var = l_name + ".running_var"
                model_state_dist_split_layer[running_mean] = checkpoint[
                    "model_state_dict"
                ][running_mean]
                model_state_dist_split_layer[running_var] = checkpoint[
                    "model_state_dict"
                ][running_var]

        self.models.load_state_dict(model_state_dist_split_layer)
        if precision == "fp_16":
            self.models.half()
        self.models.eval()
        print_model_size(self.models, split_rank, True)
        self.models.to("cuda")

    def DDP_model(
        self, mpi_comm, num_spatial_parts, spatial_size, bucket_size=25, local_rank=None
    ):
        if local_rank == None:
            local_rank = mpi_comm.local_rank

        if local_rank < mpi_comm.total_spatial_processes:
            self.models = DDP(
                self.models,
                device_ids=[0],
                bucket_cap_mb=bucket_size,
                process_group=mpi_comm.spatial_allreduce_grp,
                broadcast_buffers=False,
            )
        elif mpi_comm.LOCAL_DP_LP > 1:
            if local_rank >= mpi_comm.total_spatial_processes:
                None
                self.models = DDP(
                    self.models,
                    device_ids=[0],
                    bucket_cap_mb=bucket_size,
                    process_group=mpi_comm.LOCAL_DP_MP_Comm,
                )
        elif mpi_comm.LOCAL_DP_LP == 1:
            if local_rank >= mpi_comm.total_spatial_processes:
                None
                self.models = DDP(
                    self.models,
                    device_ids=[0],
                    bucket_cap_mb=bucket_size,
                    process_group=mpi_comm.allreduce_grp,
                    broadcast_buffers=False,
                )

    def get_output_shapes(self, GET_SHAPES_ON_CUDA):
        self.shape_list = []

        temp_dev = "cuda" if GET_SHAPES_ON_CUDA else "cpu"

        orig_input_size = self.input_size
        input_size = list(self.input_size)
        input_size[0] = 1

        temp = torch.zeros(input_size, device=temp_dev)
        for i in range(self.split_size):
            model_x = self.get_model(split_rank=i)

            if GET_SHAPES_ON_CUDA:
                model_x = model_x.to("cuda")
            y = model_x(temp)

            if isinstance(y, tuple):
                # model has multiple outputs
                temp = []
                temp_shape = []
                for one_tensor in y:
                    t_s = list(one_tensor.shape)
                    t_s[0] = orig_input_size[0]
                    t_s = tuple(t_s)
                    temp_shape.append(t_s)

                    # creating input for next model partition
                    temp.append(torch.zeros(one_tensor.shape, device=temp_dev))
                temp = tuple(temp)

                self.shape_list.append(temp_shape)

            else:
                t_s = list(y.shape)
                t_s[0] = orig_input_size[0]
                t_s = tuple(t_s)
                temp = torch.zeros(y.shape, device=temp_dev)
                self.shape_list.append(t_s)

            # delete model and collect memory from GPU
            del model_x
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()


class train_model:
    def __init__(
        self,
        model_gen,
        local_rank,
        batch_size,
        epochs,
        precision,
        eval_mode,
        criterion=None,
        optimizer=None,
        parts=1,
        ASYNC=True,
        GEMS_INVERSE=False,
    ):
        self.models = model_gen.models
        self.shape_list = model_gen.shape_list
        self.input_size = model_gen.input_size
        # self.mp_size = model_gen.split_size
        self.parts = parts
        self.epochs = epochs
        self.local_rank = local_rank
        self.precision = precision
        self.ENABLE_ASYNC = ASYNC
        self.GEMS_INVERSE = GEMS_INVERSE
        self.batch_size = batch_size

        try:
            self.num_spatial_parts
        except AttributeError:
            self.num_spatial_parts = 1

        try:
            self.split_rank
        except AttributeError:
            self.split_rank = local_rank

        try:
            self.mp_size
        except AttributeError:
            self.mp_size = model_gen.split_size

        try:
            self.split_size
        except AttributeError:
            self.split_size = self.mp_size

        if isinstance(self.shape_list[self.split_rank - 1], list):
            self.MULTIPLE_INPUT = True
        else:
            self.MULTIPLE_INPUT = False

        if isinstance(self.shape_list[self.split_rank], list):
            self.MULTIPLE_OUTPUT = True
        else:
            self.MULTIPLE_OUTPUT = False

        if criterion == None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        if not eval_mode and optimizer == None:
            # self.optimizer = optim.SGD(self.models.parameters(), lr=0.000000001, momentum=0.9)
            self.optimizer = optim.SGD(self.models.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optimizer
        self.initialize_recv_buffers()
        self.initialize_send_recv_ranks()

    def initialize_send_recv_ranks(self):
        if self.GEMS_INVERSE == False:
            self.to_send_forward = self.local_rank + 1
            self.to_recv_forward = self.local_rank - 1
            self.to_send_backward = self.local_rank - 1
            self.to_recv_backward = self.local_rank + 1
        else:
            self.to_send_forward = self.mp_size - 1 - self.local_rank - 1
            self.to_recv_forward = self.mp_size - 1 - self.local_rank + 1
            self.to_send_backward = self.mp_size - 1 - self.local_rank + 1
            self.to_recv_backward = self.mp_size - 1 - self.local_rank - 1

    def initialize_recv_buffers(self):
        self.input_x_list = []
        datatype = torch.float32
        if self.precision == "fp_16":
            datatype = torch.float16

        # intializing recv buffer for the input
        # For parts we need different buffers as in backward pass we using grad variable to
        # send partial errors to previous partition
        for part in range(self.parts):
            input_x = []
            if self.split_rank != 0:
                # multiple inputs
                if isinstance(self.shape_list[self.split_rank - 1], list):
                    for i in range(len(self.shape_list[self.split_rank - 1])):
                        one_input = torch.zeros(
                            self.shape_list[self.split_rank - 1][i],
                            requires_grad=True,
                            device="cuda",
                            dtype=datatype,
                        )
                        input_x.append(one_input)
                    input_x = tuple(input_x)
                else:
                    input_x = torch.zeros(
                        self.shape_list[self.split_rank - 1],
                        requires_grad=True,
                        device="cuda",
                        dtype=datatype,
                    )

            self.input_x_list.append(input_x)

        # recv buffer for receiving partial errors

        if self.split_rank != self.split_size - 1:
            if isinstance(self.shape_list[self.split_rank], list):
                self.grad_overhead = []
                for i in range(len(self.shape_list[self.split_rank])):
                    temp_grad = torch.zeros(
                        self.shape_list[self.split_rank][i], device="cuda"
                    )

                    self.grad_overhead.append(temp_grad)
            else:
                self.grad_overhead = torch.zeros(
                    self.shape_list[self.split_rank], device="cuda"
                )

    def receive_input_sync(self, part_number):
        tag_forward = 0
        # multiple inputs
        if self.MULTIPLE_INPUT:
            for i in range(len(self.shape_list[self.split_rank - 1])):
                dist.recv(
                    tensor=self.input_x_list[part_number][i],
                    src=self.to_recv_forward,
                    tag=tag_forward,
                )
                tag_forward += 1

        else:
            dist.recv(
                tensor=self.input_x_list[part_number],
                src=self.to_recv_forward,
                tag=tag_forward,
            )

    def receive_input_async(self, part_number):
        tag_forward = 0
        if self.MULTIPLE_INPUT:
            reqs = []

            for i in range(len(self.shape_list[self.split_rank - 1])):
                req_temp = dist.irecv(
                    tensor=self.input_x_list[part_number][i],
                    src=self.to_recv_forward,
                    tag=tag_forward,
                )
                reqs.append(req_temp)
                tag_forward += 1

            for req in reqs:
                req.wait()

        else:
            dist.recv(
                tensor=self.input_x_list[part_number],
                src=self.to_recv_forward,
                tag=tag_forward,
            )

    def send_input_sync(self, y):
        tag_forward = 0

        if self.MULTIPLE_OUTPUT:
            # Mutlitple inputs
            for one_output in y:
                dist.send(tensor=one_output, dst=self.to_send_forward, tag=tag_forward)
                tag_forward += 1
        else:
            dist.send(tensor=y, dst=self.to_send_forward, tag=tag_forward)

    def send_input_async(self, y):
        tag_forward = 0

        if self.MULTIPLE_OUTPUT:
            reqs = []
            for one_output in y:
                req = dist.isend(
                    tensor=one_output, dst=self.to_send_forward, tag=tag_forward
                )
                tag_forward += 1
                reqs.append(req)

            for req in reqs:
                req.wait()
        else:
            dist.send(tensor=y, dst=self.to_send_forward, tag=tag_forward)

    def receive_grad_sync(self):
        tag_forward = 0
        # multiple inputs
        if self.MULTIPLE_OUTPUT:
            for i in range(len(self.shape_list[self.split_rank])):
                dist.recv(
                    tensor=self.grad_overhead[i],
                    src=self.to_recv_backward,
                    tag=tag_forward,
                )
                tag_forward += 1

        else:
            dist.recv(
                tensor=self.grad_overhead, src=self.to_recv_backward, tag=tag_forward
            )

    def receive_grad_async(self):
        tag_forward = 0
        if self.MULTIPLE_OUTPUT:
            reqs = []

            for i in range(len(self.shape_list[self.split_rank])):
                req_temp = dist.irecv(
                    tensor=self.grad_overhead[i],
                    src=self.to_recv_backward,
                    tag=tag_forward,
                )
                reqs.append(req_temp)
                tag_forward += 1

            for req in reqs:
                req.wait()

        else:
            dist.recv(
                tensor=self.grad_overhead, src=self.to_recv_backward, tag=tag_forward
            )

    def send_grad_sync(self, input_x):
        tag_forward = 0

        if self.MULTIPLE_INPUT:
            # Mutlitple inputs
            for i in len(input_x):
                dist.send(
                    tensor=input_x[i].grad, dst=self.to_send_backward, tag=tag_forward
                )
                tag_forward += 1
        else:
            dist.send(tensor=input_x.grad, dst=self.to_send_backward, tag=tag_forward)

    def send_grad_async(self, input_x):
        tag_forward = 0

        if self.MULTIPLE_INPUT:
            reqs = []
            for i in range(len(input_x)):
                req = dist.isend(
                    tensor=input_x[i].grad, dst=self.to_send_backward, tag=tag_forward
                )
                tag_forward += 1
                reqs.append(req)

            for req in reqs:
                req.wait()
        else:
            dist.send(tensor=input_x.grad, dst=self.to_send_backward, tag=tag_forward)

    def forward_pass(self, data_x, data_y, part_number=0):
        # data_x: input data
        # data_y: labels
        # part_number: part number between 0 and self.parts-1 used to find right input recv buffer

        # Receive inputs if local is not 0
        if self.split_rank == 0:
            input_x = data_x
        else:
            if self.ENABLE_ASYNC == True:
                self.receive_input_async(part_number)
            else:
                self.receive_input_sync(part_number)

            if self.MULTIPLE_INPUT:
                input_x = tuple(self.input_x_list[part_number])
            else:
                input_x = self.input_x_list[part_number]

        # Apply forward pass
        torch.cuda.synchronize()

        y = self.models(input_x)

        torch.cuda.synchronize()

        if self.split_rank != self.split_size - 1:
            if self.ENABLE_ASYNC == True:
                self.send_input_async(y)
            else:
                self.send_input_sync(y)

        else:
            if (
                y.dtype != torch.float32
            ):  # converting to float32 to avoid RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Half'
                y = y.to(dtype=torch.float32)
            loss = self.criterion(y, data_y)

        if self.split_rank == self.split_size - 1:
            corrects = (data_y.eq(torch.argmax(y, dim=-1).long())).sum().float()
            return loss, corrects / self.batch_size
        else:
            return y, None

    def backward_pass(self, y, part_number=0):
        if self.split_rank != self.split_size - 1:
            if self.ENABLE_ASYNC:
                self.receive_grad_async()
            else:
                self.receive_grad_sync()

        torch.cuda.synchronize()
        if self.split_rank == self.split_size - 1:
            y.backward()
        else:
            torch.autograd.backward(y, self.grad_overhead)
        torch.cuda.synchronize()

        if self.split_rank != 0:
            if self.ENABLE_ASYNC:
                self.send_grad_async(self.input_x_list[part_number])
            else:
                self.send_grad_sync(self.input_x_list[part_number])

        if self.split_rank != 0:
            if self.MULTIPLE_INPUT:
                self.input_x_list[part_number] = list(self.input_x_list[part_number])
                for i in range(len(self.input_x_list[part_number])):
                    self.input_x_list[part_number][i] = (
                        self.input_x_list[part_number][i].detach().requires_grad_()
                    )
                self.input_x_list[part_number] = tuple(self.input_x_list[part_number])

            else:
                self.input_x_list[part_number] = (
                    self.input_x_list[part_number].detach().requires_grad_()
                )

    def run_step(self, data_x, data_y, eval_mode):
        data_x = data_x.to("cuda")
        data_y = data_y.to("cuda")

        parts_size = int(self.batch_size / self.parts)

        y_list = []
        loss = 0
        corrects = 0
        for i in range(self.parts):
            start = i * parts_size
            end = (i + 1) * parts_size
            temp_y, temp_correct = self.forward_pass(
                data_x[start:end], data_y[start:end], part_number=i
            )
            y_list.append(temp_y)

            if self.split_rank == self.split_size - 1:
                loss += temp_y.item()
                corrects += temp_correct.item()

        if eval_mode == False:
            for i in range(self.parts):
                None
                self.backward_pass(y_list[i], part_number=i)

        return loss, corrects

    def update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
