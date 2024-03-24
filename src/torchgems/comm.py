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
import os
import math
import numpy as np
from torchgems.utils import set_mpi_dist_environemnt, initialize_cuda






class MPIComm:
    def __init__(
        self,
        split_size,
        ENABLE_MASTER=False,
        ENABLE_SPATIAL=False,
        num_spatial_parts=None,
        spatial_size=None,
        LOCAL_DP_LP=1,
        DISABLE_INIT=False,
    ):
        self.ENABLE_MASTER = ENABLE_MASTER
        self.ENABLE_SPATIAL = ENABLE_SPATIAL

        self.split_size = split_size
        if ENABLE_SPATIAL == False:
            self.mp_size = split_size
        else:
            self.mp_size = (
                split_size
                + np.sum(num_spatial_parts)
                - spatial_size
                + (split_size - spatial_size) * (LOCAL_DP_LP - 1)
            )

        if DISABLE_INIT:
            self.rank = dist.get_rank()
            self.size = dist.get_world_size()
        else:
            self.size, self.rank = self.init_comm(backend="mpi")

        self.local_rank = self.rank % self.mp_size

        if self.ENABLE_MASTER:
            self.local_rank = self.mp_size - 1 - self.local_rank
            self.first_local_rank = self.mp_size - 1 - self.local_rank
            self.second_local_rank = self.local_rank

        self.num_spatial_parts = num_spatial_parts
        self.spatial_size = spatial_size
        self.LOCAL_DP_LP = LOCAL_DP_LP

        if ENABLE_SPATIAL == True and (
            num_spatial_parts == None or spatial_size == None
        ):
            assert (
                False
            ), "Spatial enabled but num_spatial_parts or spatial_size is None"

        if ENABLE_SPATIAL == True:
            if isinstance(num_spatial_parts, list):
                assert spatial_size == len(
                    num_spatial_parts
                ), "spatial size should be equal to elements in num_spatial_parts"
                self.total_spatial_processes = sum(num_spatial_parts)

                self.num_spatial_parts_list = num_spatial_parts
            else:
                self.total_spatial_processes = num_spatial_parts
        if ENABLE_SPATIAL == True:
            self.spatial_allreduce_grp = self.create_allreduce_comm_spatial()
        else:
            self.spatial_allreduce_grp = None

        # Finding split rank
        if ENABLE_SPATIAL == True:
            if self.local_rank < self.total_spatial_processes:
                self.split_rank = self.get_split_rank(
                    num_spatial_parts, self.local_rank
                )
            else:
                self.split_rank = (
                    math.floor(
                        (self.local_rank - self.total_spatial_processes)
                        / self.LOCAL_DP_LP
                    )
                    + spatial_size
                )
        else:
            self.split_rank = self.local_rank

        if LOCAL_DP_LP > 1:
            (
                self.LP_SP_Groups,
                self.SP_LP_group,
            ) = self.create_scatter_gather_spatial_MP_comm()
            self.LOCAL_DP_MP_Comm = self.create_local_DP_in_MP_comm()
            self.test_allreduce_comm(self.LOCAL_DP_MP_Comm)
        else:
            self.LP_SP_Groups, self.SP_LP_group = None, None
            self.LOCAL_DP_MP_Comm = None

        print("start_all_reduce")
        self.allreduce_grp = self.create_allreduce_comm()
        print("end_all_reduce")
        self.test_allreduce_comm(self.allreduce_grp)
        print("end_test_reduce")

    def get_split_rank(self, num_spatial_parts_list, local_rank):
        if isinstance(num_spatial_parts_list, list):
            temp_sum = 0
            split_rank = -1
            for parts in num_spatial_parts_list:
                if local_rank < temp_sum + parts:
                    split_rank += 1
                    return split_rank

                else:
                    temp_sum += parts
                    split_rank += 1
        else:
            return math.floor(local_rank / num_spatial_parts_list)

    def init_comm(self, backend="mpi"):
        print("starting init comm....")
        """Initialize the distributed environment."""
        set_mpi_dist_environemnt()
        dist.init_process_group(backend)
        size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        print("end init comm")
        print(f"Initialization completed with SIZE , {size} and Rank : {rank}")
        return size, rank

    def create_allreduce_comm_basic(self):
        # create allreduce comm for Hybrid MP-Basic
        ranks = [
            (self.mp_size * i) + self.local_rank
            for i in range(int(self.size / self.mp_size))
        ]
        allreduce_grp = torch.distributed.new_group(ranks=ranks)
        return allreduce_grp

    def create_allreduce_comm_master(self):
        # create allreduce comm for MASTER and Hybrid MASTER
        second_rank = self.size - self.local_rank - 1
        if self.ENABLE_SPATIAL:
            allreduce_grp = None
            for temp_rank in range(self.total_spatial_processes, self.mp_size):
                temp_first_rank = temp_rank
                temp_second_rank = self.mp_size - 1 - temp_first_rank
                ranks = [temp_first_rank, temp_second_rank]
                temp_allreduce_grp = torch.distributed.new_group(ranks=ranks)

                if self.first_local_rank in ranks:
                    self.first_LP_master_group = temp_allreduce_grp
                if self.second_local_rank in ranks:
                    self.second_LP_master_group = temp_allreduce_grp
        else:
            ranks = [
                temp_rank
                for temp_rank in range(self.size)
                if (
                    temp_rank % self.mp_size == self.local_rank
                    or temp_rank % self.mp_size == second_rank
                )
            ]
            allreduce_grp = torch.distributed.new_group(ranks=ranks)
        return allreduce_grp

    def create_allreduce_comm_spatial(self):
        if self.ENABLE_MASTER:
            first_local_rank = self.mp_size - 1 - self.local_rank
            second_local_rank = self.local_rank
        spatial_allreduce_grp = None
        for j in range(self.spatial_size):
            if self.spatial_size == 1:
                ranks = [
                    (self.num_spatial_parts * j) + i
                    for i in range(self.num_spatial_parts)
                ]
            else:
                ranks = [
                    (sum(self.num_spatial_parts_list[:j])) + i
                    for i in range(self.num_spatial_parts_list[j])
                ]

            if self.ENABLE_MASTER:
                for i in range(len(ranks)):
                    ranks.append(self.mp_size - 1 - ranks[i])

            temp_spatial_allreduce_grp = torch.distributed.new_group(ranks=ranks)

            if self.ENABLE_MASTER:
                if self.spatial_size == 1 and first_local_rank < self.num_spatial_parts:
                    self.first_spatial_allreduce_grp = temp_spatial_allreduce_grp

                elif (
                    self.spatial_size == 1
                    and second_local_rank < self.num_spatial_parts
                ):
                    self.second_spatial_allreduce_grp = temp_spatial_allreduce_grp

                elif self.spatial_size > 1:
                    if first_local_rank < np.sum(
                        self.num_spatial_parts[: j + 1]
                    ) and first_local_rank >= np.sum(self.num_spatial_parts[:j]):
                        self.first_spatial_allreduce_grp = temp_spatial_allreduce_grp

                    elif second_local_rank < np.sum(
                        self.num_spatial_parts[: j + 1]
                    ) and second_local_rank >= np.sum(self.num_spatial_parts[:j]):
                        self.second_spatial_allreduce_grp = temp_spatial_allreduce_grp

            if self.spatial_size == 1:
                spatial_allreduce_grp = temp_spatial_allreduce_grp
            elif self.local_rank < np.sum(
                self.num_spatial_parts[: j + 1]
            ) and self.local_rank >= np.sum(self.num_spatial_parts[:j]):
                spatial_allreduce_grp = temp_spatial_allreduce_grp

        return spatial_allreduce_grp

    def create_scatter_gather_spatial_MP_comm(self):
        if self.spatial_size == 1:
            prev_num_spatial_parts = self.num_spatial_parts
        else:
            prev_num_spatial_parts = self.num_spatial_parts_list[-1]

        start_last_spatial_rank = self.total_spatial_processes - prev_num_spatial_parts

        LP_ranks = [i + self.total_spatial_processes for i in range(self.LOCAL_DP_LP)]

        LP_SP_Groups = []

        SP_LP_group = None

        for j in range(prev_num_spatial_parts):
            temp_ranks = [start_last_spatial_rank + j] + LP_ranks

            if self.ENABLE_MASTER:
                for i in range(len(temp_ranks)):
                    temp_ranks[i] = self.mp_size - 1 - temp_ranks[i]
            temp_LP_SP_grp = torch.distributed.new_group(ranks=temp_ranks)
            LP_SP_Groups.append(temp_LP_SP_grp)

            if self.local_rank == start_last_spatial_rank + j:
                SP_LP_group = temp_LP_SP_grp

        return LP_SP_Groups, SP_LP_group

    def create_local_DP_in_MP_comm(self):
        num_mp_ranks = self.mp_size - self.total_spatial_processes
        LOCAL_DP_MP_Comm = None

        for j in range(int(num_mp_ranks / self.LOCAL_DP_LP)):
            start_rank = self.total_spatial_processes + (j * self.LOCAL_DP_LP)
            temp_ranks = [start_rank + i for i in range(self.LOCAL_DP_LP)]
            if self.ENABLE_MASTER:
                for i in range(len(temp_ranks)):
                    temp_ranks[i] = self.mp_size - 1 - temp_ranks[i]

            temp_LOCAL_DP_MP_Comm = torch.distributed.new_group(ranks=temp_ranks)

            if self.local_rank in temp_ranks:
                LOCAL_DP_MP_Comm = temp_LOCAL_DP_MP_Comm

        return LOCAL_DP_MP_Comm

    def create_allreduce_comm(self):
        if self.LOCAL_DP_LP > 1:
            return torch.distributed.new_group()
        if self.ENABLE_MASTER == False:
            return self.create_allreduce_comm_basic()
        else:
            return self.create_allreduce_comm_master()

    def test_allreduce_comm(self, allreduce_grp):
        tensor1 = torch.zeros(32, 32, 3, 3)
        tensor1 = tensor1.cuda()
        if allreduce_grp != None:
            dist.all_reduce(tensor1, op=dist.reduce_op.SUM, group=allreduce_grp)
        torch.cuda.synchronize()


def sync_comms_for_master(comm1, comm2):
    # MASTER related communicators are in comm2
    first_local_rank = comm1.local_rank
    second_local_rank = comm2.local_rank

    if first_local_rank < comm1.total_spatial_processes:
        comm1.spatial_allreduce_grp = comm2.first_spatial_allreduce_grp
        comm1.allreduce_grp_master = comm2.first_spatial_allreduce_grp

    if second_local_rank < comm1.total_spatial_processes:
        comm2.spatial_allreduce_grp = comm2.second_spatial_allreduce_grp
        comm2.allreduce_grp_master = comm2.second_spatial_allreduce_grp

    if comm1.LOCAL_DP_LP == 1:
        if first_local_rank >= comm1.total_spatial_processes:
            comm1.allreduce_grp = comm2.first_LP_master_group
            comm1.allreduce_grp_master = comm2.first_LP_master_group

        if second_local_rank >= comm1.total_spatial_processes:
            comm2.allreduce_grp = comm2.second_LP_master_group
            comm2.allreduce_grp_master = comm2.second_LP_master_group


class SyncAllreduce:
    def __init__(self, mpi_comm):
        self.ENABLE_MASTER = mpi_comm.ENABLE_MASTER
        self.mp_size = mpi_comm.mp_size
        self.size = mpi_comm.size
        self.local_rank = mpi_comm.local_rank
        self.allreduce_grp = mpi_comm.allreduce_grp
        self.rank = mpi_comm.rank

        # spatial parallelism
        self.num_spatial_parts = mpi_comm.num_spatial_parts
        self.spatial_size = mpi_comm.spatial_size
        self.spatial_allreduce_grp = mpi_comm.spatial_allreduce_grp

        if self.ENABLE_MASTER == True:
            self.divide_bs = 2 * (self.size / self.mp_size)
        elif self.spatial_size != None:
            if isinstance(self.num_spatial_parts, list):
                self.divide_bs = self.num_spatial_parts[0]
            else:
                self.divide_bs = self.num_spatial_parts

        else:
            self.divide_bs = self.size / self.mp_size

        # Global lists
        self.grad_shape_list1 = []
        self.grad_shape_list2 = []
        self.grad_num_element_list1 = []
        self.grad_num_element_list2 = []
        self.flag_grad_call_once1 = False
        self.flag_grad_call_once2 = False

    def sync_broadcast(self, model, src, grp_comm):
        for param in model.parameters():
            torch.distributed.broadcast(
                param.data, src=src, group=grp_comm, async_op=False
            )

    def sync_model_spatial(self, model_gen):
        if self.local_rank < self.spatial_size * self.num_spatial_parts:
            self.sync_broadcast(
                model_gen.models,
                src=math.floor(self.local_rank / self.num_spatial_parts),
                grp_comm=self.spatial_allreduce_grp,
            )

    def sync_model(self, model_gen1, model_gen2):
        if self.local_rank >= self.mp_size / 2:
            self.sync_broadcast(
                model_gen1.models, src=self.local_rank, grp_comm=self.allreduce_grp
            )
            self.sync_broadcast(
                model_gen2.models, src=self.local_rank, grp_comm=self.allreduce_grp
            )
        else:
            self.sync_broadcast(
                model_gen2.models,
                src=self.mp_size - self.local_rank - 1,
                grp_comm=self.allreduce_grp,
            )
            self.sync_broadcast(
                model_gen1.models,
                src=self.mp_size - self.local_rank - 1,
                grp_comm=self.allreduce_grp,
            )

    def get_grad_info(self, model):
        grad_shape_list = []
        grad_num_element_list = []
        for param in model.parameters():
            if param.grad is not None:
                temp = 1
                for i in param.shape:
                    temp = temp * i
                grad_shape_list.append(list(param.shape))
                grad_num_element_list.append(temp)
        return grad_shape_list, grad_num_element_list

    def get_grad_flatten(self, model, back=False):
        if back == False:
            if self.flag_grad_call_once1 == False:
                self.grad_shape_list1, self.grad_num_element_list1 = self.get_grad_info(
                    model
                )
                self.flag_grad_call_once1 = True
        else:
            if self.flag_grad_call_once2 == False:
                self.grad_shape_list2, self.grad_num_element_list2 = self.get_grad_info(
                    model
                )
                self.flag_grad_call_once2 = True

        flat_grad = None
        for param in model.parameters():
            if param.grad is not None:
                if flat_grad is None:
                    flat_grad = param.grad.data.clone().detach().view(-1)
                else:
                    flat_grad = torch.cat(
                        (flat_grad, param.grad.data.clone().detach().view(-1))
                    )

        return flat_grad

    def modify_grads(self, model, flat_grad, grad_num_element_list, grad_shape_list):
        temp_count_elements = 0
        temp_count_index = 0

        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = (
                    flat_grad[
                        temp_count_elements : temp_count_elements
                        + grad_num_element_list[temp_count_index]
                    ]
                    .view(grad_shape_list[temp_count_index])
                    .clone()
                    .detach()
                    / self.divide_bs
                )

                temp_count_elements += grad_num_element_list[temp_count_index]
                temp_count_index += 1

    def apply_allreduce_master(self, model_gen1, model_gen2):
        models1 = model_gen1.models
        models2 = model_gen2.models
        flat_grad1 = self.get_grad_flatten(models1, back=False)
        flat_grad2 = self.get_grad_flatten(models2, back=True)
        if self.local_rank >= self.mp_size / 2:
            dist.all_reduce(flat_grad1, op=dist.reduce_op.SUM, group=self.allreduce_grp)
            dist.all_reduce(flat_grad2, op=dist.reduce_op.SUM, group=self.allreduce_grp)
        else:
            dist.all_reduce(flat_grad2, op=dist.reduce_op.SUM, group=self.allreduce_grp)
            dist.all_reduce(flat_grad1, op=dist.reduce_op.SUM, group=self.allreduce_grp)

        self.modify_grads(
            models1, flat_grad1, self.grad_num_element_list1, self.grad_shape_list1
        )
        self.modify_grads(
            models2, flat_grad2, self.grad_num_element_list2, self.grad_shape_list2
        )

    def apply_allreduce_master_master(self, model_gen1, model_gen2, comm1, comm2):
        models1 = model_gen1.models
        models2 = model_gen2.models
        flat_grad1 = self.get_grad_flatten(models1, back=False)
        flat_grad2 = self.get_grad_flatten(models2, back=True)
        if comm1.split_rank <= comm2.split_rank:
            dist.all_reduce(
                flat_grad1, op=dist.reduce_op.SUM, group=comm1.allreduce_grp_master
            )
            dist.all_reduce(
                flat_grad2, op=dist.reduce_op.SUM, group=comm2.allreduce_grp_master
            )
        else:
            dist.all_reduce(
                flat_grad2, op=dist.reduce_op.SUM, group=comm2.allreduce_grp_master
            )
            dist.all_reduce(
                flat_grad1, op=dist.reduce_op.SUM, group=comm1.allreduce_grp_master
            )

        self.modify_grads(
            models1, flat_grad1, self.grad_num_element_list1, self.grad_shape_list1
        )
        self.modify_grads(
            models2, flat_grad2, self.grad_num_element_list2, self.grad_shape_list2
        )

    def apply_allreduce(self, model_gen, allreduce_grp):
        torch.cuda.synchronize()
        models1 = model_gen.models
        flat_grad1 = self.get_grad_flatten(models1, back=False)
        dist.all_reduce(flat_grad1.data, op=dist.reduce_op.SUM, group=allreduce_grp)
        self.modify_grads(
            models1, flat_grad1, self.grad_num_element_list1, self.grad_shape_list1
        )
        torch.cuda.synchronize()

    def apply_allreduce_master_and_update(self, tm_master, model_gen1, model_gen2):
        torch.cuda.synchronize()
        self.apply_allreduce_master(model_gen1, model_gen2)
        torch.cuda.synchronize()
        tm_master.train_model1.update()
        tm_master.train_model2.update()
        torch.cuda.synchronize()
