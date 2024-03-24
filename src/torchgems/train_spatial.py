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

from torchgems.mp_pipeline import train_model
import torch
import math
import torch.distributed as dist
from torchgems.utils import isPowerTwo

"""
For SP, image size and image size after partitioning should be power of two.
As, while performing convolution operations at different layers, odd input size
(i.e. image size which is not power of 2) will lead to truncation of input. Thus,
other GPU devices will receive truncated input with unexpected input size.
"""


def verify_spatial_config(slice_method, image_size, num_spatial_parts_list):
    spatial_part_size = num_spatial_parts_list[
        0
    ]  # Partition size for spatial parallelism

    assert slice_method in [
        "square",
        "vertical",
        "horizontal",
    ], "Possible slice methods are ['square', 'vertical', 'horizontal']"

    assert isPowerTwo(int(image_size)), "Image size should be power of Two"

    if slice_method == "square":
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


def get_shapes_spatial(
    shape_list, slice_method, spatial_size, num_spatial_parts_list, image_size_times
):
    temp_count = 0
    spatial_shapes_list = []
    spatial_part_size = num_spatial_parts_list[0]

    if slice_method == "square":
        spatial_shapes_list = []
        for output_shape in shape_list:
            if isinstance(output_shape, list):
                temp_shape = []
                for shape_tuple in output_shape:
                    if temp_count < spatial_size:
                        # reduce shape only when it is smaller than spatial size
                        x = (
                            int(shape_tuple[0]),
                            shape_tuple[1],
                            int(
                                shape_tuple[2]
                                * image_size_times
                                / math.sqrt(spatial_part_size)
                            ),
                            int(
                                shape_tuple[3]
                                * image_size_times
                                / math.sqrt(spatial_part_size)
                            ),
                        )
                        temp_shape.append(x)
                    else:
                        x = (
                            int(shape_tuple[0]),
                            shape_tuple[1],
                            int(shape_tuple[2] * image_size_times),
                            int(shape_tuple[3] * image_size_times),
                        )
                        temp_shape.append(x)
                spatial_shapes_list.append(temp_shape)
            else:
                if len(output_shape) == 2:
                    x = (int(output_shape[0]), output_shape[1])
                    spatial_shapes_list.append(x)
                else:
                    if temp_count < spatial_size:
                        x = (
                            int(output_shape[0]),
                            output_shape[1],
                            int(
                                output_shape[2]
                                * image_size_times
                                / math.sqrt(spatial_part_size)
                            ),
                            int(
                                output_shape[3]
                                * image_size_times
                                / math.sqrt(spatial_part_size)
                            ),
                        )
                        spatial_shapes_list.append(x)
                    else:
                        x = (
                            int(output_shape[0]),
                            output_shape[1],
                            int(output_shape[2] * image_size_times),
                            int(output_shape[3] * image_size_times),
                        )
                        spatial_shapes_list.append(x)
            temp_count += 1

    elif slice_method == "vertical":
        spatial_shapes_list = []
        for output_shape in shape_list:
            if isinstance(output_shape, list):
                temp_shape = []
                for shape_tuple in output_shape:
                    if temp_count < spatial_size:
                        x = (
                            int(shape_tuple[0]),
                            shape_tuple[1],
                            int(shape_tuple[2] * image_size_times / 1),
                            int(
                                shape_tuple[3]
                                * image_size_times
                                / num_spatial_parts_list[temp_count]
                            ),
                        )
                        temp_shape.append(x)
                    else:
                        x = (
                            int(shape_tuple[0]),
                            shape_tuple[1],
                            int(shape_tuple[2] * image_size_times),
                            int(shape_tuple[3] * image_size_times),
                        )
                        temp_shape.append(x)
                spatial_shapes_list.append(temp_shape)
            else:
                if len(output_shape) == 2:
                    x = (int(output_shape[0]), output_shape[1])
                    spatial_shapes_list.append(x)
                else:
                    if temp_count < spatial_size:
                        x = (
                            int(output_shape[0]),
                            output_shape[1],
                            int(output_shape[2] * image_size_times / 1),
                            int(
                                output_shape[3]
                                * image_size_times
                                / num_spatial_parts_list[temp_count]
                            ),
                        )
                        spatial_shapes_list.append(x)
                    else:
                        x = (
                            int(output_shape[0]),
                            output_shape[1],
                            int(output_shape[2] * image_size_times),
                            int(output_shape[3] * image_size_times),
                        )
                        spatial_shapes_list.append(x)
            temp_count += 1

    elif slice_method == "horizontal":
        spatial_shapes_list = []
        for output_shape in shape_list:
            if isinstance(output_shape, list):
                temp_shape = []
                for shape_tuple in output_shape:
                    if temp_count < spatial_size:
                        x = (
                            int(shape_tuple[0]),
                            shape_tuple[1],
                            int(
                                shape_tuple[2]
                                * image_size_times
                                / num_spatial_parts_list[temp_count]
                            ),
                            int(shape_tuple[3] * image_size_times / 1),
                        )
                        temp_shape.append(x)
                    else:
                        x = (
                            int(shape_tuple[0]),
                            shape_tuple[1],
                            int(shape_tuple[2] * image_size_times),
                            int(shape_tuple[3] * image_size_times),
                        )
                        temp_shape.append(x)
                spatial_shapes_list.append(temp_shape)
            else:
                if len(output_shape) == 2:
                    x = (int(output_shape[0]), output_shape[1])
                    spatial_shapes_list.append(x)
                else:
                    if temp_count < spatial_size:
                        x = (
                            int(output_shape[0]),
                            output_shape[1],
                            int(
                                output_shape[2]
                                * image_size_times
                                / num_spatial_parts_list[temp_count]
                            ),
                            int(output_shape[3] * image_size_times / 1),
                        )
                        spatial_shapes_list.append(x)
                    else:
                        x = (
                            int(output_shape[0]),
                            output_shape[1],
                            int(output_shape[2] * image_size_times),
                            int(output_shape[3] * image_size_times),
                        )
                        spatial_shapes_list.append(x)
            temp_count += 1
    return spatial_shapes_list


def split_input(inputs, image_size, slice_method, local_rank, num_spatial_parts_list):
    spatial_part_size = num_spatial_parts_list[
        0
    ]  # Partition size for spatial parallelism
    if slice_method == "square":
        image_height_local = int(image_size / math.sqrt(spatial_part_size))
        image_width_local = int(image_size / math.sqrt(spatial_part_size))

        total_rows = int(math.sqrt(spatial_part_size))
        total_cols = int(math.sqrt(spatial_part_size))

        # current position of rank in matrix of math.sqrt(spatial_part_size) * math.sqrt(spatial_part_size)
        row = int(local_rank / total_cols)
        col = int(local_rank % total_cols)

        start_left = col * image_width_local
        end_right = (col + 1) * image_width_local

        start_top = row * image_height_local
        end_bottom = (row + 1) * image_height_local

        return inputs[:, :, start_top:end_bottom, start_left:end_right]

    elif slice_method == "vertical":
        image_height_local = int(image_size / spatial_part_size)
        image_width_local = int(image_size / spatial_part_size)

        start_left = local_rank * image_width_local
        end_right = (local_rank + 1) * image_width_local

        if local_rank == spatial_part_size - 1:
            # In case of GPU count, partition size will be uneven and last
            # rank will receive remaining image
            return inputs[:, :, :, start_left:]
        else:
            return inputs[:, :, :, start_left:end_right]

    elif slice_method == "horizontal":
        image_height_local = int(image_size / spatial_part_size)
        image_width_local = int(image_size / spatial_part_size)

        start_top = local_rank * image_height_local
        end_bottom = (local_rank + 1) * image_height_local

        if local_rank == spatial_part_size - 1:
            # In case of odd GPU count, partition size will be uneven and last
            # rank will receive remaining image
            return inputs[:, :, start_top:, :]
        else:
            return inputs[:, :, start_top:end_bottom, :]


class train_model_spatial(train_model):
    def __init__(
        self,
        model_gen,
        local_rank,
        batch_size,
        epochs,
        spatial_size=1,
        num_spatial_parts=4,
        criterion=None,
        optimizer=None,
        parts=1,
        ASYNC=True,
        GEMS_INVERSE=False,
        slice_method="square",
        LOCAL_DP_LP=1,
        mpi_comm=None,
    ):
        self.slice_method = slice_method
        # model_gen.mp_size = (spatial_size * num_spatial_parts) - spatial_size + model_gen.mp_size

        # Data parallelism in LP with spatial parallelism (LBANN like)
        # SP 	LP(DP)
        # 0		4
        # 1		5
        # 2		6
        # 3 	7
        # LOCAL_DP_LP gives the number of nodes in DP for local
        self.LOCAL_DP_LP = LOCAL_DP_LP
        if LOCAL_DP_LP > 1:
            self.ENABLE_LOCAL_DP_LP = True
            self.LP_DP_Group_List = mpi_comm.LP_SP_Groups
            self.SP_LP_group = mpi_comm.SP_LP_group
        else:
            self.ENABLE_LOCAL_DP_LP = False

        self.spatial_size = spatial_size
        if isinstance(num_spatial_parts, list):
            self.local_rank = local_rank
            self.num_spatial_parts_list = num_spatial_parts
            # Number processes (one mp cluster) involve in spatial parallelism
            self.total_spatial_processes = sum(num_spatial_parts)

            if self.local_rank < self.total_spatial_processes:
                (
                    self.spatial_local_rank,
                    self.num_spatial_parts,
                ) = self.get_local_spatial_rank(num_spatial_parts, local_rank)
            else:
                self.num_spatial_parts = num_spatial_parts[-1]

            assert spatial_size == len(
                num_spatial_parts
            ), "Spatial size is not equal to lenght of num_spatial_parts"
        else:
            self.local_rank = local_rank
            self.spatial_local_rank = local_rank
            self.num_spatial_parts = num_spatial_parts
            self.total_spatial_processes = num_spatial_parts
            assert spatial_size == 1, "Spatial size is not 1"

        self.split_size = model_gen.split_size

        if local_rank < self.total_spatial_processes:
            self.split_rank = self.get_split_rank(num_spatial_parts, local_rank)
        else:
            # self.split_rank = local_rank - self.total_spatial_processes + spatial_size
            self.split_rank = (
                math.floor(
                    (self.local_rank - self.total_spatial_processes) / self.LOCAL_DP_LP
                )
                + spatial_size
            )

        self.mp_size = mpi_comm.mp_size
        super(train_model_spatial, self).__init__(
            model_gen,
            local_rank,
            batch_size,
            epochs,
            criterion=criterion,
            optimizer=optimizer,
            parts=parts,
            ASYNC=ASYNC,
            GEMS_INVERSE=GEMS_INVERSE,
        )

        # Call this function before initializing the recv buffers
        if self.ENABLE_LOCAL_DP_LP:
            self.update_shape_list_Local_DP_LP()
            # To initilize the buffers again with updated shapes
            super(train_model_spatial, self).initialize_recv_buffers()

        # special case when local rank is joining spatial inputs
        if self.split_rank == self.spatial_size:
            self.joint_inputs = []
            shapes = self.shape_list[self.split_rank - 1]

            self.initialize_recv_buffers_joint()

        if self.split_rank != 0 and self.local_rank < self.total_spatial_processes:
            recv_ranks_num = int(
                self.num_spatial_parts_list[self.split_rank - 1]
                / self.num_spatial_parts_list[self.split_rank]
            )
            if recv_ranks_num > 1:
                self.initialize_recv_buffers_spatial_intermediate()

    def update_shape_list_Local_DP_LP(self):
        # it does not update the shape of splits before the joint rank (MP rank that
        # takes input from the last spatial parallelism)

        # it updates the batch size accroding to the LOCAL_DP_LP

        start = self.spatial_size

        if self.split_rank == self.spatial_size:
            start = self.spatial_size - 1

        for i in range(start, self.split_size):
            if isinstance(self.shape_list[i], list):
                for j in range(len(self.shape_list[i])):
                    temp_tuple = list(self.shape_list[i][j])

                    temp_tuple[0] = int(temp_tuple[0] / self.LOCAL_DP_LP)
                    self.shape_list[i][j] = tuple(temp_tuple)
            else:
                temp_tuple = list(self.shape_list[i])

                temp_tuple[0] = int(temp_tuple[0] / self.LOCAL_DP_LP)
                self.shape_list[i] = tuple(temp_tuple)

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

    def get_local_spatial_rank(self, num_spatial_parts_list, local_rank):
        temp_sum = 0
        for parts in num_spatial_parts_list:
            if local_rank < temp_sum + parts:
                spatial_local_rank = local_rank - temp_sum
                num_spatial_parts = parts
                break

            else:
                temp_sum += parts

        return spatial_local_rank, num_spatial_parts

    def initialize_recv_buffers_spatial_intermediate(self):
        self.input_x_list = []

        num_recvs = int(
            self.num_spatial_parts_list[self.split_rank - 1]
            / self.num_spatial_parts_list[self.split_rank]
        )

        # intializing recv buffer for the input
        # For parts we need different buffers as in backward pass we using grad variable to
        # send partial errors to previous partition
        for part in range(self.parts):
            input_x_list_ranks = []

            for rank in range(num_recvs):
                input_x = []
                if self.split_rank != 0:
                    # multiple inputs
                    if isinstance(self.shape_list[self.split_rank - 1], list):
                        for i in range(len(self.shape_list[self.split_rank - 1])):
                            one_input = torch.zeros(
                                self.shape_list[self.split_rank - 1][i],
                                requires_grad=True,
                                device="cuda",
                            )
                            input_x.append(one_input)
                        input_x = tuple(input_x)
                    else:
                        input_x = torch.zeros(
                            self.shape_list[self.split_rank - 1],
                            requires_grad=True,
                            device="cuda",
                        )
                input_x_list_ranks.append(input_x)

            self.input_x_list.append(input_x_list_ranks)

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

    def initialize_recv_buffers_joint(self):
        self.input_x_list = []
        ranks = [
            self.local_rank - 1 - i for i in range(self.num_spatial_parts - 1, -1, -1)
        ]

        # intializing recv buffer for the input
        # For parts we need different buffers as in backward pass we using grad variable to
        # send partial errors to previous partition
        for part in range(self.parts):
            input_x_list_ranks = []

            for rank in ranks:
                input_x = []
                if self.split_rank != 0:
                    # multiple inputs
                    if isinstance(self.shape_list[self.split_rank - 1], list):
                        for i in range(len(self.shape_list[self.split_rank - 1])):
                            one_input = torch.zeros(
                                self.shape_list[self.split_rank - 1][i],
                                requires_grad=True,
                                device="cuda",
                            )
                            input_x.append(one_input)
                        input_x = tuple(input_x)
                    else:
                        input_x = torch.zeros(
                            self.shape_list[self.split_rank - 1],
                            requires_grad=True,
                            device="cuda",
                        )
                input_x_list_ranks.append(input_x)

            self.input_x_list.append(input_x_list_ranks)

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

    def initialize_send_recv_ranks(self):
        self.increase_rank = 1
        self.decrease_rank = 1
        self.SKEWED_RECV_SPATIAL = False

        # These ranks does not matter except for first spatial convolution
        # Rank joining spatial and layer prarallelism does not use it

        if (
            self.split_rank == 0 and self.spatial_size == 1
        ) or self.split_rank == self.spatial_size - 1:
            if self.local_rank < self.total_spatial_processes:
                self.increase_rank = self.num_spatial_parts

                if self.increase_rank + self.local_rank >= self.total_spatial_processes:
                    self.increase_rank = self.total_spatial_processes - self.local_rank

            self.decrease_rank = self.increase_rank

            # Case
            # num_spatial_parts 4,2 split_size 4
            # last spatial rank will recieve multiple inputs from multiple ranks and needs to send to next MP process
            if self.spatial_size > 1 and self.split_rank != 0:
                recv_ranks_num = int(
                    self.num_spatial_parts_list[self.split_rank - 1]
                    / self.num_spatial_parts_list[self.split_rank]
                )
                if recv_ranks_num > 1:
                    self.SKEWED_RECV_SPATIAL = True

                else:
                    self.decrease_rank = self.num_spatial_parts_list[self.split_rank]

        elif self.local_rank < self.total_spatial_processes:
            send_ranks_num = int(
                self.num_spatial_parts_list[self.split_rank]
                / self.num_spatial_parts_list[self.split_rank + 1]
            )

            if send_ranks_num == 1:
                self.increase_rank = self.num_spatial_parts
            else:
                recv_rank = sum(self.num_spatial_parts_list[:1]) + math.floor(
                    self.spatial_local_rank / send_ranks_num
                )
                self.increase_rank = recv_rank - self.local_rank

            # Decrease rank does not matter as this rank will not send/recv anything backwards
            self.decrease_rank = self.increase_rank

            if self.split_rank == 0:
                self.SKEWED_RECV_SPATIAL = False
            else:
                recv_ranks_num = int(
                    self.num_spatial_parts_list[self.split_rank - 1]
                    / self.num_spatial_parts_list[self.split_rank]
                )
                if recv_ranks_num > 1:
                    self.SKEWED_RECV_SPATIAL = True
                else:
                    self.SKEWED_RECV_SPATIAL = False

        elif self.LOCAL_DP_LP > 1 and self.local_rank >= self.total_spatial_processes:
            # For Local_DP_LP only this case matters
            self.increase_rank = self.LOCAL_DP_LP
            self.decrease_rank = self.LOCAL_DP_LP

        if self.GEMS_INVERSE == False:
            self.to_send_forward = self.local_rank + self.increase_rank
            self.to_recv_forward = self.local_rank - self.decrease_rank
            self.to_send_backward = self.local_rank - self.decrease_rank
            self.to_recv_backward = self.local_rank + self.increase_rank
        else:
            self.to_send_forward = (
                self.mp_size - 1 - self.local_rank - self.increase_rank
            )
            self.to_recv_forward = (
                self.mp_size - 1 - self.local_rank + self.decrease_rank
            )
            self.to_send_backward = (
                self.mp_size - 1 - self.local_rank + self.decrease_rank
            )
            self.to_recv_backward = (
                self.mp_size - 1 - self.local_rank - self.increase_rank
            )

    def recv_input_spatial_async(self, part_number):
        num_recvs = int(
            self.num_spatial_parts_list[self.split_rank - 1]
            / self.num_spatial_parts_list[self.split_rank]
        )

        first_local_rank_last_spatial = sum(
            self.num_spatial_parts_list[: self.split_rank - 1]
        )

        # position in current spatial rank
        mypos = self.spatial_local_rank

        # Generate ranks
        first_rank_recv = mypos * num_recvs + first_local_rank_last_spatial
        recv_ranks = [first_rank_recv + i for i in range(num_recvs)]

        if self.GEMS_INVERSE:
            temp_recv_ranks = []
            for rank in recv_ranks:
                temp_recv_ranks.append(self.mp_size - 1 - rank)
            recv_ranks = temp_recv_ranks

        reqs = []
        # Recv input
        for rank in range(len(recv_ranks)):
            tag_forward = 0
            if self.MULTIPLE_INPUT == True:
                for i in range(len(self.shape_list[self.split_rank - 1])):
                    req_temp = dist.irecv(
                        tensor=self.input_x_list[part_number][rank][i],
                        src=recv_ranks[rank],
                        tag=tag_forward,
                    )
                    reqs.append(req_temp)
                    tag_forward += 1
            else:
                req_temp = dist.irecv(
                    tensor=self.input_x_list[part_number][rank],
                    src=recv_ranks[rank],
                    tag=tag_forward,
                )
                reqs.append(req_temp)

        for req in reqs:
            req.wait()

    def receive_input_async_joint(self, part_number, ranks):
        ranks = [
            self.local_rank - 1 - i for i in range(self.num_spatial_parts - 1, -1, -1)
        ]

        if self.GEMS_INVERSE:
            for i in range(len(ranks)):
                ranks[i] = self.mp_size - 1 - ranks[i]

        reqs = []

        for rank in range(len(ranks)):
            tag_forward = 0
            if self.MULTIPLE_INPUT == True:
                for i in range(len(self.shape_list[self.split_rank - 1])):
                    req_temp = dist.irecv(
                        tensor=self.input_x_list[part_number][rank][i],
                        src=ranks[rank],
                        tag=tag_forward,
                    )
                    reqs.append(req_temp)
                    tag_forward += 1
            else:
                req_temp = dist.irecv(
                    tensor=self.input_x_list[part_number][rank],
                    src=ranks[rank],
                    tag=tag_forward,
                )
                reqs.append(req_temp)

        for req in reqs:
            req.wait()

    def send_grad_async_joint(self, input_x_list):
        # No need for writing modification for slide_methods as input_list are used and there is no slicing of image

        ranks = [
            self.local_rank - 1 - i for i in range(self.num_spatial_parts - 1, -1, -1)
        ]

        if self.GEMS_INVERSE:
            for i in range(len(ranks)):
                ranks[i] = self.mp_size - 1 - ranks[i]

        reqs = []

        for partition in range(self.num_spatial_parts):
            tag_forward = 0

            if self.MULTIPLE_INPUT:
                for i in range(len(self.shape_list[self.split_rank - 1])):
                    shape = self.shape_list[self.split_rank - 1][i]
                    to_send = input_x_list[partition][i].grad.data.clone().detach()
                    req = dist.isend(
                        tensor=to_send, dst=ranks[partition], tag=tag_forward
                    )
                    tag_forward += 1
                    reqs.append(req)

            else:
                shape = self.shape_list[self.split_rank - 1]
                to_send = input_x_list[partition].grad.data.clone().detach()
                torch.cuda.synchronize()
                req = dist.isend(tensor=to_send, dst=ranks[partition], tag=tag_forward)
                reqs.append(req)

        for req in reqs:
            req.wait()

    def send_grad_async_spatial(self, input_x_list):
        # No need for writing modification for slide_methods as input_list are used and there is no slicing of image

        num_recvs = int(
            self.num_spatial_parts_list[self.split_rank - 1]
            / self.num_spatial_parts_list[self.split_rank]
        )

        first_local_rank_last_spatial = sum(
            self.num_spatial_parts_list[: self.split_rank - 1]
        )

        # position in current spatial rank
        mypos = self.spatial_local_rank

        # Generate ranks
        first_rank_recv = mypos * num_recvs + first_local_rank_last_spatial
        recv_ranks = [first_rank_recv + i for i in range(num_recvs)]

        if self.GEMS_INVERSE:
            for i in range(len(recv_ranks)):
                recv_ranks[i] = self.mp_size - 1 - recv_ranks[i]

        reqs = []

        for partition in range(num_recvs):
            tag_forward = 0

            if self.MULTIPLE_INPUT:
                for i in range(len(self.shape_list[self.split_rank - 1])):
                    shape = self.shape_list[self.split_rank - 1][i]
                    to_send = input_x_list[partition][i].grad.data.clone().detach()
                    req = dist.isend(
                        tensor=to_send, dst=recv_ranks[partition], tag=tag_forward
                    )
                    tag_forward += 1
                    reqs.append(req)

            else:
                shape = self.shape_list[self.split_rank - 1]
                to_send = input_x_list[partition].grad.data.clone().detach()
                torch.cuda.synchronize()
                req = dist.isend(
                    tensor=to_send, dst=recv_ranks[partition], tag=tag_forward
                )
                reqs.append(req)

        for req in reqs:
            req.wait()

    def send_input_spatial_MP_joint_LP_DP(self, y):
        # This function is called by last spatial split to send its output
        # to next split (MP/LP) with Data Parallelism
        parts_size = int(self.batch_size / self.parts)
        assert (
            parts_size % self.LOCAL_DP_LP == 0
        ), "Part size (batch size/parts) should be divided by Local DP LP "
        per_rank = int(parts_size / self.LOCAL_DP_LP)

        # initializing temporary buffers

        if self.MULTIPLE_OUTPUT:
            for one_output in y:
                scatter_list = []
                # scatter_list.append(tempB)
                for i in range(self.LOCAL_DP_LP):
                    scatter_list.append(one_output[i * per_rank : (i + 1) * per_rank])
                    shapes = tuple(one_output[i * per_rank : (i + 1) * per_rank].shape)
                    dtype = one_output[i * per_rank : (i + 1) * per_rank].dtype
                tempA = torch.zeros(shapes, dtype=dtype, device="cuda")
                tempB = torch.zeros(shapes, dtype=dtype, device="cuda")

                src = self.local_rank
                if self.GEMS_INVERSE:
                    src = self.mp_size - 1 - src

                scatter_list = [tempB] + scatter_list
                dist.scatter(
                    tensor=tempA,
                    scatter_list=scatter_list,
                    src=self.local_rank,
                    group=self.SP_LP_group,
                )

        else:
            scatter_list = []
            scatter_list.append(tempB)
            for i in range(self.LOCAL_DP_LP):
                scatter_list.append(y[i * per_rank : (i + 1) * per_rank])

            src = self.local_rank
            if self.GEMS_INVERSE:
                src = self.mp_size - 1 - src
            dist.scatter(
                tensor=tempA, scatter_list=scatter_list, src=src, group=self.SP_LP_group
            )

    def recv_input_MP_joint_LP_DP(self, part_number):
        # This function is called by first MP split to recv its input
        # from previous split that is spatial (MP/LP) with Data Parallelism
        #########################################################
        ranks = [
            self.local_rank - 1 - i for i in range(self.num_spatial_parts - 1, -1, -1)
        ]
        reqs = []

        if self.spatial_size > 1:
            prev_num_spatial_parts = self.num_spatial_parts_list[-1]

        else:
            prev_num_spatial_parts = self.num_spatial_parts

        start_rank_last_spatial = self.total_spatial_processes - prev_num_spatial_parts

        for rank in range(prev_num_spatial_parts):
            tag_forward = 0
            if self.MULTIPLE_INPUT == True:
                for i in range(len(self.shape_list[self.split_rank - 1])):
                    src = start_rank_last_spatial + rank
                    if self.GEMS_INVERSE:
                        src = self.mp_size - 1 - src
                    dist.scatter(
                        tensor=self.input_x_list[part_number][rank][i],
                        scatter_list=None,
                        src=src,
                        group=self.LP_DP_Group_List[rank],
                    )
                    self.input_x_list[part_number][rank][i].requires_grad = True

                    # reqs.append(req_temp)

            else:
                src = start_rank_last_spatial + rank
                if self.GEMS_INVERSE:
                    src = self.mp_size - 1 - src
                dist.scatter(
                    tensor=self.input_x_list[part_number][rank],
                    scatter_list=None,
                    src=src,
                    group=self.LP_DP_Group_List[rank],
                )
                # reqs.append(req_temp)

        for req in reqs:
            req.wait()

    def send_grad_MP_joint_LP_DP(self, input_x_list):
        # This function is called by first MP split to recv its input
        # from previous split that is spatial (MP/LP) with Data Parallelism
        #########################################################

        # No need for writing modification for slide_methods as input_list are used and there is no slicing of image

        if self.spatial_size > 1:
            prev_num_spatial_parts = self.num_spatial_parts_list[-1]

        else:
            prev_num_spatial_parts = self.num_spatial_parts

        start_rank_last_spatial = self.total_spatial_processes - prev_num_spatial_parts

        reqs = []

        for rank in range(prev_num_spatial_parts):
            tag_forward = 0

            if self.MULTIPLE_INPUT:
                for i in range(len(self.shape_list[self.split_rank - 1])):
                    shape = self.shape_list[self.split_rank - 1][i]
                    # to_send =input_x_list[:,:,partition_i*shape[2]:(partition_i+1)*shape[2],  partition_j*shape[3]:(partition_j+1)*shape[3]].grad.data.clone().detach()
                    to_send = input_x_list[rank][i].grad.data.clone().detach()
                    torch.cuda.synchronize()
                    dst = start_rank_last_spatial + rank
                    if self.GEMS_INVERSE:
                        dst = self.mp_size - 1 - dst
                    dist.gather(
                        tensor=to_send,
                        gather_list=None,
                        dst=dst,
                        group=self.LP_DP_Group_List[rank],
                    )

                    # reqs.append(req)

            else:
                shape = self.shape_list[self.split_rank - 1]
                to_send = input_x_list[rank].grad.data.clone().detach()
                torch.cuda.synchronize()

                dst = start_rank_last_spatial + rank
                if self.GEMS_INVERSE:
                    dst = self.mp_size - 1 - dst
                dist.gather(
                    tensor=to_send,
                    gather_list=None,
                    dst=dst,
                    group=self.LP_DP_Group_List[rank],
                )

                # reqs.append(req)

        for req in reqs:
            req.wait()

        ######################################################

    def recv_grad_spatial_MP_joint_LP_DP(self, y):
        # This function is called by first MP split to recv its input
        # from previous split that is spatial (MP/LP) with Data Parallelism
        #########################################################

        parts_size = int(self.batch_size / self.parts)
        assert (
            parts_size % self.LOCAL_DP_LP == 0
        ), "Part size (batch size/parts) should be divided by Local DP LP "
        per_rank = int(parts_size / self.LOCAL_DP_LP)

        # initializing temporary buffers

        if self.MULTIPLE_OUTPUT:
            for one_output in self.grad_overhead:
                gather_list = []
                # scatter_list.append(tempB)
                for i in range(self.LOCAL_DP_LP):
                    shapes = tuple(one_output[i * per_rank : (i + 1) * per_rank].shape)
                    dtype = one_output[i * per_rank : (i + 1) * per_rank].dtype

                    gather_list.append(torch.zeros(shapes, dtype=dtype, device="cuda"))

                tempA = torch.zeros(shapes, dtype=dtype, device="cuda")
                tempB = torch.zeros(shapes, dtype=dtype, device="cuda")

                gather_list = [tempB] + gather_list

                dst = self.local_rank
                if self.GEMS_INVERSE:
                    dst = self.mp_size - 1 - dst

                dist.gather(
                    tensor=tempA,
                    gather_list=gather_list,
                    dst=dst,
                    group=self.SP_LP_group,
                )

                for i in range(self.LOCAL_DP_LP):
                    one_output[i * per_rank : (i + 1) * per_rank] = gather_list[i]

        else:
            gather_list = []
            gather_list.append(tempB)
            for i in range(self.LOCAL_DP_LP):
                gather_list.append(y[i * per_rank : (i + 1) * per_rank])
                shapes = tuple(y[i * per_rank : (i + 1) * per_rank].shape)
                dtype = y[i * per_rank : (i + 1) * per_rank].dtype
            tempA = torch.zeros(shapes, dtype=dtype, device="cuda")
            tempB = torch.zeros(shapes, dtype=dtype, device="cuda")
            gather_list = [tempB] + gather_list

            dst = self.local_rank
            if self.GEMS_INVERSE:
                dst = self.mp_size - 1 - dst

            dist.gather(
                tensor=tempA, gather_list=gather_list, dst=dst, group=self.SP_LP_group
            )
            for i in range(self.LOCAL_DP_LP):
                self.grad_overhead[i * per_rank : (i + 1) * per_rank] = gather_list[i]

        ##########################################################

    def recv_inputs_joint(self, part_number):
        ranks = [
            self.local_rank - 1 - i for i in range(self.num_spatial_parts - 1, -1, -1)
        ]

        self.receive_input_async_joint(part_number, ranks)

    def recv_input_spatial(self, part_number):
        self.recv_input_spatial_async(part_number)

    def merge_inputs_joint(self, part_number):
        # 0 | 1
        # -------
        # 2 | 3
        # partition_i : row
        # partition_j : col
        # spatial_rank =  partition_i * sqrt(num_spatial_parts) + partition_j

        if self.MULTIPLE_INPUT:
            for i in range(len(self.shape_list[self.split_rank - 1])):
                shape = self.shape_list[self.split_rank - 1][i]
                for partition_i in range(int(math.sqrt(self.num_spatial_parts))):
                    for partition_j in range(int(math.sqrt(self.num_spatial_parts))):
                        self.joint_inputs[part_number][i][
                            :,
                            :,
                            partition_i * shape[2] : (partition_i + 1) * shape[2],
                            partition_j * shape[3] : (partition_j + 1) * shape[3],
                        ].set_(
                            self.input_x_list[part_number][
                                partition_i * int(math.sqrt(self.num_spatial_parts))
                                + partition_j
                            ][i]
                        )

            return tuple(self.joint_inputs[part_number])
        else:
            shape = self.shape_list[self.split_rank - 1]
            for partition_i in range(int(math.sqrt(self.num_spatial_parts))):
                for partition_j in range(int(math.sqrt(self.num_spatial_parts))):
                    self.joint_inputs[part_number][
                        :,
                        :,
                        partition_i * shape[2] : (partition_i + 1) * shape[2],
                        partition_j * shape[3] : (partition_j + 1) * shape[3],
                    ].set_(
                        self.input_x_list[part_number][
                            partition_i * int(math.sqrt(self.num_spatial_parts))
                            + partition_j
                        ]
                    )
            return self.joint_inputs[part_number]

    def merge_inputs_joint_cat(self, part_number):
        if self.slice_method == "square":
            # 0 | 1
            # -------
            # 2 | 3
            # partition_i : row
            # partition_j : col
            # spatial_rank =  partition_i * sqrt(num_spatial_parts) + partition_j

            if self.MULTIPLE_INPUT:
                outs = []
                for i in range(len(self.shape_list[self.split_rank - 1])):
                    shape = self.shape_list[self.split_rank - 1][i]

                    temp_row = []
                    for partition_i in range(int(math.sqrt(self.num_spatial_parts))):
                        temp_col = []
                        for partition_j in range(
                            int(math.sqrt(self.num_spatial_parts))
                        ):
                            temp_col.append(
                                self.input_x_list[part_number][
                                    partition_i * int(math.sqrt(self.num_spatial_parts))
                                    + partition_j
                                ][i]
                            )

                        temp_row.append(torch.cat(temp_col, axis=-1))

                    out = torch.cat(temp_row, axis=-2)
                    outs.append(out)

                return tuple(outs)
            else:
                shape = self.shape_list[self.split_rank - 1]
                temp_row = []
                for partition_i in range(int(math.sqrt(self.num_spatial_parts))):
                    temp_col = []
                    for partition_j in range(int(math.sqrt(self.num_spatial_parts))):
                        temp_col.append(
                            self.input_x_list[part_number][
                                partition_i * int(math.sqrt(self.num_spatial_parts))
                                + partition_j
                            ]
                        )

                    temp_row.append(torch.cat(temp_col, axis=-1))
                out = torch.cat(temp_row, axis=-2)

                return out

        elif self.slice_method == "vertical":
            # 0 | 1 | 2 | 3
            if self.MULTIPLE_INPUT:
                outs = []
                for i in range(len(self.shape_list[self.split_rank - 1])):
                    shape = self.shape_list[self.split_rank - 1][i]

                    temp_col = []
                    for partition_j in range(self.num_spatial_parts):
                        temp_col.append(self.input_x_list[part_number][partition_j][i])

                    out = torch.cat(temp_col, axis=-1)
                    outs.append(out)

                return tuple(outs)
            else:
                shape = self.shape_list[self.split_rank - 1]
                temp_col = []
                for partition_j in range(self.num_spatial_parts):
                    temp_col.append(self.input_x_list[part_number][partition_j])

                out = torch.cat(temp_col, axis=-1)

                return out

        elif self.slice_method == "horizontal":
            #  0
            # ---
            #  1
            # ---
            #  2
            # ---
            #  3
            if self.MULTIPLE_INPUT:
                outs = []
                for i in range(len(self.shape_list[self.split_rank - 1])):
                    shape = self.shape_list[self.split_rank - 1][i]

                    temp_row = []
                    for partition_i in range(self.num_spatial_parts):
                        temp_row.append(self.input_x_list[part_number][partition_i][i])

                    out = torch.cat(temp_row, axis=-2)
                    outs.append(out)

                return tuple(outs)
            else:
                shape = self.shape_list[self.split_rank - 1]
                temp_row = []
                for partition_i in range(self.num_spatial_parts):
                    temp_row.append(self.input_x_list[part_number][partition_i])

                out = torch.cat(temp_row, axis=-2)

                return out

    def merge_inputs_intermediate_spatial(self, part_number):
        num_recvs = int(
            self.num_spatial_parts_list[self.split_rank - 1]
            / self.num_spatial_parts_list[self.split_rank]
        )
        if num_recvs == 1:
            if self.MULTIPLE_INPUT:
                input_x = tuple(self.input_x_list[part_number])
            else:
                input_x = self.input_x_list[part_number]

        assert (
            self.slice_method != "square"
        ), "Currently different num_spatial_parts are not supported for square slice method"

        if self.slice_method == "vertical":
            # 0 | 1 | 2 | 3
            if self.MULTIPLE_INPUT:
                outs = []
                for i in range(len(self.shape_list[self.split_rank - 1])):
                    temp_col = []

                    for partition_j in range(num_recvs):
                        temp_col.append(self.input_x_list[part_number][partition_j][i])

                    out = torch.cat(temp_col, axis=-1)
                    outs.append(out)

                return tuple(outs)
            else:
                temp_col = []
                for partition_j in range(num_recvs):
                    temp_col.append(self.input_x_list[part_number][partition_j])

                out = torch.cat(temp_col, axis=-1)

                return out

        elif self.slice_method == "horizontal":
            #  0
            # ---
            #  1
            # ---
            #  2
            # ---
            #  3
            if self.MULTIPLE_INPUT:
                outs = []
                for i in range(len(self.shape_list[self.split_rank - 1])):
                    temp_row = []
                    for partition_i in range(num_recvs):
                        temp_row.append(self.input_x_list[part_number][partition_i][i])

                    out = torch.cat(temp_row, axis=-2)
                    outs.append(out)

                return tuple(outs)
            else:
                temp_row = []
                for partition_i in range(num_recvs):
                    temp_row.append(self.input_x_list[part_number][partition_i])

                out = torch.cat(temp_row, axis=-2)

                return out

    def forward_pass(self, data_x, data_y, part_number=0):
        # data_x: input data
        # data_y: labels
        # part_number: part number between 0 and self.parts-1 used to find right input recv buffer

        # Receive inputs if local is not 0
        if self.split_rank == 0:
            input_x = data_x
        else:
            if self.ENABLE_ASYNC == True:
                if self.split_rank == self.spatial_size:
                    if self.ENABLE_LOCAL_DP_LP:
                        self.recv_input_MP_joint_LP_DP(part_number)
                    else:
                        self.recv_inputs_joint(part_number)
                elif self.SKEWED_RECV_SPATIAL:
                    self.recv_input_spatial(part_number)
                else:
                    self.receive_input_async(part_number)
            else:
                if self.local_rank == self.total_spatial_processes:
                    self.recv_inputs_joint(part_number)
                elif self.SKEWED_RECV_SPATIAL:
                    self.recv_input_spatial(part_number)
                else:
                    self.receive_input_sync(part_number)

            # join spatial inputs
            if self.split_rank == self.spatial_size:
                input_x = self.merge_inputs_joint_cat(part_number)
            elif self.split_rank != 0 and self.SKEWED_RECV_SPATIAL:
                input_x = self.merge_inputs_intermediate_spatial(part_number)
            else:
                if self.MULTIPLE_INPUT:
                    input_x = tuple(self.input_x_list[part_number])
                else:
                    input_x = self.input_x_list[part_number]

        # Apply forward pass

        torch.cuda.synchronize()

        # For pipeline parallelism support
        if (
            isinstance(
                self.models, torch.nn.parallel.distributed.DistributedDataParallel
            )
            and part_number != self.parts - 1
        ):
            with self.models.no_sync():
                y = self.models(input_x)
        else:
            y = self.models(input_x)

        torch.cuda.synchronize()

        if self.split_rank != self.split_size - 1:
            if self.ENABLE_ASYNC == True:
                if self.split_rank == self.spatial_size - 1 and self.ENABLE_LOCAL_DP_LP:
                    self.send_input_spatial_MP_joint_LP_DP(y)
                else:
                    self.send_input_async(y)
            else:
                if self.split_rank == self.spatial_size - 1 and self.ENABLE_LOCAL_DP_LP:
                    self.send_input_spatial_MP_joint_LP_DP(y)
                else:
                    self.send_input_sync(y)

        else:
            pos = self.local_rank - (self.mp_size - self.LOCAL_DP_LP)
            parts_size = int(self.batch_size / self.parts)
            per_rank = int(parts_size / self.LOCAL_DP_LP)
            data_y = data_y[per_rank * pos : per_rank * (pos + 1)]

            # For pipeline parallelism support
            if (
                isinstance(
                    self.models, torch.nn.parallel.distributed.DistributedDataParallel
                )
                and part_number != self.parts - 1
            ):
                with self.models.no_sync():
                    loss = self.criterion(y, data_y)
            else:
                loss = self.criterion(y, data_y)

        if self.split_rank == self.split_size - 1:
            corrects = (data_y.eq(torch.argmax(y, dim=-1).long())).sum().float()
            return loss, corrects / self.batch_size
        else:
            return y, None

    def backward_pass(self, y, part_number=0):
        if self.split_rank != self.split_size - 1:
            if self.ENABLE_ASYNC:
                if self.split_rank == self.spatial_size - 1 and self.ENABLE_LOCAL_DP_LP:
                    self.recv_grad_spatial_MP_joint_LP_DP(y)
                else:
                    self.receive_grad_async()
            else:
                if self.split_rank == self.spatial_size - 1 and self.ENABLE_LOCAL_DP_LP:
                    self.recv_grad_spatial_MP_joint_LP_DP(y)
                else:
                    self.receive_grad_sync()

        torch.cuda.synchronize()
        if (
            isinstance(
                self.models, torch.nn.parallel.distributed.DistributedDataParallel
            )
            and part_number != self.parts - 1
        ):
            with self.models.no_sync():
                if self.split_rank == self.split_size - 1:
                    y.backward()
                else:
                    torch.autograd.backward(y, self.grad_overhead)
        else:
            if self.split_rank == self.split_size - 1:
                y.backward()
            else:
                torch.autograd.backward(y, self.grad_overhead)
        torch.cuda.synchronize()

        if self.split_rank != 0:
            if self.split_rank == self.spatial_size:
                if self.ENABLE_LOCAL_DP_LP:
                    self.send_grad_MP_joint_LP_DP(self.input_x_list[part_number])
                else:
                    self.send_grad_async_joint(self.input_x_list[part_number])
            elif self.SKEWED_RECV_SPATIAL:
                self.send_grad_async_spatial(self.input_x_list[part_number])
            else:
                if self.ENABLE_ASYNC:
                    self.send_grad_async(self.input_x_list[part_number])
                else:
                    self.send_input_sync(self.input_x_list[part_number])

        if self.split_rank != 0:
            if self.split_rank == self.spatial_size:
                if self.MULTIPLE_INPUT:
                    for i in range(self.num_spatial_parts):
                        self.input_x_list[part_number][i] = list(
                            self.input_x_list[part_number][i]
                        )
                        for j in range(len(self.shape_list[self.split_rank - 1])):
                            self.input_x_list[part_number][i][j] = (
                                self.input_x_list[part_number][i][j]
                                .detach()
                                .requires_grad_()
                            )
                        self.input_x_list[part_number][i] = tuple(
                            self.input_x_list[part_number][i]
                        )

                else:
                    for i in range(self.num_spatial_parts):
                        self.input_x_list[part_number][i] = (
                            self.input_x_list[part_number][i].detach().requires_grad_()
                        )

            elif self.SKEWED_RECV_SPATIAL:
                recv_ranks_num = int(
                    self.num_spatial_parts_list[self.split_rank - 1]
                    / self.num_spatial_parts_list[self.split_rank]
                )
                if self.MULTIPLE_INPUT:
                    for i in range(recv_ranks_num):
                        self.input_x_list[part_number][i] = list(
                            self.input_x_list[part_number][i]
                        )
                        for j in range(len(self.shape_list[self.split_rank - 1])):
                            self.input_x_list[part_number][i][j] = (
                                self.input_x_list[part_number][i][j]
                                .detach()
                                .requires_grad_()
                            )
                        self.input_x_list[part_number][i] = tuple(
                            self.input_x_list[part_number][i]
                        )

                else:
                    for i in range(recv_ranks_num):
                        self.input_x_list[part_number][i] = (
                            self.input_x_list[part_number][i].detach().requires_grad_()
                        )

            else:
                if self.MULTIPLE_INPUT:
                    self.input_x_list[part_number] = list(
                        self.input_x_list[part_number]
                    )
                    for j in range(len(self.shape_list[self.split_rank - 1])):
                        self.input_x_list[part_number][j] = (
                            self.input_x_list[part_number][j].detach().requires_grad_()
                        )
                    self.input_x_list[part_number] = tuple(
                        self.input_x_list[part_number]
                    )
                else:
                    self.input_x_list[part_number] = (
                        self.input_x_list[part_number].detach().requires_grad_()
                    )
