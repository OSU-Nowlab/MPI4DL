import torch.nn as nn
import torch
import torch.distributed as dist
import math


class conv_spatial(nn.Conv2d):
    def __init__(
        self,
        local_rank,
        spatial_size,
        num_spatial_parts,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        halo_len=None,
        padding_mode="zeros",
        slice_method="square",
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(padding, int):
            padding = (padding, padding)

        # number of parts in one image

        if isinstance(num_spatial_parts, list):
            self.local_rank = local_rank
            (
                self.spatial_local_rank,
                self.num_spatial_parts,
            ) = self.get_local_spatial_rank(num_spatial_parts, local_rank)
        else:
            self.local_rank = local_rank
            self.spatial_local_rank = local_rank
            self.num_spatial_parts = num_spatial_parts
        # number of sequential spatial layers, Most of the times I expect it to be 1
        self.spatial_size = spatial_size

        # slice_method: "vertical" "horizontal" "square"
        self.slice_method = slice_method

        if halo_len is not None:
            assert (
                halo_len == 0
            ), "Error: Custom Halo Len is not supported (only halo_len=0 is supported)"

            self.halo_len_height = halo_len
            self.halo_len_width = halo_len

            # update this code
            if kernel_size[0] > 1:
                if self.spatial_local_rank == 0:
                    padding_left, padding_right, padding_top, padding_bottom = (
                        padding[1],
                        self.halo_len_width,
                        padding[0],
                        self.halo_len_height,
                    )
                elif self.spatial_local_rank == 1:
                    padding_left, padding_right, padding_top, padding_bottom = (
                        self.halo_len_width,
                        padding[1],
                        padding[0],
                        self.halo_len_height,
                    )
                elif self.spatial_local_rank == 2:
                    padding_left, padding_right, padding_top, padding_bottom = (
                        padding[1],
                        self.halo_len_width,
                        self.halo_len_height,
                        padding[0],
                    )
                elif self.spatial_local_rank == 3:
                    padding_left, padding_right, padding_top, padding_bottom = (
                        self.halo_len_width,
                        padding[1],
                        self.halo_len_height,
                        padding[0],
                    )
            else:
                padding_left, padding_right, padding_top, padding_bottom = (
                    padding[1],
                    padding[1],
                    padding[0],
                    padding[0],
                )

        else:
            # Halo len on height row/first dimension
            self.halo_len_height = int((kernel_size[0] - 1) / 2)
            # Halo len on width column/second dimension
            self.halo_len_width = int((kernel_size[1] - 1) / 2)

            assert (
                self.halo_len_height == padding[0] or self.halo_len_width == padding[1]
            ), "Spatial not supported yet for this configuration"
            # self.halo_len = padding
            padding_left, padding_right, padding_top, padding_bottom = (
                self.halo_len_width,
                self.halo_len_width,
                self.halo_len_height,
                self.halo_len_height,
            )

        super(conv_spatial, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode="zeros",
        )

        self.pads = nn.ZeroPad2d(
            (padding_left, padding_right, padding_top, padding_bottom)
        )

        if self.halo_len_height > 0 or self.halo_len_width > 0:
            self.get_neighbours()
            self.get_neighbours_rank()
            self.get_index_locations()

            print(self.neighbours)
        self.shapes_recv = None
        self.recv_tensors = []
        self.send_tensors = []

        self.set_tags()

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

    def set_tags(self):
        self.send_tag = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        self.recv_tag = [900, 800, 700, 600, 500, 400, 300, 200, 100]

        # self.send_tag = [0]*9
        # self.recv_tag = [0]*9

    def get_index_locations(self):
        locations_recv = []
        locations_recv.append(
            [[None, self.halo_len_height], [None, self.halo_len_width]]
        )  # 1

        locations_recv.append(
            [
                [None, self.halo_len_height],
                [
                    self.halo_len_width,
                    -self.halo_len_width if self.halo_len_width else None,
                ],
            ]
        )  # 2

        locations_recv.append(
            [[None, self.halo_len_height], [-self.halo_len_width, None]]
        )  # 3

        locations_recv.append(
            [
                [
                    self.halo_len_height,
                    -self.halo_len_height if self.halo_len_height else None,
                ],
                [None, self.halo_len_width],
            ]
        )  # 4

        locations_recv.append([[None, None], [None, None]])  # 5

        locations_recv.append(
            [
                [
                    self.halo_len_height,
                    -self.halo_len_height if self.halo_len_height else None,
                ],
                [-self.halo_len_width, None],
            ]
        )  # 6

        locations_recv.append(
            [[-self.halo_len_height, None], [None, self.halo_len_width]]
        )  # 7

        locations_recv.append(
            [
                [-self.halo_len_height, None],
                [
                    self.halo_len_width,
                    -self.halo_len_width if self.halo_len_width else None,
                ],
            ]
        )  # 8

        locations_recv.append(
            [[-self.halo_len_height, None], [-self.halo_len_width, None]]
        )  # 9

        self.locations_recv = locations_recv

        locations_send = []
        locations_send.append(
            [
                [self.halo_len_height, 2 * self.halo_len_height],
                [self.halo_len_width, 2 * self.halo_len_width],
            ]
        )  # 1

        locations_send.append(
            [
                [self.halo_len_height, 2 * self.halo_len_height],
                [
                    self.halo_len_width,
                    -self.halo_len_width if self.halo_len_width else None,
                ],
            ]
        )  # 2

        locations_send.append(
            [
                [self.halo_len_height, 2 * self.halo_len_height],
                [-2 * self.halo_len_width, -1 * self.halo_len_width],
            ]
        )  # 3

        locations_send.append(
            [
                [
                    self.halo_len_height,
                    -self.halo_len_height if self.halo_len_height else None,
                ],
                [self.halo_len_width, 2 * self.halo_len_width],
            ]
        )  # 4

        locations_send.append([[None, None], [None, None]])  # 5

        locations_send.append(
            [
                [
                    self.halo_len_height,
                    -self.halo_len_height if self.halo_len_height else None,
                ],
                [-2 * self.halo_len_width, -1 * self.halo_len_width],
            ]
        )  # 6

        locations_send.append(
            [
                [-2 * self.halo_len_height, -1 * self.halo_len_height],
                [self.halo_len_width, 2 * self.halo_len_width],
            ]
        )  # 7

        locations_send.append(
            [
                [-2 * self.halo_len_height, -1 * self.halo_len_height],
                [
                    self.halo_len_width,
                    -self.halo_len_width if self.halo_len_width else None,
                ],
            ]
        )  # 8

        locations_send.append(
            [
                [-2 * self.halo_len_height, -1 * self.halo_len_height],
                [-2 * self.halo_len_width, -1 * self.halo_len_width],
            ]
        )  # 9
        self.locations_send = locations_send

    def get_shapes_recv(self, shapes):
        shapes_recv = []

        shapes_recv.append([self.halo_len_height, self.halo_len_width])  # 1
        shapes_recv.append(
            [self.halo_len_height, shapes[3] - 2 * self.halo_len_width]
        )  # 2
        shapes_recv.append([self.halo_len_height, self.halo_len_width])  # 3

        shapes_recv.append(
            [shapes[2] - 2 * self.halo_len_height, self.halo_len_width]
        )  # 4
        shapes_recv.append([None, None])  # 5
        shapes_recv.append(
            [shapes[2] - 2 * self.halo_len_height, self.halo_len_width]
        )  # 6

        shapes_recv.append([self.halo_len_height, self.halo_len_width])  # 7
        shapes_recv.append(
            [self.halo_len_height, shapes[3] - 2 * self.halo_len_width]
        )  # 8
        shapes_recv.append([self.halo_len_height, self.halo_len_width])  # 9

        return shapes_recv

    def start_halo_exchange(self, halo_input):
        req = []
        for i in range(9):
            if self.neighbours[i] == 1:
                # print("Local rank:",self.local_rank, " to:",self.local_rank + self.rank_neighbours[i], " I:",i)
                temp = (
                    halo_input[
                        :,
                        :,
                        self.locations_send[i][0][0] : self.locations_send[i][0][1],
                        self.locations_send[i][1][0] : self.locations_send[i][1][1],
                    ]
                    .clone()
                    .detach()
                )

                torch.cuda.synchronize()

                temp_req = dist.isend(
                    temp, self.rank_neighbours[i], tag=self.send_tag[i]
                )
                req.append(temp_req)
                self.send_tag[i] += 1

        self.recv_tensors = []

        shapes = halo_input.shape
        self.halo_input_shape = shapes
        if self.shapes_recv == None:
            self.shapes_recv = self.get_shapes_recv(shapes)

        for i in range(9):
            if self.neighbours[i] == 1:
                temp_tensor = torch.zeros(
                    shapes[0],
                    shapes[1],
                    self.shapes_recv[i][0],
                    self.shapes_recv[i][1],
                    dtype=torch.float,
                    device="cuda",
                )

                """
				Synchronization is necessary at this point as all GPU operations in PyTorch are asynchronous 
				MPI copy operation is not under PyTorch therefore it can start before pytorch finishes initilization of tensor with zeros 
				It will lead to data corruption 
				Spent 1 week on this issue (data validation) 
				KEEP THIS IN MIND
				"""

                torch.cuda.synchronize()

                temp_req = dist.irecv(
                    tensor=temp_tensor,
                    src=self.rank_neighbours[i],
                    tag=self.recv_tag[i],
                )
                req.append(temp_req)
                self.recv_tag[i] += 1

                self.recv_tensors.append(temp_tensor)
            else:
                self.recv_tensors.append([])

        return req

    def end_halo_exchange(self, reqs):
        for req in reqs:
            req.wait()

    def copy_halo_exchange_values(self, halo_input):
        for i in range(9):
            if self.neighbours[i] == 1:
                halo_input[
                    :,
                    :,
                    self.locations_recv[i][0][0] : self.locations_recv[i][0][1],
                    self.locations_recv[i][1][0] : self.locations_recv[i][1][1],
                ] = self.recv_tensors[i]

    def make_tensor_halo_compute(self, halo_input):
        self.halo_input_range = 2 * self.halo_len

        # 0 1 2
        # 3 4 5
        # 6 7 8

        # horizontal

        horizontal_tensor_up = None
        if self.neighbours[0] == 1:
            # concat both 0 with 3 and 1 with 4 position
            horizontal_tensor_up = torch.cat(
                (
                    self.recv_tensors[0],
                    self.recv_tensors[3][:, :, : 2 * self.halo_len, :],
                ),
                axis=2,
            )

            horizontal_tensor_temp = torch.cat(
                (
                    self.recv_tensors[1],
                    halo_input[
                        :,
                        :,
                        self.halo_len : 3 * self.halo_len,
                        self.halo_len : -self.halo_len,
                    ],
                ),
                axis=2,
            )

            horizontal_tensor_up = torch.cat(
                (horizontal_tensor_up, horizontal_tensor_temp), axis=3
            )
        elif self.neighbours[1] == 1:
            # concat 1 with 4 position
            horizontal_tensor_up = torch.cat(
                (
                    self.recv_tensors[1],
                    halo_input[
                        :,
                        :,
                        self.halo_len : 3 * self.halo_len,
                        self.halo_len : -self.halo_len,
                    ],
                ),
                axis=2,
            )

        if self.neighbours[2] == 1:
            horizontal_tensor_temp = torch.cat(
                (
                    self.recv_tensors[2],
                    self.recv_tensors[5][:, :, : 2 * self.halo_len, :],
                ),
                axis=2,
            )

            horizontal_tensor_up = torch.cat(
                (horizontal_tensor_up, horizontal_tensor_temp), axis=3
            )

        horizontal_tensor_down = None
        if self.neighbours[6] == 1:
            # concat both 6 with 3 and 7 with 4 position
            horizontal_tensor_down = torch.cat(
                (
                    self.recv_tensors[3][:, :, -2 * self.halo_len :, :],
                    self.recv_tensors[6],
                ),
                axis=2,
            )

            horizontal_tensor_temp = torch.cat(
                (
                    halo_input[
                        :,
                        :,
                        -3 * self.halo_len : -self.halo_len,
                        self.halo_len : -self.halo_len,
                    ],
                    self.recv_tensors[7],
                ),
                axis=2,
            )

            horizontal_tensor_down = torch.cat(
                (horizontal_tensor_down, horizontal_tensor_temp), axis=3
            )
        elif self.neighbours[7] == 1:
            # concat 7 with 4 position
            horizontal_tensor_down = torch.cat(
                (
                    halo_input[
                        :,
                        :,
                        -3 * self.halo_len : -self.halo_len,
                        self.halo_len : -self.halo_len,
                    ],
                    self.recv_tensors[7],
                ),
                axis=2,
            )

        if self.neighbours[8] == 1:
            horizontal_tensor_temp = torch.cat(
                (
                    self.recv_tensors[5][:, :, -2 * self.halo_len :, :],
                    self.recv_tensors[8],
                ),
                axis=2,
            )

            horizontal_tensor_down = torch.cat(
                (horizontal_tensor_down, horizontal_tensor_temp), axis=3
            )

        # Vertical

        vertical_tensor_left = None
        if self.neighbours[0] == 1:
            vertical_tensor_left = torch.cat(
                (
                    self.recv_tensors[0],
                    self.recv_tensors[1][:, :, :, : 2 * self.halo_len],
                ),
                axis=3,
            )

            vertical_tensor_temp = torch.cat(
                (
                    self.recv_tensors[3],
                    halo_input[
                        :,
                        :,
                        self.halo_len : -self.halo_len,
                        self.halo_len : 3 * self.halo_len,
                    ],
                ),
                axis=3,
            )

            vertical_tensor_left = torch.cat(
                (vertical_tensor_left, vertical_tensor_temp), axis=2
            )
        elif self.neighbours[3] == 1:
            vertical_tensor_left = torch.cat(
                (
                    self.recv_tensors[3],
                    halo_input[
                        :,
                        :,
                        self.halo_len : -self.halo_len,
                        self.halo_len : 3 * self.halo_len,
                    ],
                ),
                axis=3,
            )

        if self.neighbours[6] == 1:
            vertical_tensor_temp = torch.cat(
                (
                    self.recv_tensors[6],
                    self.recv_tensors[7][:, :, :, : 2 * self.halo_len],
                ),
                axis=3,
            )
            vertical_tensor_left = torch.cat(
                (vertical_tensor_left, vertical_tensor_temp), axis=2
            )

        vertical_tensor_right = None
        if self.neighbours[2] == 1:
            vertical_tensor_right = torch.cat(
                (
                    self.recv_tensors[1][:, :, :, -2 * self.halo_len :],
                    self.recv_tensors[2],
                ),
                axis=3,
            )

            vertical_tensor_temp = torch.cat(
                (
                    halo_input[
                        :,
                        :,
                        self.halo_len : -self.halo_len,
                        -3 * self.halo_len : -self.halo_len,
                    ],
                    self.recv_tensors[5],
                ),
                axis=3,
            )

            vertical_tensor_right = torch.cat(
                (vertical_tensor_right, vertical_tensor_temp), axis=2
            )
        elif self.neighbours[5] == 1:
            vertical_tensor_right = torch.cat(
                (
                    halo_input[
                        :,
                        :,
                        self.halo_len : -self.halo_len,
                        -3 * self.halo_len : -self.halo_len,
                    ],
                    self.recv_tensors[5],
                ),
                axis=3,
            )

        if self.neighbours[8] == 1:
            vertical_tensor_temp = torch.cat(
                (
                    self.recv_tensors[7][:, :, :, -2 * self.halo_len :],
                    self.recv_tensors[8],
                ),
                axis=3,
            )
            vertical_tensor_right = torch.cat(
                (vertical_tensor_right, vertical_tensor_temp), axis=2
            )

        if self.neighbours[1] != 1 and self.neighbours[5] == 1:
            padding_vertical_top_right = halo_input[
                :, :, 0 : self.halo_len, -3 * self.halo_len :
            ]
            vertical_tensor_right = torch.cat(
                (padding_vertical_top_right.data, vertical_tensor_right), axis=2
            )

        if self.neighbours[1] != 1 and self.neighbours[3] == 1:
            padding_vertical_top_left = halo_input[
                :, :, 0 : self.halo_len, : 3 * self.halo_len
            ]
            vertical_tensor_left = torch.cat(
                (padding_vertical_top_left.data, vertical_tensor_left), axis=2
            )

        if self.neighbours[7] != 1 and self.neighbours[5] == 1:
            padding_vertical_down_right = halo_input[
                :, :, -self.halo_len :, -3 * self.halo_len :
            ]
            vertical_tensor_right = torch.cat(
                (vertical_tensor_right, padding_vertical_down_right), axis=2
            )

        if self.neighbours[7] != 1 and self.neighbours[3] == 1:
            padding_vertical_down_left = halo_input[
                :, :, -self.halo_len :, : 3 * self.halo_len
            ]
            vertical_tensor_left = torch.cat(
                (vertical_tensor_left, padding_vertical_down_left), axis=2
            )

        if self.neighbours[3] != 1 and self.neighbours[1] == 1:
            padding_horizontal_up_left = halo_input[
                :, :, : 3 * self.halo_len, : self.halo_len
            ]
            horizontal_tensor_up = torch.cat(
                (padding_horizontal_up_left, horizontal_tensor_up), axis=3
            )

        if self.neighbours[3] != 1 and self.neighbours[7] == 1:
            padding_horizontal_down_left = halo_input[
                :, :, -3 * self.halo_len :, : self.halo_len
            ]
            horizontal_tensor_down = torch.cat(
                (padding_horizontal_down_left, horizontal_tensor_down), axis=3
            )

        if self.neighbours[5] != 1 and self.neighbours[1] == 1:
            padding_horizontal_up_right = halo_input[
                :, :, : 3 * self.halo_len, -self.halo_len :
            ]
            horizontal_tensor_up = torch.cat(
                (horizontal_tensor_up, padding_horizontal_up_right), axis=3
            )

        if self.neighbours[5] != 1 and self.neighbours[7] == 1:
            padding_horizontal_down_right = halo_input[
                :, :, -3 * self.halo_len :, -self.halo_len :
            ]
            horizontal_tensor_down = torch.cat(
                (horizontal_tensor_down, padding_horizontal_down_right), axis=3
            )

        if horizontal_tensor_up == None and horizontal_tensor_down == None:
            horizontal_tensor = None
        elif horizontal_tensor_down == None:
            horizontal_tensor = horizontal_tensor_up
        elif horizontal_tensor_up == None:
            horizontal_tensor = horizontal_tensor_down
        else:
            horizontal_tensor = torch.cat(
                (horizontal_tensor_up, horizontal_tensor_down), axis=2
            )

        if vertical_tensor_left == None and vertical_tensor_right == None:
            vertical_tensor = None
        elif vertical_tensor_left == None:
            vertical_tensor = vertical_tensor_right
        elif vertical_tensor_right == None:
            vertical_tensor = vertical_tensor_left
        else:
            vertical_tensor = torch.cat(
                (vertical_tensor_left, vertical_tensor_right), axis=3
            )

        return horizontal_tensor, vertical_tensor

    def compute_halo_exchange(self, horizontal_tensor, vertical_tensor):
        res_horizontal, res_vertical = None, None
        if horizontal_tensor != None:
            res_horizontal = super(conv_spatial, self).forward(horizontal_tensor)
        if vertical_tensor != None:
            res_vertical = super(conv_spatial, self).forward(vertical_tensor)

        return res_horizontal, res_vertical

    def compute_halo_exchange_one(self, horizontal_tensor, vertical_tensor, halo_input):
        # print("LOCAL RANK:",self.local_rank, " Sucess")
        if self.spatial_local_rank == 0:
            None
            # print(horizontal_tensor,vertical_tensor)

        if self.spatial_local_rank == 0:
            halo_input[:, :, 5:6, :] = horizontal_tensor[:, :, 2:3, :]
            halo_input[:, :, :, -1:] = vertical_tensor[:, :, :, -1:]

        if self.spatial_local_rank == 1:
            halo_input[:, :, 5:6, :] = horizontal_tensor[:, :, 2:3, :]
            halo_input[:, :, :, 0:1] = vertical_tensor[:, :, :, 0:1]

        if self.spatial_local_rank == 2:
            halo_input[:, :, 0:1, :] = horizontal_tensor[:, :, 0:1, :]
            halo_input[:, :, :, -1:] = vertical_tensor[:, :, :, -1:]

        if self.spatial_local_rank == 3:
            halo_input[:, :, 0:1, :] = horizontal_tensor[:, :, 0:1, :]
            halo_input[:, :, :, 0:1] = vertical_tensor[:, :, :, 0:1]

        torch.cuda.synchronize()
        res = super(conv_spatial, self).forward(halo_input)

        return res

    def merge_final_image(self, res_final, res_horizontal, res_vertical):
        if self.neighbours[3] == 1:
            start = self.halo_len
        else:
            start = 0

        if self.neighbours[5] == 1:
            end = -self.halo_len
        else:
            end = None

        if self.neighbours[1] == 1:
            res_final = torch.cat(
                (res_horizontal[:, :, : self.halo_len, start:end], res_final), axis=2
            )

        if self.neighbours[7] == 1:
            if self.neighbours[1] == 1:
                res_final = torch.cat(
                    (res_final, res_horizontal[:, :, self.halo_len :, start:end]),
                    axis=2,
                )
            else:
                res_final = torch.cat(
                    (res_final, res_horizontal[:, :, :, start:end]), axis=2
                )

        if self.neighbours[3] == 1:
            res_final = torch.cat(
                (res_vertical[:, :, : self.halo_len, :], res_final), axis=3
            )

        if self.neighbours[5] == 1:
            if self.neighbours[3] == 1:
                res_final = torch.cat(
                    (res_final, res_vertical[:, :, self.halo_len :, :]), axis=3
                )
            else:
                res_final = torch.cat((res_final, res_vertical[:, :, :, :]), axis=3)
        return res_final

    def copy_final_image(self, res_final, res_horizontal, res_vertical):
        shapes = res_final.shape
        if self.neighbours[1] == 1:
            res_final[:, :, : self.halo_len, :] = res_horizontal[
                :, :, : self.halo_len, :
            ]

        if self.neighbours[7] == 1:
            if self.neighbours[1] == 1:
                res_final[:, :, -self.halo_len :, :] = res_horizontal[
                    :, :, self.halo_len :, :
                ]
            else:
                res_final[:, :, -self.halo_len :, :] = res_horizontal

        if self.neighbours[3] == 1:
            res_final[:, :, :, : self.halo_len] = res_vertical[:, :, : shapes[2], :]

        if self.neighbours[5] == 1:
            if self.neighbours[3] == 1:
                res_final[:, :, :, -self.halo_len :] = res_vertical[
                    :, :, shapes[2] :, :
                ]
            else:
                res_final[:, :, :, -self.halo_len :] = res_vertical
        return res_final

    def start_halo_exchange_nochange(self, halo_input):
        req = []
        for i in range(9):
            if self.neighbours[i] == 1:
                temp_req = dist.isend(
                    halo_input[
                        :,
                        :,
                        self.locations_send[i][0][0] : self.locations_send[i][0][1],
                        self.locations_send[i][1][0] : self.locations_send[i][1][1],
                    ],
                    self.rank_neighbours[i],
                    tag=self.send_tag[i],
                )
                req.append(temp_req)
                self.send_tag[i] += 1

        for i in range(9):
            if self.neighbours[i] == 1:
                temp_req = dist.irecv(
                    tensor=halo_input[
                        :,
                        :,
                        self.locations_recv[i][0][0] : self.locations_recv[i][0][1],
                        self.locations_recv[i][1][0] : self.locations_recv[i][1][1],
                    ],
                    src=self.rank_neighbours[i],
                    tag=self.recv_tag[i],
                )
                req.append(temp_req)
                self.recv_tag[i] += 1

        return req

    def end_halo_exchange_nochange(self, reqs):
        for req in reqs:
            req.wait()

    def get_neighbours_rank(self):
        self.rank_neighbours = []

        if self.slice_method == "square":
            # 0 1 2
            # 2 3 4
            # 5 6 7
            total_rows = int(math.sqrt(self.num_spatial_parts))
            total_cols = int(math.sqrt(self.num_spatial_parts))

            top_left = -(
                total_cols + 1
            )  # top_left will be (total_cols + 1) away from current rank
            top = -total_cols
            top_right = -(total_cols - 1)
            left = -1
            right = +1
            bottom_left = total_cols - 1
            bottom = total_cols
            bottom_right = total_cols + 1
            rank_offset = [
                top_left,
                top,
                top_right,
                left,
                0,
                right,
                bottom_left,
                bottom,
                bottom_right,
            ]

        elif self.slice_method == "vertical":
            rank_offset = [0, 0, 0, -1, 0, +1, 0, 0, 0]

        elif self.slice_method == "horizontal":
            rank_offset = [0, -1, 0, 0, 0, 0, 0, +1, 0]

        for i in range(9):
            if self.neighbours[i] == 1:
                self.rank_neighbours.append(self.local_rank + rank_offset[i])
            else:
                self.rank_neighbours.append(-1)

        # GEMS Inverse
        if self.local_rank != dist.get_rank():
            world_size = dist.get_world_size()
            for i in range(9):
                if self.neighbours[i] == 1:
                    self.rank_neighbours[i] = world_size - 1 - self.rank_neighbours[i]
                else:
                    self.rank_neighbours.append(-1)

    def set_neighbours_based_on_kernel_size(self):
        # Updates the neighbors based on kernel size
        if self.kernel_size[0] == 1:
            self.neighbours[0] = 0
            self.neighbours[1] = 0
            self.neighbours[2] = 0

            self.neighbours[6] = 0
            self.neighbours[7] = 0
            self.neighbours[8] = 0

        if self.kernel_size[1] == 1:
            self.neighbours[0] = 0
            self.neighbours[3] = 0
            self.neighbours[6] = 0

            self.neighbours[2] = 0
            self.neighbours[5] = 0
            self.neighbours[8] = 0

    def get_neighbours(self):
        # This if and else can be removed
        if self.spatial_local_rank < self.num_spatial_parts:
            self.ENABLE_SPATIAL = True
        else:
            self.ENABLE_SPATIAL = False
            self.neighbours = None
            return

        self.spatial_rank = self.spatial_local_rank

        if self.spatial_local_rank < self.num_spatial_parts:
            self.ENABLE_SPATIAL = True
        else:
            self.ENABLE_SPATIAL = False
            self.neighbours = None
            return

        self.spatial_rank = self.spatial_local_rank

        # Neighbour
        #  0   1   2
        #  3   4   5
        #  6   7   8

        if self.slice_method == "square":
            self.neighbours = []
            total_rows = int(math.sqrt(self.num_spatial_parts))
            total_cols = int(math.sqrt(self.num_spatial_parts))

            # current rank position in matrix of total_rows * total_cols
            row = self.local_rank / total_rows
            col = self.local_rank % total_cols
            dir = [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 0],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            ]

            for d in dir:
                neighbour_row = row + d[0]
                neighbour_col = col + d[1]
                if neighbour_row == row and neighbour_col == col:
                    self.neighbours.append(0)
                elif (
                    neighbour_row < 0
                    or neighbour_row >= total_rows
                    or neighbour_col < 0
                    or neighbour_col >= total_cols
                ):
                    self.neighbours.append(0)
                else:
                    self.neighbours.append(1)

        elif self.slice_method == "vertical":
            if self.spatial_rank == 0:
                self.neighbours = [0, 0, 0, 0, 0, 1, 0, 0, 0]
            elif self.spatial_rank == self.num_spatial_parts - 1:
                self.neighbours = [0, 0, 0, 1, 0, 0, 0, 0, 0]
            else:
                self.neighbours = [0, 0, 0, 1, 0, 1, 0, 0, 0]

        elif self.slice_method == "horizontal":
            if self.spatial_rank == 0:
                self.neighbours = [0, 0, 0, 0, 0, 0, 0, 1, 0]
            elif self.spatial_rank == self.num_spatial_parts - 1:
                self.neighbours = [0, 1, 0, 0, 0, 0, 0, 0, 0]
            else:
                self.neighbours = [0, 1, 0, 0, 0, 0, 0, 1, 0]

        self.set_neighbours_based_on_kernel_size()

    def forward(self, tensor):
        tensor = self.pads(tensor)
        if self.halo_len_height > 0 or self.halo_len_width > 0:
            torch.cuda.synchronize()
            reqs = self.start_halo_exchange(tensor)
            self.end_halo_exchange(reqs)
            self.copy_halo_exchange_values(tensor)
            # torch.cuda.synchronize()
        res_final = super(conv_spatial, self).forward(tensor)

        return res_final

    """

	def forward(self,input):
		#print("Awesome",self.neighbours, self.rank_neighbours)
		s = torch.cuda.Stream()
		halo_input = self.padding_layer(input)

		#self.weight = torch.nn.Parameter(self.weight.int())

		#self.bias= torch.nn.Parameter(self.bias.int())
		torch.cuda.synchronize()

		if(self.halo_len>0):

			with torch.cuda.stream(s):
				torch.cuda.synchronize()
				reqs = self.start_halo_exchange(halo_input)
				self.end_halo_exchange(reqs)
				s.synchronize()
				#self.copy_halo_exchange_values(halo_input)

				horizontal_tensor, vertical_tensor = self.make_tensor_halo_compute(halo_input)
				s.synchronize()
				res_final = self.compute_halo_exchange_one(horizontal_tensor,vertical_tensor,halo_input)


			s.synchronize()
			torch.cuda.synchronize()
			
			return res_final
		else:
			res_final = super(conv_spatial,self).forward(halo_input)
			return res_final
	"""


class halo_exchange_layer(nn.Module):
    def __init__(
        self,
        local_rank,
        spatial_size,
        num_spatial_parts,
        halo_len,
        padding_mode="zeros",
        slice_method="square",
    ):
        super(halo_exchange_layer, self).__init__()
        # slice_method: "vertical" "horizontal" "square"
        self.slice_method = slice_method

        if isinstance(num_spatial_parts, list):
            self.local_rank = local_rank
            (
                self.spatial_local_rank,
                self.num_spatial_parts,
            ) = self.get_local_spatial_rank(num_spatial_parts, local_rank)
        else:
            self.local_rank = local_rank
            self.spatial_local_rank = local_rank
            self.num_spatial_parts = num_spatial_parts

        # number of sequential spatial layers, Most of the times I expect it to be 1
        self.spatial_size = spatial_size

        padding = halo_len

        self.halo_len = halo_len

        if self.spatial_local_rank == 0:
            padding_left, padding_right, padding_top, padding_bottom = (
                padding,
                halo_len,
                padding,
                halo_len,
            )
        elif self.spatial_local_rank == 1:
            padding_left, padding_right, padding_top, padding_bottom = (
                halo_len,
                padding,
                padding,
                halo_len,
            )
        elif self.spatial_local_rank == 2:
            padding_left, padding_right, padding_top, padding_bottom = (
                padding,
                halo_len,
                halo_len,
                padding,
            )
        elif self.spatial_local_rank == 3:
            padding_left, padding_right, padding_top, padding_bottom = (
                halo_len,
                padding,
                halo_len,
                padding,
            )
        else:  # TBD: REPLACE
            padding_left, padding_right, padding_top, padding_bottom = (
                halo_len,
                padding,
                halo_len,
                padding,
            )
        self.pads = nn.ZeroPad2d(
            (padding_left, padding_right, padding_top, padding_bottom)
        )

        self.get_neighbours()
        self.get_neighbours_rank()
        self.get_index_locations()
        self.shapes_recv = None
        self.recv_tensors = []
        self.send_tensors = []
        self.set_tags()

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

    def set_tags(self):
        self.send_tag = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        self.recv_tag = [900, 800, 700, 600, 500, 400, 300, 200, 100]

        # self.send_tag = [0]*9
        # self.recv_tag = [0]*9

    def get_index_locations(self):
        locations_recv = []
        locations_recv.append([[None, self.halo_len], [None, self.halo_len]])  # 1
        locations_recv.append(
            [[None, self.halo_len], [self.halo_len, -self.halo_len]]
        )  # 2
        locations_recv.append([[None, self.halo_len], [-self.halo_len, None]])  # 3
        locations_recv.append(
            [[self.halo_len, -self.halo_len], [None, self.halo_len]]
        )  # 4
        locations_recv.append([[None, None], [None, None]])  # 5
        locations_recv.append(
            [[self.halo_len, -self.halo_len], [-self.halo_len, None]]
        )  # 6
        locations_recv.append([[-self.halo_len, None], [None, self.halo_len]])  # 7
        locations_recv.append(
            [[-self.halo_len, None], [self.halo_len, -self.halo_len]]
        )  # 8
        locations_recv.append([[-self.halo_len, None], [-self.halo_len, None]])  # 9

        self.locations_recv = locations_recv

        locations_send = []
        locations_send.append(
            [[self.halo_len, 2 * self.halo_len], [self.halo_len, 2 * self.halo_len]]
        )  # 1
        locations_send.append(
            [[self.halo_len, 2 * self.halo_len], [self.halo_len, -self.halo_len]]
        )  # 2
        locations_send.append(
            [
                [self.halo_len, 2 * self.halo_len],
                [-2 * self.halo_len, -1 * self.halo_len],
            ]
        )  # 3
        locations_send.append(
            [[self.halo_len, -self.halo_len], [self.halo_len, 2 * self.halo_len]]
        )  # 4
        locations_send.append([[None, None], [None, None]])  # 5
        locations_send.append(
            [[self.halo_len, -self.halo_len], [-2 * self.halo_len, -1 * self.halo_len]]
        )  # 6
        locations_send.append(
            [
                [-2 * self.halo_len, -1 * self.halo_len],
                [self.halo_len, 2 * self.halo_len],
            ]
        )  # 7
        locations_send.append(
            [[-2 * self.halo_len, -1 * self.halo_len], [self.halo_len, -self.halo_len]]
        )  # 8
        locations_send.append(
            [
                [-2 * self.halo_len, -1 * self.halo_len],
                [-2 * self.halo_len, -1 * self.halo_len],
            ]
        )  # 9
        self.locations_send = locations_send

    def get_shapes_recv(self, shapes):
        shapes_recv = []

        shapes_recv.append([self.halo_len, self.halo_len])  # 1
        shapes_recv.append([self.halo_len, shapes[3] - 2 * self.halo_len])  # 2
        shapes_recv.append([self.halo_len, self.halo_len])  # 3

        shapes_recv.append([shapes[2] - 2 * self.halo_len, self.halo_len])  # 4
        shapes_recv.append([None, None])  # 5
        shapes_recv.append([shapes[2] - 2 * self.halo_len, self.halo_len])  # 6

        shapes_recv.append([self.halo_len, self.halo_len])  # 7
        shapes_recv.append([self.halo_len, shapes[3] - 2 * self.halo_len])  # 8
        shapes_recv.append([self.halo_len, self.halo_len])  # 9

        return shapes_recv

    def start_halo_exchange(self, halo_input):
        req = []
        for i in range(9):
            if self.neighbours[i] == 1:
                # print("Local rank:",self.local_rank, " to:",self.local_rank + self.rank_neighbours[i], " I:",i)
                temp = (
                    halo_input[
                        :,
                        :,
                        self.locations_send[i][0][0] : self.locations_send[i][0][1],
                        self.locations_send[i][1][0] : self.locations_send[i][1][1],
                    ]
                    .clone()
                    .detach()
                )

                torch.cuda.synchronize()

                temp_req = dist.isend(
                    temp, self.rank_neighbours[i], tag=self.send_tag[i]
                )
                req.append(temp_req)
                self.send_tag[i] += 1

        self.recv_tensors = []

        shapes = halo_input.shape
        self.halo_input_shape = shapes
        if self.shapes_recv == None:
            self.shapes_recv = self.get_shapes_recv(shapes)

        for i in range(9):
            if self.neighbours[i] == 1:
                temp_tensor = torch.zeros(
                    shapes[0],
                    shapes[1],
                    self.shapes_recv[i][0],
                    self.shapes_recv[i][1],
                    dtype=torch.float,
                    device="cuda",
                )

                """
				Synchronization is necessary at this point as all GPU operations in PyTorch are asynchronous 
				MPI copy operation is not under PyTorch therefore it can start before pytorch finishes initilization of tensor with zeros 
				It will lead to data corruption 
				Spent 1 week on this issue (data validation) 
				KEEP THIS IN MIND
				"""

                torch.cuda.synchronize()

                temp_req = dist.irecv(
                    tensor=temp_tensor,
                    src=self.rank_neighbours[i],
                    tag=self.recv_tag[i],
                )
                req.append(temp_req)
                self.recv_tag[i] += 1

                self.recv_tensors.append(temp_tensor)
            else:
                self.recv_tensors.append([])

        return req

    def end_halo_exchange(self, reqs):
        for req in reqs:
            req.wait()

    def get_neighbours_rank(self):
        self.rank_neighbours = []

        if self.slice_method == "square":
            # 0 1 2
            # 2 3 4
            # 5 6 7
            total_rows = int(math.sqrt(self.num_spatial_parts))
            total_cols = int(math.sqrt(self.num_spatial_parts))

            top_left = -(
                total_cols + 1
            )  # top_left will be (total_cols + 1) away from current rank
            top = -total_cols
            top_right = -(total_cols - 1)
            left = -1
            right = +1
            bottom_left = total_cols - 1
            bottom = total_cols
            bottom_right = total_cols + 1
            rank_offset = [
                top_left,
                top,
                top_right,
                left,
                0,
                right,
                bottom_left,
                bottom,
                bottom_right,
            ]

        elif self.slice_method == "vertical":
            rank_offset = [0, 0, 0, -1, 0, +1, 0, 0, 0]

        elif self.slice_method == "horizontal":
            rank_offset = [0, -1, 0, 0, 0, 0, 0, +1, 0]

        for i in range(9):
            if self.neighbours[i] == 1:
                self.rank_neighbours.append(self.local_rank + rank_offset[i])
            else:
                self.rank_neighbours.append(-1)

        # GEMS Inverse
        if self.local_rank != dist.get_rank():
            world_size = dist.get_world_size()
            for i in range(9):
                if self.neighbours[i] == 1:
                    self.rank_neighbours[i] = world_size - 1 - self.rank_neighbours[i]
                else:
                    self.rank_neighbours.append(-1)

    def get_neighbours(self):
        if self.spatial_local_rank < self.num_spatial_parts:
            self.ENABLE_SPATIAL = True
        else:
            self.ENABLE_SPATIAL = False
            self.neighbours = None
            return

        self.spatial_rank = self.spatial_local_rank

        # Neighbour
        #  0   1   2
        #  3   4   5
        #  6   7   8
        if self.slice_method == "square":
            self.neighbours = []
            total_rows = int(math.sqrt(self.num_spatial_parts))
            total_cols = int(math.sqrt(self.num_spatial_parts))

            # current rank position in matrix of total_rows * total_cols
            row = self.local_rank / total_rows
            col = self.local_rank % total_cols
            dir = [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 0],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            ]

            for d in dir:
                neighbour_row = row + d[0]
                neighbour_col = col + d[1]
                if neighbour_row == row and neighbour_col == col:
                    self.neighbours.append(0)
                elif (
                    neighbour_row < 0
                    or neighbour_row >= total_rows
                    or neighbour_col < 0
                    or neighbour_col >= total_cols
                ):
                    self.neighbours.append(0)
                else:
                    self.neighbours.append(1)

        elif self.slice_method == "vertical":
            if self.spatial_rank == 0:
                self.neighbours = [0, 0, 0, 0, 0, 1, 0, 0, 0]
            elif self.spatial_rank == self.num_spatial_parts - 1:
                self.neighbours = [0, 0, 0, 1, 0, 0, 0, 0, 0]
            else:
                self.neighbours = [0, 0, 0, 1, 0, 1, 0, 0, 0]

        elif self.slice_method == "horizontal":
            if self.spatial_rank == 0:
                self.neighbours = [0, 0, 0, 0, 0, 0, 0, 1, 0]
            elif self.spatial_rank == self.num_spatial_parts - 1:
                self.neighbours = [0, 1, 0, 0, 0, 0, 0, 0, 0]
            else:
                self.neighbours = [0, 1, 0, 0, 0, 0, 0, 1, 0]

    def copy_halo_exchange_values(self, halo_input):
        for i in range(9):
            if self.neighbours[i] == 1:
                halo_input[
                    :,
                    :,
                    self.locations_recv[i][0][0] : self.locations_recv[i][0][1],
                    self.locations_recv[i][1][0] : self.locations_recv[i][1][1],
                ] = self.recv_tensors[i]

    def forward(self, tensor):
        tensor = self.pads(tensor)

        torch.cuda.synchronize()
        reqs = self.start_halo_exchange(tensor)
        self.end_halo_exchange(reqs)
        self.copy_halo_exchange_values(tensor)
        torch.cuda.synchronize()

        return tensor


class Pool(nn.Module):
    def __init__(
        self,
        local_rank,
        spatial_size,
        num_spatial_parts,
        kernel_size,
        stride,
        padding,
        slice_method="square",
        dilation=1,
        return_indices=False,
        count_include_pad=True,
        divisor_override=None,
        ceil_mode=False,
        operation=None,
    ) -> None:
        """
        Kernel size: tuple of ints (int, int) or int
        Stride: tuple of ints (int, int) or int
        Padding: tuple of ints (int, int) or int



        count_include_pad = True will not work
        I don't have any solution.
        """
        super(Pool, self).__init__()

        assert dilation == 1, "dilation > 1, Not Supported"
        assert return_indices == False, "return_indices == True, not supported"
        assert ceil_mode == False, "ceil model == True, not supported"
        assert operation != None, "operation is none"

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        halo_len = math.floor((kernel_size[0] - 1) / 2)

        assert kernel_size[0] == kernel_size[1], "Kernel Size should be same in pooling"
        assert stride[0] == stride[1], "Stride should be same in pooling"
        assert padding[0] == padding[1], "Padding should be same in pooling"
        assert (
            halo_len == padding[0]
        ), "halo_len should be equal to padding in pool layers "

        self.halo_len = halo_len
        self.padding = padding

        if halo_len != 0:
            self.halo_len_layer = halo_exchange_layer(
                local_rank=local_rank,
                spatial_size=spatial_size,
                num_spatial_parts=num_spatial_parts,
                halo_len=halo_len,
                slice_method=slice_method,
            )

        if operation == "MaxPool2d":
            # getattr(torch.nn, 'MaxPool2d')
            self.pool = getattr(torch.nn, "MaxPool2d")(
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

        elif operation == "AvgPool2d":
            # getattr(torch.nn, 'MaxPool2d')
            self.pool = getattr(torch.nn, "AvgPool2d")(
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            )

        else:
            assert False, "Only MaxPool2d and AvgPool2d are supported"

    def forward(self, tensor):
        x = tensor
        if self.halo_len != 0:
            x = self.halo_len_layer(x)
        x = self.pool(x)

        return x
