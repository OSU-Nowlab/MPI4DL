import torch.nn as nn
import torch
import torch.distributed as dist
import time


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
        padding_mode="zeros",
    ):
        self.local_rank = local_rank
        # number of parts in one image
        self.num_spatial_parts = num_spatial_parts
        # number of sequential spatial layers, Most of the times I expect it to be 1
        self.spatial_size = spatial_size
        self.get_neighbours()
        self.get_neighbours_rank()

        self.halo_len = (kernel_size - 1) / 2

        assert (
            self.halo_len == padding
        ), "Spatial not supported yet for this configuration"
        self.halo_len = padding
        super(conv_spatial, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

        self.padding_layer = nn.ZeroPad2d(padding)
        self.get_index_locations()
        self.shapes_recv = None
        self.recv_tensors = []
        self.send_tensors = []

        self.set_tags()

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
                to_send = (
                    halo_input[
                        :,
                        :,
                        self.locations_send[i][0][0] : self.locations_send[i][0][1],
                        self.locations_send[i][1][0] : self.locations_send[i][1][1],
                    ]
                    .clone()
                    .detach()
                )
                # torch.cuda.synchronize()

                t = time.time()
                temp_req = dist.isend(
                    to_send, self.rank_neighbours[i], tag=self.send_tag[i]
                )
                if self.local_rank == 0:
                    None
                    # print("sending to:",self.rank_neighbours[i], " Shape:", to_send.shape, " Time taken:" ,time.time()-t)
                req.append(temp_req)
                self.send_tag[i] += 1

        # self.recv_tensors = []
        if len(self.recv_tensors) == 0:
            flag_recv_tensors_init = 0
        else:
            flag_recv_tensors_init = 1

        shapes = halo_input.shape
        self.halo_input_shape = shapes
        if self.shapes_recv == None:
            self.shapes_recv = self.get_shapes_recv(shapes)

        for i in range(9):
            if self.neighbours[i] == 1:
                if flag_recv_tensors_init == 0:
                    temp_tensor = torch.zeros(
                        shapes[0],
                        shapes[1],
                        self.shapes_recv[i][0],
                        self.shapes_recv[i][1],
                        dtype=torch.float,
                        device="cuda",
                    )
                    self.recv_tensors.append(temp_tensor)

                temp_req = dist.irecv(
                    tensor=self.recv_tensors[i],
                    src=self.rank_neighbours[i],
                    tag=self.recv_tag[i],
                )
                req.append(temp_req)
                self.recv_tag[i] += 1

                # self.recv_tensors.append(temp_tensor)
            else:
                if flag_recv_tensors_init == 0:
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
        if self.local_rank == 0:
            None
            # print(horizontal_tensor,vertical_tensor)

        if self.local_rank == 0:
            halo_input[:, :, 5:6, :] = horizontal_tensor[:, :, 2:3, :]
            halo_input[:, :, :, -1:] = vertical_tensor[:, :, :, -1:]

        if self.local_rank == 1:
            halo_input[:, :, 5:6, :] = horizontal_tensor[:, :, 2:3, :]
            halo_input[:, :, :, 0:1] = vertical_tensor[:, :, :, 0:1]

        if self.local_rank == 2:
            halo_input[:, :, 0:1, :] = horizontal_tensor[:, :, 0:1, :]
            halo_input[:, :, :, -1:] = vertical_tensor[:, :, :, -1:]

        if self.local_rank == 3:
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
        if self.num_spatial_parts == 2:
            rank_offset = [0, 0, 0, -1, 0, +1, 0, 0, 0]
        elif self.num_spatial_parts == 4:
            rank_offset = [-3, -2, -1, -1, 0, +1, +1, +2, +3]
        elif self.num_spatial_parts == 9:
            rank_offset = [-4, -3, -2, -1, 0, +1, +2, +3, +4]

        for i in range(9):
            if self.neighbours[i] == 1:
                self.rank_neighbours.append(self.local_rank + rank_offset[i])
            else:
                self.rank_neighbours.append(-1)

    def get_neighbours(self):
        if self.local_rank < self.num_spatial_parts * self.spatial_size:
            self.ENABLE_SPATIAL = True
        else:
            self.ENABLE_SPATIAL = False
            self.neighbours = None
            return

        self.spatial_rank = self.local_rank % self.num_spatial_parts

        # Neighbour
        #  0   1   2
        #  3   4   5
        #  6   7   8
        if self.num_spatial_parts == 2:
            # 0 | 1
            if self.spatial_rank == 0:
                self.neighbours = [0, 0, 0, 0, 0, 1, 0, 0, 0]
            else:
                self.neighbours = [0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif self.num_spatial_parts == 4:
            # 0 | 1
            # -----
            # 2 | 3
            if self.spatial_rank == 0:
                self.neighbours = [0, 0, 0, 0, 0, 1, 0, 1, 1]
            elif self.spatial_rank == 1:
                self.neighbours = [0, 0, 0, 1, 0, 0, 1, 1, 0]
            elif self.spatial_rank == 2:
                self.neighbours = [0, 1, 1, 0, 0, 1, 0, 0, 0]
            elif self.spatial_rank == 3:
                self.neighbours = [1, 1, 0, 1, 0, 0, 0, 0, 0]

        elif self.num_spatial_parts == 9:
            # 0 | 1 | 2
            # -----------
            # 3 | 4 | 5
            # -----------
            # 6 | 7 | 8
            if self.spatial_rank == 0:
                self.neighbours = [0, 0, 0, 0, 0, 1, 0, 1, 1]
            elif self.spatial_rank == 1:
                self.neighbours = [0, 0, 0, 1, 0, 1, 1, 1, 1]
            elif self.spatial_rank == 2:
                self.neighbours = [0, 0, 0, 1, 0, 0, 1, 1, 0]
            elif self.spatial_rank == 3:
                self.neighbours = [0, 1, 1, 0, 0, 1, 0, 1, 1]
            elif self.spatial_rank == 4:
                self.neighbours = [1, 1, 1, 1, 0, 1, 1, 1, 1]
            elif self.spatial_rank == 5:
                self.neighbours = [1, 1, 0, 1, 0, 0, 1, 1, 0]
            elif self.spatial_rank == 6:
                self.neighbours = [0, 1, 1, 0, 0, 1, 0, 0, 0]
            elif self.spatial_rank == 7:
                self.neighbours = [1, 1, 1, 1, 0, 1, 0, 0, 0]
            elif self.spatial_rank == 8:
                self.neighbours = [1, 1, 0, 1, 0, 0, 0, 0, 0]

    def forward(self, input):
        s = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        halo_input = self.padding_layer(input)
        torch.cuda.synchronize()

        if self.halo_len > 0:
            with torch.cuda.stream(s):
                # torch.cuda.synchronize()
                t = time.time()

                reqs = self.start_halo_exchange(halo_input)
                # print(time.time()-t)

                self.end_halo_exchange(reqs)

                if self.local_rank == 0:
                    print(time.time() - t)

                res_final = super(conv_spatial, self).forward(halo_input)

                s.synchronize()
                # self.copy_halo_exchange_values(halo_input)
                # horizontal_tensor, vertical_tensor = self.make_tensor_halo_compute(halo_input)
                # s.synchronize()
                # res_horizontal, res_vertical = self.compute_halo_exchange(horizontal_tensor,vertical_tensor)

            # s.synchronize()
            # torch.cuda.synchronize()

            # s.synchronize()

            # print("Local rank:",self.local_rank,res_final.shape, res_horizontal.shape, res_vertical.shape)
            # res_final = self.copy_final_image(res_final,res_horizontal,res_vertical)

            return res_final
        else:
            res_final = super(conv_spatial, self).forward(halo_input)
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
