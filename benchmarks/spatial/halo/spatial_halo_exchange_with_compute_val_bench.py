import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import os
import time
import math
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Halo exchange benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fp16-allreduce",
        action="store_true",
        default=False,
        help="use fp16 compression during allreduce",
    )

    parser.add_argument("--image-size", type=int, default=8, help="Full image size")
    parser.add_argument("--batch-size", type=int, default=1, help="input batch size")
    parser.add_argument("--halo-len", type=int, default=1, help="halo length")
    parser.add_argument("--warmup", type=int, default=10, help="warmups")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    parser.add_argument(
        "--in-channels", type=int, default=1, help="number of input channels"
    )
    parser.add_argument(
        "--out-channels", type=int, default=256, help="number of output channels"
    )
    parser.add_argument(
        "--num-spatial-parts",
        type=int,
        default="4",
        help="Number of partitions in spatial parallelism",
    )

    parser.add_argument(
        "--slice-method",
        type=str,
        default="square",
        help="Slice method (square, vertical, and horizontal)",
    )

    return parser


parser_obj = get_parser()
args = parser_obj.parse_args()

warmup = args.warmup
slice_method = args.slice_method
num_spatial_parts = args.num_spatial_parts


def validate_config(num_spatial_parts, comm_size):
    assert (
        num_spatial_parts > 1
    ), "num_spatial_parts should be greater than 1 for spatial parallelism."
    assert (
        comm_size >= num_spatial_parts
    ), "Spatial parts {num_spatial_parts} require {num_spatial_parts} GPUs."

    if slice_method == "square":
        parts = int(math.sqrt(num_spatial_parts))
        assert (
            parts * parts == num_spatial_parts
        ), "Invalid number of spatial parts for square spatial type"


class halo_bench_pt2pt(nn.Conv2d):
    def __init__(
        self, local_rank, num_spatial_parts, halo_len, in_channels, out_channels
    ):
        self.local_rank = local_rank
        self.slice_method = slice_method
        # number of parts in one image
        self.num_spatial_parts = num_spatial_parts
        self.halo_len = halo_len
        self.shapes_recv = None
        self.recv_tensors = []

        self.get_neighbours()
        self.get_neighbours_rank()
        self.set_tags()
        self.get_index_locations()
        super(halo_bench_pt2pt, self).__init__(
            in_channels,
            out_channels,
            kernel_size=halo_len * 2 + 1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

    def set_tags(self):
        self.send_tag = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        self.recv_tag = [900, 800, 700, 600, 500, 400, 300, 200, 100]

    def get_neighbours_rank(self):
        self.rank_neighbours = []

        if self.slice_method == "square":
            # 0 1 2
            # 2 3 4
            # 5 6 7
            total_rows = int(math.sqrt(self.num_spatial_parts))
            total_cols = int(math.sqrt(self.num_spatial_parts))

            # top_left will be (total_cols + 1) away from current rank
            top_left = -(total_cols + 1)
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

    def get_neighbours(self):
        if self.local_rank < self.num_spatial_parts:
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
				Synchronization is necessary at this point as all GPU operations 
    			in PyTorch are asynchronous. MPI copy operation is not under 
       			PyTorch therefore it can start before pytorch finishes 
          		initilization of tensor with zeros. It will lead to data 
            	corruption.
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

    def run(self, tensor):

        reqs = self.start_halo_exchange(tensor)
        self.end_halo_exchange(reqs)
        self.copy_halo_exchange_values(tensor)
        # torch.cuda.synchronize()
        res_final = super(halo_bench_pt2pt, self).forward(tensor)

        return tensor, res_final
        print("Rank:", self.local_rank, "\n", tensor)


def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def initialize_cuda():
    my_local_rank = env2int(
        ["MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"],
        0,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(my_local_rank % 4)

    torch.cuda.init()


def init_comm(backend="mpi"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend)
    size = dist.get_world_size()
    rank = dist.get_rank()
    return size, rank


def create_input_vertical(halo_len, image_size, num_spatial_parts, rank):
    image_height_local = int(image_size[0])
    image_width_local = int(image_size[1] / num_spatial_parts)  # use num_spatial_parts

    np_x = np.asarray(
        list(
            range(0, image_size[0] * image_size[1] * args.in_channels * args.batch_size)
        ),
        dtype=np.float32,
    )
    np_x.resize(args.batch_size, args.in_channels, image_size[0], image_size[1])

    pad_width = [(0, 0), (0, 0), (halo_len, halo_len), (halo_len, halo_len)]

    expected_output = np.pad(np_x, pad_width=pad_width, mode="constant")

    expected_out_width = image_width_local + 2 * halo_len
    expected_out_height = image_height_local + 2 * halo_len

    start_left = rank * image_width_local
    end_right = (rank + 1) * image_width_local + 2 * halo_len

    if rank == num_spatial_parts - 1:
        # In case of odd number of GPUs, partition size will be uneven and last
        # rank will receive remaining image
        expected_output = expected_output[:, :, :, start_left:]
    else:
        expected_output = expected_output[:, :, :, start_left:end_right]

    start_left_i = rank * image_width_local
    end_right_i = (rank + 1) * image_width_local

    if rank == num_spatial_parts - 1:
        # In case of odd number of GPUs, partition size will be uneven and last
        # rank will receive remaining image
        input_local = np_x[:, :, :, start_left_i:]
    else:
        input_local = np_x[:, :, :, start_left_i:end_right_i]

    input_tensor_local = torch.tensor(input_local, dtype=torch.float, device="cuda")
    pads = nn.ZeroPad2d(halo_len)
    input_tensor_local = pads(input_tensor_local)

    return input_tensor_local, expected_output


def create_input_horizontal(halo_len, image_size, num_spatial_parts, rank):
    image_height_local = int(image_size[0] / num_spatial_parts)  # use num_spatial_parts
    image_width_local = int(image_size[1])

    np_x = np.asarray(
        list(
            range(0, image_size[0] * image_size[1] * args.in_channels * args.batch_size)
        ),
        dtype=np.float32,
    )
    np_x.resize(args.batch_size, args.in_channels, image_size[0], image_size[1])

    pad_width = [(0, 0), (0, 0), (halo_len, halo_len), (halo_len, halo_len)]

    expected_output = np.pad(np_x, pad_width=pad_width, mode="constant")

    expected_out_width = image_width_local + 2 * halo_len
    expected_out_height = image_height_local + 2 * halo_len

    start_top = rank * image_height_local
    end_bottom = (rank + 1) * image_height_local + 2 * halo_len

    if rank == num_spatial_parts - 1:
        # In case of odd number of GPUs, partition size will be uneven and last
        # rank will receive remaining image
        expected_output = expected_output[:, :, start_top:, :]
    else:
        expected_output = expected_output[:, :, start_top:end_bottom, :]

    start_top_i = rank * image_height_local
    end_bottom_i = (rank + 1) * image_height_local

    if rank == num_spatial_parts - 1:
        # In case of odd number of GPUs, partition size will be uneven and last
        # rank will receive remaining image
        input_local = np_x[:, :, start_top_i:, :]
    else:
        input_local = np_x[:, :, start_top_i:end_bottom_i, :]

    input_tensor_local = torch.tensor(input_local, dtype=torch.float, device="cuda")
    pads = nn.ZeroPad2d(halo_len)
    input_tensor_local = pads(input_tensor_local)

    return input_tensor_local, expected_output


def create_input_square(halo_len, image_size, num_spatial_parts, rank):
    image_height_local = int(image_size[0] / math.sqrt(num_spatial_parts))
    image_width_local = int(image_size[1] / math.sqrt(num_spatial_parts))

    np_x = np.asarray(
        list(
            range(0, image_size[0] * image_size[1] * args.in_channels * args.batch_size)
        ),
        dtype=np.float32,
    )
    np_x.resize(args.batch_size, args.in_channels, image_size[0], image_size[1])

    pad_width = [(0, 0), (0, 0), (halo_len, halo_len), (halo_len, halo_len)]

    total_rows = int(math.sqrt(num_spatial_parts))
    total_cols = int(math.sqrt(num_spatial_parts))
    # position of rank in matrix math.sqrt(num_spatial_parts) * math.sqrt(num_spatial_parts)
    row = int(rank / total_rows)
    col = int(rank % total_cols)

    expected_output = np.pad(np_x, pad_width=pad_width, mode="constant")

    expected_out_width = image_width_local + 2 * halo_len
    expected_out_height = image_height_local + 2 * halo_len

    e_left_idx = col * image_width_local
    e_right_idx = (col + 1) * image_width_local + 2 * halo_len

    e_top_idx = row * image_height_local
    e_bottom_idx = (row + 1) * image_height_local + 2 * halo_len

    expected_output = expected_output[
        :, :, e_top_idx:e_bottom_idx, e_left_idx:e_right_idx
    ]

    left_idx = col * image_width_local
    right_idx = (col + 1) * image_width_local

    top_idx = row * image_height_local
    bottom_idx = (row + 1) * image_height_local

    input_local = np_x[:, :, top_idx:bottom_idx, left_idx:right_idx]

    input_tensor_local = torch.tensor(input_local, dtype=torch.float, device="cuda")
    pads = nn.ZeroPad2d(halo_len)
    input_tensor_local = pads(input_tensor_local)

    return input_tensor_local, expected_output


def create_input(halo_len, image_size, num_spatial_parts, rank, slice_method):
    if slice_method == "vertical":
        return create_input_vertical(halo_len, image_size, num_spatial_parts, rank)
    elif slice_method == "horizontal":
        return create_input_horizontal(halo_len, image_size, num_spatial_parts, rank)
    elif slice_method == "square":
        return create_input_square(halo_len, image_size, num_spatial_parts, rank)


def test_output_square(image_size, output, expected_output, rank, size):
    # only padding ==  halo_len case is supported
    image_height_local = int(image_size[0] / math.sqrt(size))
    image_width_local = int(image_size[1] / math.sqrt(size))
    expected_out_width = image_width_local
    expected_out_height = image_height_local

    total_rows = int(math.sqrt(size))
    total_cols = int(math.sqrt(size))

    row = int(rank / total_rows)
    col = int(rank % total_cols)

    e_left_idx = col * expected_out_width
    e_right_idx = (col + 1) * expected_out_width

    e_top_idx = row * expected_out_height
    e_bottom_idx = (row + 1) * expected_out_height

    expected_output = expected_output[
        :, :, e_top_idx:e_bottom_idx, e_left_idx:e_right_idx
    ]

    expected_output = expected_output.detach().cpu().numpy()
    output = output.detach().cpu().numpy()

    if np.equal(output, expected_output).all():
        print(" Validation passed Rank:" + str(rank))
    else:
        if rank == 0:
            # debug statements
            # uneq = np.not_equal(output,expected_output)
            # print("Rank:"+str(rank), output[uneq].size , expected_output[uneq].size)
            None

        print(
            " Validation failed Rank.............................................................................:"
            + str(rank)
        )


def test_output_vertical(image_size, output, expected_output, rank, size):
    # only padding ==  halo_len case is supported
    image_height_local = int(image_size[0] / (size))
    image_width_local = int(image_size[1] / (size))
    expected_out_width = image_width_local
    expected_out_height = image_height_local

    start_left = rank * image_width_local
    end_right = (rank + 1) * image_width_local

    if rank == size - 1:
        # In case of odd number of GPUs, partition size will be uneven and last
        # rank will receive remaining image
        expected_output = expected_output[:, :, :, start_left:]
    else:
        expected_output = expected_output[:, :, :, start_left:end_right]

    expected_output = expected_output.detach().cpu().numpy()
    output = output.detach().cpu().numpy()

    if np.equal(output, expected_output).all():
        print(" Validation passed Rank:" + str(rank))
    else:
        if rank == 0:
            # debug statements
            # uneq = np.not_equal(output,expected_output)
            # print("Rank:"+str(rank), output[uneq].size , expected_output[uneq].size)
            None

        print(
            " Validation failed Rank.............................................................................:"
            + str(rank)
        )


def test_output_horizontal(image_size, output, expected_output, rank, size):
    # only padding ==  halo_len case is supported
    image_height_local = int(image_size[0] / (size))
    image_width_local = int(image_size[1] / (size))
    expected_out_width = image_width_local
    expected_out_height = image_height_local

    start_top = rank * image_height_local
    end_bottom = (rank + 1) * image_height_local

    if rank == size - 1:
        expected_output = expected_output[:, :, start_top:, :]
    else:
        expected_output = expected_output[:, :, start_top:end_bottom, :]

    expected_output = expected_output.detach().cpu().numpy()
    output = output.detach().cpu().numpy()

    if np.equal(output, expected_output).all():
        print(" Validation passed Rank:" + str(rank))
    else:
        if rank == 0:
            # debug statements
            # uneq = np.not_equal(output,expected_output)
            # print("Rank:"+str(rank), output[uneq].size , expected_output[uneq].size)
            None

        print(
            " Validation failed Rank.............................................................................:"
            + str(rank)
        )


def test_output(image_size, output, expected_output, rank, size):
    if args.slice_method == "vertical":

        test_output_vertical(image_size, output, expected_output, rank, size)

    elif args.slice_method == "horizontal":

        test_output_horizontal(image_size, output, expected_output, rank, size)

    elif args.slice_method == "square":

        test_output_square(image_size, output, expected_output, rank, size)


def test_output_recv(output, expected_output, rank):
    np_out = output.to("cpu").numpy()

    if np.equal(np_out, expected_output).all():
        print("Recv Validation passed Rank:" + str(rank))
    else:
        uneq = np.not_equal(np_out.astype("int"), expected_output.astype("int"))
        print(
            "Recv Rank:" + str(rank),
            np_out.astype("int")[uneq],
            expected_output.astype("int")[uneq],
        )
        print("Recv Validation failed Rank:" + str(rank))


halo_len = args.halo_len
iterations = args.iterations


initialize_cuda()
size, rank = init_comm()

image_size = (args.image_size, args.image_size)


input_tensor_local, expected_output_recv = create_input(
    halo_len=halo_len,
    image_size=image_size,
    num_spatial_parts=num_spatial_parts,
    rank=rank,
    slice_method=slice_method,
)


b_pt2pt = halo_bench_pt2pt(
    local_rank=rank,
    num_spatial_parts=num_spatial_parts,
    halo_len=halo_len,
    in_channels=args.in_channels,
    out_channels=args.out_channels,
)
b_pt2pt.weight.data.fill_(1.0)
b_pt2pt.bias.data.fill_(1.0)
b_pt2pt.cuda()

for i in range(warmup):
    recv, y = b_pt2pt.run(input_tensor_local)

start_event = torch.cuda.Event(enable_timing=True, blocking=True)
end_event = torch.cuda.Event(enable_timing=True, blocking=True)

start_event.record()

for i in range(iterations):
    recv, y = b_pt2pt.run(input_tensor_local)

output = y
end_event.record()
torch.cuda.synchronize()

t = start_event.elapsed_time(end_event)

print("Rank:" + str(rank) + " Time taken (ms):" + str(t / iterations))

test_output_recv(recv, expected_output_recv, rank)


"""
Sequential processing 
"""
input_seq = np.asarray(
    list(range(0, args.batch_size * args.in_channels * image_size[0] * image_size[1])),
    dtype=np.float32,
)

input_seq.resize(args.batch_size, args.in_channels, image_size[0], image_size[1])
input_tensor_seq = torch.tensor(input_seq, dtype=torch.float, device="cuda")

conv_seq = nn.Conv2d(
    args.in_channels,
    args.out_channels,
    kernel_size=halo_len * 2 + 1,
    stride=1,
    padding=halo_len,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
)

conv_seq.weight.data.fill_(1.0)
conv_seq.bias.data.fill_(1.0)

conv_seq.cuda()

torch.cuda.synchronize()

for i in range(warmup):
    y = conv_seq.forward(input_tensor_seq)

start_event_seq = torch.cuda.Event(enable_timing=True, blocking=True)
end_event_seq = torch.cuda.Event(enable_timing=True, blocking=True)

start_event_seq.record()
for i in range(iterations):
    y = conv_seq.forward(input_tensor_seq)
expected_output = y

end_event_seq.record()
torch.cuda.synchronize()

t = start_event_seq.elapsed_time(end_event_seq)

print("Rank:" + str(rank) + " Time taken Seq (ms):" + str(t / iterations))


test_output(image_size, output, expected_output, rank, size)
