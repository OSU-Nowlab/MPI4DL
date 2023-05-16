import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from collections import OrderedDict
import sys

sys.path.append("../")
from torchgems.spatial_new import conv_spatial, halo_exchange_layer, Pool
import math
from collections import OrderedDict
from typing import Any, TYPE_CHECKING, Iterator, List, Tuple, Union, cast
from torch import Tensor

global_info = {}


def set_basic_informations(local_rank, spatial_size, num_spatial_parts, slice_method):
    global_info["local_rank"] = local_rank
    global_info["spatial_size"] = spatial_size
    global_info["num_spatial_parts"] = num_spatial_parts
    global_info["slice_method"] = slice_method


__all__: List[str] = []


class Operation(nn.Module):
    """Includes the operation name into the representation string for
    debugging.
    """

    def __init__(self, name: str, module: nn.Module):
        super().__init__()
        self.name = name
        self.module = module

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.name}]"

    def forward(self, ENABLE_SPATIAL: bool, *args: Any) -> Any:  # type: ignore
        return self.module(ENABLE_SPATIAL, *args)


class FactorizedReduce(nn.Module):
    def __init__(self, ENABLE_SPATIAL: bool, in_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))

        self.conv1 = nn.Conv2d(
            in_channels, out_channels // 2, kernel_size=1, stride=2, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels, out_channels // 2, kernel_size=1, stride=2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        x = input
        x = self.relu(x)
        # x = torch.cat([self.conv1(x), self.conv2(self.pad(x[:, :, 1:, 1:]))], dim=1)
        x = torch.cat([self.conv1(x), self.conv2(x)], dim=1)
        x = self.bn(x)
        return x


def none(ENABLE_SPATIAL: bool, channels: int, stride: int) -> Operation:
    module: nn.Module
    if stride == 1:
        module = nn.Identity()
    else:
        module = FactorizedReduce(ENABLE_SPATIAL, channels, channels)
    return Operation("none", module)


def avg_pool_3x3_d2(ENABLE_SPATIAL: bool, channels: int, stride: int) -> Operation:
    module = nn.AvgPool2d(3, stride=stride, padding=0, count_include_pad=False)

    return Operation("avg_pool_3x3", module)


def avg_pool_3x3(ENABLE_SPATIAL: bool, channels: int, stride: int) -> Operation:
    if ENABLE_SPATIAL:
        module = Pool(
            local_rank=global_info["local_rank"],
            spatial_size=global_info["spatial_size"],
            num_spatial_parts=global_info["num_spatial_parts"],
            slice_method=global_info["slice_method"],
            operation="AvgPool2d",
            kernel_size=3,
            stride=stride,
            padding=1,
            count_include_pad=False,
        )

    else:
        module = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

    return Operation("avg_pool_3x3", module)


def max_pool_3x3_d2(ENABLE_SPATIAL: bool, channels: int, stride: int) -> Operation:
    module = nn.AvgPool2d(3, stride=stride, padding=0, count_include_pad=False)

    return Operation("max_pool_3x3", module)


def max_pool_3x3(ENABLE_SPATIAL: bool, channels: int, stride: int) -> Operation:
    if ENABLE_SPATIAL:
        module = Pool(
            local_rank=global_info["local_rank"],
            spatial_size=global_info["spatial_size"],
            num_spatial_parts=global_info["num_spatial_parts"],
            slice_method=global_info["slice_method"],
            operation="AvgPool2d",
            kernel_size=3,
            stride=stride,
            padding=1,
            count_include_pad=False,
        )

    else:
        module = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

    return Operation("max_pool_3x3", module)


def max_pool_2x2(ENABLE_SPATIAL: bool, channels: int, stride: int) -> Operation:
    if ENABLE_SPATIAL:
        module = Pool(
            local_rank=global_info["local_rank"],
            spatial_size=global_info["spatial_size"],
            num_spatial_parts=global_info["num_spatial_parts"],
            slice_method=global_info["slice_method"],
            operation="MaxPool2d",
            kernel_size=2,
            stride=stride,
            padding=0,
            count_include_pad=False,
        )

    else:
        module = nn.MaxPool2d(2, stride=stride, padding=0)
    return Operation("max_pool_2x2", module)


def conv_1x7_7x1_d2(ENABLE_SPATIAL: bool, channels: int, stride: int) -> Operation:
    c = channels

    module = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(c, c // 4, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(c // 4),
        nn.ReLU(inplace=False),
        nn.Conv2d(
            c // 4,
            c // 4,
            kernel_size=(1, 7),
            stride=(1, stride),
            padding=(0, 0),
            bias=False,
        ),
        nn.BatchNorm2d(c // 4),
        nn.ReLU(inplace=False),
        nn.Conv2d(
            c // 4,
            c // 4,
            kernel_size=(7, 1),
            stride=(stride, 1),
            padding=(0, 0),
            bias=False,
        ),
        nn.BatchNorm2d(c // 4),
        nn.ReLU(inplace=False),
        nn.Conv2d(c // 4, c, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(c),
    )
    return Operation("conv_1x7_7x1", module)


def conv_1x7_7x1(ENABLE_SPATIAL: bool, channels: int, stride: int) -> Operation:
    c = channels
    if ENABLE_SPATIAL:
        module = nn.Sequential(
            nn.ReLU(inplace=False),
            conv_spatial(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                in_channels=c,
                out_channels=c // 4,
                kernel_size=1,
                stride=1,
                bias=False,
                padding=0,
                slice_method=global_info["slice_method"],
            ),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=False),
            conv_spatial(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                in_channels=c // 4,
                out_channels=c // 4,
                kernel_size=(1, 7),
                stride=(1, stride),
                padding=(0, 3),
                bias=False,
                slice_method=global_info["slice_method"],
            ),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=False),
            conv_spatial(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                in_channels=c // 4,
                out_channels=c // 4,
                kernel_size=(7, 1),
                stride=(stride, 1),
                padding=(3, 0),
                bias=False,
                slice_method=global_info["slice_method"],
            ),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=False),
            conv_spatial(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                in_channels=c // 4,
                out_channels=c,
                kernel_size=1,
                stride=1,
                bias=False,
                padding=0,
                slice_method=global_info["slice_method"],
            ),
            nn.BatchNorm2d(c),
        )

    else:
        module = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c, c // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c // 4,
                c // 4,
                kernel_size=(1, 7),
                stride=(1, stride),
                padding=(0, 3),
                bias=False,
            ),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c // 4,
                c // 4,
                kernel_size=(7, 1),
                stride=(stride, 1),
                padding=(3, 0),
                bias=False,
            ),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(c // 4, c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
        )
    return Operation("conv_1x7_7x1", module)


def conv_1x1(ENABLE_SPATIAL: bool, channels: int, stride: int) -> Operation:
    c = channels
    module = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(c, c, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(c),
    )
    return Operation("conv_1x1", module)


def conv_3x3_d2(ENABLE_SPATIAL: bool, channels: int, stride: int) -> Operation:
    c = channels

    module = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(c, c // 4, kernel_size=1, bias=False),
        nn.BatchNorm2d(c // 4),
        nn.ReLU(inplace=False),
        nn.Conv2d(c // 4, c // 4, kernel_size=3, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(c // 4),
        nn.ReLU(inplace=False),
        nn.Conv2d(c // 4, c, kernel_size=1, bias=False),
        nn.BatchNorm2d(c),
    )
    return Operation("conv_3x3", module)


def conv_3x3(ENABLE_SPATIAL: bool, channels: int, stride: int) -> Operation:
    c = channels
    if ENABLE_SPATIAL:
        module = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c, c // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=False),
            conv_spatial(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                in_channels=c // 4,
                out_channels=c // 4,
                kernel_size=3,
                stride=stride,
                bias=False,
                padding=1,
                slice_method=global_info["slice_method"],
            ),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(c // 4, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
        )

    else:
        module = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c, c // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c // 4, c // 4, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(c // 4, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
        )
    return Operation("conv_3x3", module)


# The genotype for AmoebaNet-D
NORMAL_OPERATIONS = [
    # (0, max_pool_3x3),
    # (0, conv_1x1),
    # (2, none),
    # (2, max_pool_3x3),
    # (0, none),
    # (1, conv_1x7_7x1),
    # (1, conv_1x1),
    # (1, conv_1x7_7x1),
    # (0, avg_pool_3x3),
    # (3, conv_1x1),
    (1, conv_1x1),
    (1, max_pool_3x3),
    (1, none),
    (0, conv_1x7_7x1),
    (0, conv_1x1),
    (0, conv_1x7_7x1),
    (2, max_pool_3x3),
    (2, none),
    (1, avg_pool_3x3),
    (5, conv_1x1),
]

# 0: Skip connection
# 1: output
# 2: 0 with halo exchange 3 (halo_len)
# 3: 1 with halo exhange 2 (halo_len)
# 4: 1 with halo exhange 1 (halo_len)
# 5: output of first two operations


NORMAL_OPERATIONS_D2 = [
    # (0, max_pool_3x3),
    # (0, conv_1x1),
    # (2, none),
    # (2, max_pool_3x3),
    # (0, none),
    # (1, conv_1x7_7x1),
    # (1, conv_1x1),
    # (1, conv_1x7_7x1),
    # (0, avg_pool_3x3),
    # (3, conv_1x1),
    (4, conv_1x1),
    (3, max_pool_3x3_d2),
    (1, none),
    (2, conv_1x7_7x1_d2),
    (0, conv_1x1),
    (2, conv_1x7_7x1_d2),
    (5, max_pool_3x3_d2),
    (5, none),
    (4, avg_pool_3x3_d2),
    (8, conv_1x1),
]

# NORMAL_OPERATIONS_D2 = [
#     # (0, max_pool_3x3),
#     # (0, conv_1x1),
#     # (2, none),
#     # (2, max_pool_3x3),
#     # (0, none),
#     # (1, conv_1x7_7x1),
#     # (1, conv_1x1),
#     # (1, conv_1x7_7x1),
#     # (0, avg_pool_3x3),
#     # (3, conv_1x1),

#     (1, conv_1x1),
#     (4, max_pool_3x3_d2),
#     (1, none),
#     (2, conv_1x7_7x1_d2),
#     (0, conv_1x1),
#     (2, conv_1x7_7x1_d2),
#     (5, max_pool_3x3),
#     (5, none),
#     (4, avg_pool_3x3_d2),
#     (8, conv_1x1),
# ]

# According to the paper for AmoebaNet-D, 'normal_concat' should be [4, 5, 6]
# just like 'reduce_concat'. But 'normal_concat' in the reference AmoebaNet-D
# implementation by TensorFlow is defined as [0, 3, 4, 6], which is different
# with the paper.
#
# For now, we couldn't be sure which is correct. But the GPipe paper seems to
# rely on the setting of TensorFlow's implementation. With this, we can
# reproduce the size of model parameters reported at Table 1 in the paper,
# exactly.
#
# Regularized Evolution for Image Classifier Architecture Search
#   https://arxiv.org/pdf/1802.01548.pdf
#
# GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism
#   https://arxiv.org/pdf/1811.06965.pdf
#
# The AmoebaNet-D implementation by TensorFlow
#   https://github.com/tensorflow/tpu/blob/c753c0a/models/official/amoeba_net
#

NORMAL_CONCAT_D2 = [0, 6, 7, 9]
NORMAL_CONCAT = [0, 3, 4, 6]

REDUCTION_OPERATIONS = [
    (0, max_pool_2x2),
    (0, max_pool_3x3),
    (2, none),
    (1, conv_3x3),
    (2, conv_1x7_7x1),
    (2, max_pool_3x3),
    (3, none),
    (1, max_pool_2x2),
    (2, avg_pool_3x3),
    (3, conv_1x1),
]
REDUCTION_CONCAT = [4, 5, 6]


"""AmoebaNet-D for ImageNet"""


__all__ = ["amoebanetd"]

if TYPE_CHECKING:
    NamedModules = OrderedDict[str, nn.Module]
else:
    NamedModules = OrderedDict


def relu_conv_bn(
    ENABLE_SPATIAL: bool,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    stride: int = 1,
    padding: int = 0,
) -> nn.Module:
    if ENABLE_SPATIAL:
        return nn.Sequential(
            nn.ReLU(inplace=False),
            conv_spatial(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                padding=padding,
                slice_method=global_info["slice_method"],
            ),
            nn.BatchNorm2d(out_channels),
        )

    else:
        return nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )


class Classify(nn.Module):
    def __init__(self, channels_prev: int, num_classes: int):
        super().__init__()
        # self.pool = nn.AvgPool2d(7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(channels_prev, num_classes)

    def forward(self, states: Tuple[Tensor, Tensor]) -> Tensor:  # type: ignore
        x, _ = states
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


class Stem(nn.Module):
    def __init__(self, ENABLE_SPATIAL, channels: int) -> None:
        super().__init__()

        if ENABLE_SPATIAL:
            self.conv = conv_spatial(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                in_channels=3,
                out_channels=channels,
                kernel_size=3,
                stride=2,
                bias=False,
                padding=1,
                slice_method=global_info["slice_method"],
            )

        else:
            self.conv = nn.Conv2d(3, channels, 3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        # self.conv = nn.Conv2d(3, channels, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        x = input
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class Cell_D2(nn.Module):
    def __init__(
        self,
        ENABLE_SPATIAL: bool,
        channels_prev_prev: int,
        channels_prev: int,
        channels: int,
        reduction: bool,
        reduction_prev: bool,
    ) -> None:
        super().__init__()

        self.reduce1 = relu_conv_bn(
            ENABLE_SPATIAL=ENABLE_SPATIAL,
            in_channels=channels_prev,
            out_channels=channels,
        )

        self.reduce2: nn.Module = nn.Identity()
        if reduction_prev:
            self.reduce2 = FactorizedReduce(
                ENABLE_SPATIAL, channels_prev_prev, channels
            )
        elif channels_prev_prev != channels:
            self.reduce2 = relu_conv_bn(
                ENABLE_SPATIAL=ENABLE_SPATIAL,
                in_channels=channels_prev_prev,
                out_channels=channels,
            )

        if reduction:
            self.indices, op_classes = zip(*REDUCTION_OPERATIONS)
            self.concat = REDUCTION_CONCAT
        else:
            self.indices, op_classes = zip(*NORMAL_OPERATIONS_D2)
            self.concat = NORMAL_CONCAT_D2

            self.s3_layer = halo_exchange_layer(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                halo_len=3,
                slice_method=global_info["slice_method"],
            )

            self.s4_layer = halo_exchange_layer(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                halo_len=2,
                slice_method=global_info["slice_method"],
            )

        self.operations = nn.ModuleList()

        for i, op_class in zip(self.indices, op_classes):
            if reduction and i < 2:
                stride = 2
            else:
                stride = 1

            op = op_class(ENABLE_SPATIAL=True, channels=channels, stride=stride)
            self.operations.append(op)

    def extra_repr(self) -> str:
        return f"indices: {self.indices}"

    def forward(
        self,  # type: ignore
        input_or_states: Union[Tensor, Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        if isinstance(input_or_states, tuple):
            s1, s2 = input_or_states
        else:
            s1 = s2 = input_or_states

        skip = s1

        s1 = self.reduce1(s1)
        s2 = self.reduce2(s2)

        s3 = self.s3_layer(s1)
        s4 = self.s4_layer(s2)

        s5 = s4[:, :, 1:-1, 1:-1]

        _states = [s1, s2, s3, s4, s5]

        operations = cast(nn.ModuleList, self.operations)
        indices = cast(List[int], self.indices)

        for i in range(0, len(operations), 2):
            h1 = _states[indices[i]]
            h2 = _states[indices[i + 1]]

            op1 = operations[i]
            op2 = operations[i + 1]

            h1 = op1(h1)
            h2 = op2(h2)

            if i == 6:
                s = h1 + h2[:, :, 1:-1, 1:-1]
            else:
                s = h1 + h2
            _states.append(s)

        return torch.cat([_states[i] for i in self.concat], dim=1), skip


class Cell(nn.Module):
    def __init__(
        self,
        ENABLE_SPATIAL: bool,
        channels_prev_prev: int,
        channels_prev: int,
        channels: int,
        reduction: bool,
        reduction_prev: bool,
    ) -> None:
        super().__init__()

        self.reduce1 = relu_conv_bn(
            ENABLE_SPATIAL=ENABLE_SPATIAL,
            in_channels=channels_prev,
            out_channels=channels,
        )

        self.reduce2: nn.Module = nn.Identity()
        if reduction_prev:
            self.reduce2 = FactorizedReduce(
                ENABLE_SPATIAL, channels_prev_prev, channels
            )
        elif channels_prev_prev != channels:
            self.reduce2 = relu_conv_bn(
                ENABLE_SPATIAL=ENABLE_SPATIAL,
                in_channels=channels_prev_prev,
                out_channels=channels,
            )

        if reduction:
            self.indices, op_classes = zip(*REDUCTION_OPERATIONS)
            self.concat = REDUCTION_CONCAT
        else:
            self.indices, op_classes = zip(*NORMAL_OPERATIONS)
            self.concat = NORMAL_CONCAT

        self.operations = nn.ModuleList()

        for i, op_class in zip(self.indices, op_classes):
            if reduction and i < 2:
                stride = 2
            else:
                stride = 1

            op = op_class(ENABLE_SPATIAL, channels, stride)
            self.operations.append(op)

    def extra_repr(self) -> str:
        return f"indices: {self.indices}"

    def forward(
        self,  # type: ignore
        input_or_states: Union[Tensor, Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        if isinstance(input_or_states, tuple):
            s1, s2 = input_or_states
        else:
            s1 = s2 = input_or_states

        skip = s1

        s1 = self.reduce1(s1)
        s2 = self.reduce2(s2)

        _states = [s1, s2]

        operations = cast(nn.ModuleList, self.operations)
        indices = cast(List[int], self.indices)

        for i in range(0, len(operations), 2):
            h1 = _states[indices[i]]
            h2 = _states[indices[i + 1]]

            op1 = operations[i]
            op2 = operations[i + 1]

            h1 = op1(h1)
            h2 = op2(h2)

            s = h1 + h2
            _states.append(s)

        return torch.cat([_states[i] for i in self.concat], dim=1), skip


def amoebanetd(
    num_classes: int = 10,
    num_layers: int = 4,
    num_filters: int = 512,
) -> nn.Sequential:
    """Builds an AmoebaNet-D model for ImageNet."""
    layers: NamedModules = OrderedDict()

    ENABLE_SPATIAL = False

    assert num_layers % 3 == 0
    repeat_normal_cells = num_layers // 3

    channels = num_filters // 4
    channels_prev_prev = channels_prev = channels
    reduction_prev = False

    def make_cells(
        ENABLE_SPATIAL: bool, reduction: bool, channels_scale: int, repeat: int
    ) -> Iterator[Cell]:
        nonlocal channels_prev_prev
        nonlocal channels_prev
        nonlocal channels
        nonlocal reduction_prev

        channels *= channels_scale

        for i in range(repeat):
            cell = Cell(
                ENABLE_SPATIAL,
                channels_prev_prev,
                channels_prev,
                channels,
                reduction,
                reduction_prev,
            )

            channels_prev_prev = channels_prev
            channels_prev = channels * len(cell.concat)
            reduction_prev = reduction

            yield cell

    def reduction_cell(ENABLE_SPATIAL) -> Cell:
        return next(
            make_cells(ENABLE_SPATIAL, reduction=True, channels_scale=2, repeat=1)
        )

    def normal_cells(ENABLE_SPATIAL) -> Iterator[Tuple[int, Cell]]:
        return enumerate(
            make_cells(
                ENABLE_SPATIAL,
                reduction=False,
                channels_scale=1,
                repeat=repeat_normal_cells,
            )
        )

    # Stem for ImageNet
    layers["stem1"] = Stem(ENABLE_SPATIAL, channels)
    layers["stem2"] = reduction_cell(ENABLE_SPATIAL)
    layers["stem3"] = reduction_cell(ENABLE_SPATIAL)

    # AmoebaNet cells
    layers.update(
        (f"cell1_normal{i+1}", cell) for i, cell in normal_cells(ENABLE_SPATIAL)
    )
    layers["cell2_reduction"] = reduction_cell(ENABLE_SPATIAL)

    print("Layers:", len(layers))

    layers.update(
        (f"cell3_normal{i+1}", cell) for i, cell in normal_cells(ENABLE_SPATIAL)
    )
    layers["cell4_reduction"] = reduction_cell(ENABLE_SPATIAL)
    layers.update(
        (f"cell5_normal{i+1}", cell) for i, cell in normal_cells(ENABLE_SPATIAL)
    )

    # Finally, classifier
    layers["classify"] = Classify(channels_prev, num_classes)

    print("Layers:", len(layers))
    return nn.Sequential(layers)


def amoebanetd_spatial(
    local_rank,
    spatial_size,
    num_spatial_parts,
    mp_size,
    balance=None,
    slice_method="square",
    num_classes: int = 10,
    num_layers: int = 4,
    num_filters: int = 512,
) -> nn.Sequential:
    """Builds an AmoebaNet-D model for ImageNet."""

    set_basic_informations(local_rank, spatial_size, num_spatial_parts, slice_method)

    layers: NamedModules = OrderedDict()

    ENABLE_SPATIAL = True

    assert num_layers % 3 == 0
    repeat_normal_cells = num_layers // 3
    predicted_num_layers = repeat_normal_cells * 3 + 6

    _, end_layer = get_start_end_layer_index(
        predicted_num_layers, balance, mp_size, local_rank=0
    )

    channels = num_filters // 4
    channels_prev_prev = channels_prev = channels
    reduction_prev = False

    layers_processed = 0

    def make_cells(reduction: bool, channels_scale: int, repeat: int) -> Iterator[Cell]:
        nonlocal channels_prev_prev
        nonlocal channels_prev
        nonlocal channels
        nonlocal reduction_prev
        nonlocal layers_processed
        nonlocal ENABLE_SPATIAL

        channels *= channels_scale

        for i in range(repeat):
            if layers_processed >= end_layer:
                ENABLE_SPATIAL = False
            layers_processed = layers_processed + 1

            if reduction == False and ENABLE_SPATIAL == True:
                cell = Cell_D2(
                    ENABLE_SPATIAL,
                    channels_prev_prev,
                    channels_prev,
                    channels,
                    reduction,
                    reduction_prev,
                )

            else:
                cell = Cell(
                    ENABLE_SPATIAL,
                    channels_prev_prev,
                    channels_prev,
                    channels,
                    reduction,
                    reduction_prev,
                )

            channels_prev_prev = channels_prev
            channels_prev = channels * len(cell.concat)
            reduction_prev = reduction

            yield cell

    def reduction_cell() -> Cell:
        return next(make_cells(reduction=True, channels_scale=2, repeat=1))

    def normal_cells() -> Iterator[Tuple[int, Cell]]:
        return enumerate(
            make_cells(reduction=False, channels_scale=1, repeat=repeat_normal_cells)
        )

    # Stem for ImageNet
    layers["stem1"] = Stem(ENABLE_SPATIAL, channels)
    layers_processed = layers_processed + 1
    layers["stem2"] = reduction_cell()
    layers_processed = layers_processed + 1
    layers["stem3"] = reduction_cell()
    layers_processed = layers_processed + 1

    assert end_layer > 3, "There should be atleast 3 layers in "

    # AmoebaNet cells
    layers.update((f"cell1_normal{i+1}", cell) for i, cell in normal_cells())
    layers["cell2_reduction"] = reduction_cell()

    print("Layers")
    layers.update((f"cell3_normal{i+1}", cell) for i, cell in normal_cells())
    layers["cell4_reduction"] = reduction_cell()
    layers.update((f"cell5_normal{i+1}", cell) for i, cell in normal_cells())

    # Finally, classifier
    layers["classify"] = Classify(channels_prev, num_classes)

    return nn.Sequential(layers)


def get_start_end_layer_index(num_layers, balance, mp_size, local_rank=0):
    # return the index of start and end layer for the model
    # based on the size of model parallelism and local rank of the process
    # it can be modified by giving balance parameter
    if balance == None:
        part_layer = int(num_layers / mp_size)
        start_layer = local_rank * part_layer

        if local_rank != mp_size - 1:
            end_layer = (local_rank + 1) * part_layer
        else:
            end_layer = num_layers
    else:
        assert sum(balance) == num_layers, "balance and number of layers differs"

        if local_rank == 0:
            start_layer = 0
            end_layer = balance[0]
            part_layer = balance[0]
        else:
            start_layer = sum(balance[:local_rank])
            end_layer = sum(balance[: local_rank + 1])
            part_layer = end_layer - start_layer

    return start_layer, end_layer
