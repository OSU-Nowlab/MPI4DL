import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from collections import OrderedDict
import sys

sys.path.append("../")
from torchgems.spatial_new import conv_spatial, halo_exchange_layer
import math

global_info = {}


def set_basic_informations(local_rank, spatial_size, num_spatial_parts, slice_method):
    global_info["local_rank"] = local_rank
    global_info["spatial_size"] = spatial_size
    global_info["num_spatial_parts"] = num_spatial_parts
    global_info["slice_method"] = slice_method


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        padding=0,
        ENABLE_SPATIAL=False,
    ):
        super(BasicConv2d, self).__init__()
        if ENABLE_SPATIAL:
            self.conv = conv_spatial(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                slice_method=global_info["slice_method"],
            )
        else:
            self.conv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):
    def __init__(self, ENABLE_SPATIAL=False):
        super(Mixed_3a, self).__init__()

        if ENABLE_SPATIAL:
            self.halo_len_layer = halo_exchange_layer(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                halo_len=1,
                slice_method=global_info["slice_method"],
            )
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=0)
        else:
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv = BasicConv2d(
            64, 96, kernel_size=3, stride=2, ENABLE_SPATIAL=ENABLE_SPATIAL, padding=1
        )

        self.ENABLE_SPATIAL = ENABLE_SPATIAL

    def forward(self, x):
        if self.ENABLE_SPATIAL:
            x_halo = self.halo_len_layer(x)
            x0 = self.maxpool(x_halo)
        else:
            x0 = self.maxpool(x)

        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):
    def __init__(self, ENABLE_SPATIAL=False):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(
                160, 64, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
            ),
            BasicConv2d(
                64,
                96,
                kernel_size=3,
                stride=1,
                ENABLE_SPATIAL=ENABLE_SPATIAL,
                padding=1,
            ),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(
                160, 64, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
            ),
            BasicConv2d(
                64,
                64,
                kernel_size=(1, 7),
                stride=1,
                padding=(0, 3),
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
            BasicConv2d(
                64,
                64,
                kernel_size=(7, 1),
                stride=1,
                padding=(3, 0),
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
            BasicConv2d(
                64,
                96,
                kernel_size=(3, 3),
                stride=1,
                ENABLE_SPATIAL=ENABLE_SPATIAL,
                padding=1,
            ),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):
    def __init__(self, ENABLE_SPATIAL=False):
        super(Mixed_5a, self).__init__()

        if ENABLE_SPATIAL:
            self.halo_len_layer = halo_exchange_layer(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                halo_len=1,
                slice_method=global_info["slice_method"],
            )
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=0)
        else:
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv = BasicConv2d(
            192, 192, kernel_size=3, stride=2, ENABLE_SPATIAL=ENABLE_SPATIAL, padding=1
        )
        self.ENABLE_SPATIAL = ENABLE_SPATIAL

    def forward(self, x):
        x0 = self.conv(x)
        if self.ENABLE_SPATIAL:
            x_halo = self.halo_len_layer(x)
            x1 = self.maxpool(x_halo)
        else:
            x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class branch_pooling(nn.Module):
    def __init__(self, ENABLE_SPATIAL=False):
        super(branch_pooling, self).__init__()

        if ENABLE_SPATIAL:
            self.halo_len_layer = halo_exchange_layer(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                halo_len=1,
                slice_method=global_info["slice_method"],
            )
            self.pool = nn.AvgPool2d(3, stride=1, padding=0, count_include_pad=False)
        else:
            self.pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.ENABLE_SPATIAL = ENABLE_SPATIAL

    def forward(self, x):
        if self.ENABLE_SPATIAL:
            x = self.halo_len_layer(x)
        x = self.pool(x)
        return x


class Inception_A(nn.Module):
    def __init__(self, ENABLE_SPATIAL=False):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(
            384, 96, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(
                384, 64, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
            ),
            BasicConv2d(
                64,
                96,
                kernel_size=3,
                stride=1,
                padding=1,
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(
                384, 64, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
            ),
            BasicConv2d(
                64,
                96,
                kernel_size=3,
                stride=1,
                padding=1,
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
            BasicConv2d(
                96,
                96,
                kernel_size=3,
                stride=1,
                padding=1,
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
        )

        self.branch3 = nn.Sequential(
            branch_pooling(),
            BasicConv2d(
                384, 96, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
            ),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):
    def __init__(self, ENABLE_SPATIAL=False):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(
            384, 384, kernel_size=3, stride=2, ENABLE_SPATIAL=ENABLE_SPATIAL, padding=1
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(
                384, 192, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
            ),
            BasicConv2d(
                192,
                224,
                kernel_size=3,
                stride=1,
                padding=1,
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
            BasicConv2d(
                224,
                256,
                kernel_size=3,
                stride=2,
                ENABLE_SPATIAL=ENABLE_SPATIAL,
                padding=1,
            ),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):
    def __init__(self, ENABLE_SPATIAL=False):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(
            1024, 384, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(
                1024, 192, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
            ),
            BasicConv2d(
                192,
                224,
                kernel_size=(1, 7),
                stride=1,
                padding=(0, 3),
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
            BasicConv2d(
                224,
                256,
                kernel_size=(7, 1),
                stride=1,
                padding=(3, 0),
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(
                1024, 192, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
            ),
            BasicConv2d(
                192,
                192,
                kernel_size=(7, 1),
                stride=1,
                padding=(3, 0),
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
            BasicConv2d(
                192,
                224,
                kernel_size=(1, 7),
                stride=1,
                padding=(0, 3),
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
            BasicConv2d(
                224,
                224,
                kernel_size=(7, 1),
                stride=1,
                padding=(3, 0),
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
            BasicConv2d(
                224,
                256,
                kernel_size=(1, 7),
                stride=1,
                padding=(0, 3),
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
        )

        self.branch3 = nn.Sequential(
            branch_pooling(),
            BasicConv2d(
                1024, 128, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
            ),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):
    def __init__(self, ENABLE_SPATIAL=False):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(
                1024, 192, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
            ),
            BasicConv2d(
                192,
                192,
                kernel_size=3,
                stride=2,
                padding=1,
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(
                1024, 256, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
            ),
            BasicConv2d(
                256,
                256,
                kernel_size=(1, 7),
                stride=1,
                padding=(0, 3),
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
            BasicConv2d(
                256,
                320,
                kernel_size=(7, 1),
                stride=1,
                padding=(3, 0),
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
            BasicConv2d(
                320,
                320,
                kernel_size=3,
                stride=2,
                padding=1,
                ENABLE_SPATIAL=ENABLE_SPATIAL,
            ),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):
    def __init__(self, ENABLE_SPATIAL=False):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(
            1536, 256, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
        )

        self.branch1_0 = BasicConv2d(
            1536, 384, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
        )
        self.branch1_1a = BasicConv2d(
            384,
            256,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            ENABLE_SPATIAL=ENABLE_SPATIAL,
        )
        self.branch1_1b = BasicConv2d(
            384,
            256,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            ENABLE_SPATIAL=ENABLE_SPATIAL,
        )

        self.branch2_0 = BasicConv2d(
            1536, 384, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
        )
        self.branch2_1 = BasicConv2d(
            384,
            448,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            ENABLE_SPATIAL=ENABLE_SPATIAL,
        )
        self.branch2_2 = BasicConv2d(
            448,
            512,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            ENABLE_SPATIAL=ENABLE_SPATIAL,
        )
        self.branch2_3a = BasicConv2d(
            512,
            256,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            ENABLE_SPATIAL=ENABLE_SPATIAL,
        )
        self.branch2_3b = BasicConv2d(
            512,
            256,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            ENABLE_SPATIAL=ENABLE_SPATIAL,
        )

        self.branch3 = nn.Sequential(
            branch_pooling(),
            BasicConv2d(
                1536, 256, kernel_size=1, stride=1, ENABLE_SPATIAL=ENABLE_SPATIAL
            ),
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class end_part(nn.Module):
    def __init__(self, image_size, num_classes, ENABLE_SPATIAL=False):
        super(end_part, self).__init__()

        self.padding = 3

        self.pool = nn.AvgPool2d(8, padding=self.padding, count_include_pad=False)

        h_in = math.floor(image_size / 32)
        h_out = math.floor((h_in + 2 * self.padding - 8) / 8 + 1)

        self.classif = nn.Linear(1536 * h_out * h_out, num_classes)

        self.ENABLE_SPATIAL = ENABLE_SPATIAL

        if ENABLE_SPATIAL:
            self.halo_len_layer = halo_exchange_layer(
                local_rank=global_info["local_rank"],
                spatial_size=global_info["spatial_size"],
                num_spatial_parts=global_info["num_spatial_parts"],
                halo_len=1,
                slice_method=global_info["slice_method"],
            )

    def forward(self, x):
        if self.ENABLE_SPATIAL:
            x = self.halo_len_layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        out = self.classif(x)

        return out


def get_inceptionv4(image_size, num_classes=1001):
    model = nn.Sequential(
        BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1),
        BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
        Mixed_3a(),
        Mixed_4a(),
        Mixed_5a(),
        Inception_A(),
        Inception_A(),
        Inception_A(),
        Inception_A(),
        Reduction_A(),  # Mixed_6a
        Inception_B(),
        Inception_B(),
        Inception_B(),
        Inception_B(),
        Inception_B(),
        Inception_B(),
        Inception_B(),
        Reduction_B(),  # Mixed_7a
        Inception_C(),
        Inception_C(),
        Inception_C(),
        end_part(image_size, num_classes),
    )
    return model


def get_inceptionv4_spatial(
    image_size, num_classes, local_rank, spatial_size, num_spatial_parts
):
    set_basic_informations(local_rank, spatial_size, num_spatial_parts, "square")

    model = nn.Sequential(
        BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1, ENABLE_SPATIAL=True),
        BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1, ENABLE_SPATIAL=True),
        BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1, ENABLE_SPATIAL=True),
        Mixed_3a(ENABLE_SPATIAL=True),
        Mixed_4a(ENABLE_SPATIAL=True),
        Mixed_5a(ENABLE_SPATIAL=True),
        Inception_A(ENABLE_SPATIAL=True),
        Inception_A(ENABLE_SPATIAL=True),
        Inception_A(ENABLE_SPATIAL=True),
        Inception_A(ENABLE_SPATIAL=True),
        Reduction_A(ENABLE_SPATIAL=True),  # Mixed_6a
        Inception_B(ENABLE_SPATIAL=True),
        Inception_B(ENABLE_SPATIAL=True),
        Inception_B(ENABLE_SPATIAL=True),
        Inception_B(ENABLE_SPATIAL=True),
        Inception_B(ENABLE_SPATIAL=True),
        Inception_B(),
        Inception_B(),
        Reduction_B(),  # Mixed_7a
        Inception_C(),
        Inception_C(),
        Inception_C(),
        end_part(image_size, num_classes),
    )
    return model


class InceptionV4(nn.Module):
    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2, ENABLE_SPATIAL=True),
            BasicConv2d(32, 32, kernel_size=3, stride=1, ENABLE_SPATIAL=True),
            BasicConv2d(
                32, 64, kernel_size=3, stride=1, padding=1, ENABLE_SPATIAL=True
            ),
            Mixed_3a(ENABLE_SPATIAL=True),
            Mixed_4a(ENABLE_SPATIAL=True),
            Mixed_5a(ENABLE_SPATIAL=True),
            Inception_A(ENABLE_SPATIAL=True),
            Inception_A(ENABLE_SPATIAL=True),
            Inception_A(ENABLE_SPATIAL=True),
            Inception_A(ENABLE_SPATIAL=True),
            Reduction_A(ENABLE_SPATIAL=True),  # Mixed_6a
            Inception_B(ENABLE_SPATIAL=True),
            Inception_B(ENABLE_SPATIAL=True),
            Inception_B(ENABLE_SPATIAL=True),
            Inception_B(ENABLE_SPATIAL=True),
            Inception_B(ENABLE_SPATIAL=True),
            Inception_B(),
            Inception_B(),
            Reduction_B(),  # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C(),
            end_part(),
        )
        self.classif = nn.Linear(1536, num_classes)

    def forward(self, x):
        return self
