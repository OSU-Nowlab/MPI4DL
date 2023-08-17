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
import torch.nn.functional as F
from collections import OrderedDict


class resnet_layer(nn.Module):
    def __init__(
        self,
        in_num_filters,
        num_filters=16,
        kernel_size=3,
        strides=1,
        activation="relu",
        batch_normalization=True,
        conv_first=True,
    ):
        super(resnet_layer, self).__init__()
        self.conv_first = conv_first
        self.activation = activation
        padding = int((kernel_size - 1) / 2)
        self.batch_normalization = batch_normalization
        self.conv1 = nn.Conv2d(
            in_channels=in_num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
        )
        self.batch_first = nn.BatchNorm2d(
            num_features=in_num_filters,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.batch_last = nn.BatchNorm2d(
            num_features=num_filters,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.act = nn.ReLU()

    def forward(self, x):
        if self.conv_first:
            x = self.conv1(x)
            if self.batch_normalization:
                x = self.batch_last(x)
            if self.activation is not None:
                x = self.act(x)
        else:
            if self.batch_normalization:
                x = self.batch_first(x)
            if self.activation is not None:
                x = self.act(x)
            x = self.conv1(x)

        # x = self.fc3(x)
        return x


class make_cell_v1(nn.Module):
    def __init__(self, stack, resblock, strides, in_filters, out_filters):
        super(make_cell_v1, self).__init__()
        self.r1 = resnet_layer(
            in_num_filters=in_filters, num_filters=out_filters, strides=strides
        )
        self.r2 = resnet_layer(
            in_num_filters=out_filters,
            num_filters=out_filters,
            strides=1,
            activation=None,
        )
        self.resblock = resblock
        self.stack = stack
        if resblock == 0 and stack > 0:
            self.r3 = resnet_layer(
                in_num_filters=in_filters,
                num_filters=out_filters,
                strides=strides,
                activation=None,
                batch_normalization=False,
                kernel_size=1,
            )

    def forward(self, x):
        y = x
        y = self.r1(y)
        y = self.r2(y)
        if self.resblock == 0 and self.stack > 0:
            x = self.r3(x)

        x = x + y
        x = F.relu(x)
        return x


class end_part_v1(nn.Module):
    def __init__(self, kernel_size, batch_size, num_filters, image_size, num_classes):
        super(end_part_v1, self).__init__()
        self.batch_size = batch_size
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=None,
            padding=0,
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None,
        )
        self.flatten_size = (
            num_filters
            * int(image_size / (4 * kernel_size))
            * int(image_size / (4 * kernel_size))
        )
        self.fc1 = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        x = self.pool(x)

        x = x.view(-1, self.flatten_size)
        x = F.softmax(self.fc1(x))

        return x


def get_resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")

    layers = OrderedDict()
    name = 1
    in_filters = 3
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    layers[str(name)] = resnet_layer(in_num_filters=in_filters)
    name += 1

    in_filters = num_filters
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2
            layers[str(name)] = make_cell_v1(
                stack, res_block, strides, in_filters, num_filters
            )
            name += 1
            in_filters = num_filters
        num_filters *= 2

    layers[str(name)] = end_part_v1(
        kernel_size=8,
        batch_size=input_shape[0],
        num_filters=int(num_filters / 2),
        image_size=input_shape[2],
        num_classes=num_classes,
    )
    return nn.Sequential(layers)


class make_cell_v2(nn.Module):
    def __init__(
        self,
        resblock,
        strides,
        in_filters,
        out_filters1,
        out_filters2,
        activation,
        batch_normalization,
    ):
        super(make_cell_v2, self).__init__()
        self.r1 = resnet_layer(
            in_num_filters=in_filters,
            num_filters=out_filters1,
            strides=strides,
            activation=activation,
            batch_normalization=batch_normalization,
            conv_first=False,
        )
        self.r2 = resnet_layer(
            in_num_filters=out_filters1, num_filters=out_filters1, conv_first=False
        )
        self.r3 = resnet_layer(
            in_num_filters=out_filters1,
            num_filters=out_filters2,
            kernel_size=1,
            conv_first=False,
        )
        self.resblock = resblock
        if resblock == 0:
            self.r4 = resnet_layer(
                in_num_filters=in_filters,
                num_filters=out_filters2,
                strides=strides,
                activation=None,
                batch_normalization=False,
                kernel_size=1,
            )

    def forward(self, x):
        y = x
        y = self.r1(y)
        y = self.r2(y)
        y = self.r3(y)
        if self.resblock == 0:
            x = self.r4(x)

        x = x + y
        # x = F.relu(x)
        return x


class end_part_v2(nn.Module):
    def __init__(self, kernel_size, batch_size, num_filters, image_size, num_classes):
        super(end_part_v2, self).__init__()
        self.batch_size = batch_size
        self.batch_last = nn.BatchNorm2d(
            num_features=num_filters,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=None,
            padding=0,
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None,
        )
        self.flatten_size = (
            num_filters
            * int(image_size / (4 * kernel_size))
            * int(image_size / (4 * kernel_size))
        )
        self.fc1 = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        x = F.relu(self.batch_last(x))
        x = self.pool(x)

        x = x.view(-1, self.flatten_size)
        x = F.softmax(self.fc1(x))

        return x


def get_resnet_v2(input_shape, depth, num_classes=10):
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")

    layers = OrderedDict()
    name = 1
    in_filters = 3
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    # inputs = Input(shape=input_shape)
    layers[str(name)] = resnet_layer(in_num_filters=in_filters, conv_first=True)
    name += 1
    # in_filters = num_filters

    in_filters = num_filters_in
    # return nn.Sequential(layers)
    for stage in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            activation = "relu"
            batch_normalization = True

            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            layers[str(name)] = make_cell_v2(
                res_block,
                strides,
                in_filters,
                num_filters_in,
                num_filters_out,
                activation,
                batch_normalization,
            )
            name += 1
            in_filters = num_filters_out
        num_filters_in = num_filters_out

    layers[str(name)] = end_part_v2(
        kernel_size=8,
        batch_size=input_shape[0],
        num_filters=int(num_filters_in),
        image_size=input_shape[2],
        num_classes=num_classes,
    )
    return nn.Sequential(layers)


# model = get_resnet_v1((16,3,32,32),20)
