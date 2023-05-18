import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchgems.spatial_new import conv_spatial


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


class resnet_layer_spatial(nn.Module):
    def __init__(
        self,
        local_rank,
        spatial_size,
        num_spatial_parts,
        in_num_filters,
        num_filters=16,
        kernel_size=3,
        strides=1,
        activation="relu",
        batch_normalization=True,
        conv_first=True,
        slice_method="square",
    ):
        super(resnet_layer_spatial, self).__init__()
        self.conv_first = conv_first
        self.activation = activation
        padding = int((kernel_size - 1) / 2)
        self.batch_normalization = batch_normalization
        self.conv1 = conv_spatial(
            local_rank=local_rank,
            spatial_size=spatial_size,
            num_spatial_parts=num_spatial_parts,
            in_channels=in_num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            slice_method=slice_method,
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


class make_cell_v1_spatial(nn.Module):
    def __init__(
        self,
        local_rank,
        spatial_size,
        num_spatial_parts,
        stack,
        resblock,
        strides,
        in_filters,
        out_filters,
    ):
        super(make_cell_v1_spatial, self).__init__()
        self.r1 = resnet_layer_spatial(
            local_rank,
            spatial_size,
            num_spatial_parts,
            in_num_filters=in_filters,
            num_filters=out_filters,
            strides=strides,
        )
        self.r2 = resnet_layer_spatial(
            local_rank,
            spatial_size,
            num_spatial_parts,
            in_num_filters=out_filters,
            num_filters=out_filters,
            strides=1,
            activation=None,
        )
        self.resblock = resblock
        self.stack = stack
        if resblock == 0 and stack > 0:
            self.r3 = resnet_layer_spatial(
                local_rank,
                spatial_size,
                num_spatial_parts,
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
    def __init__(self, kernel_size, batch_size, num_filters, image_size):
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
        # print("flatten_size",self.flatten_size,image_size,kernel_size,num_filters, int(image_size/(4*kernel_size)))
        self.fc1 = nn.Linear(self.flatten_size, 10)

    def forward(self, x):
        x = self.pool(x)

        x = x.view(-1, self.flatten_size)
        x = F.softmax(self.fc1(x))

        return x


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


def get_resnet_v1(
    input_shape,
    depth,
    local_rank,
    mp_size,
    spatial_size=1,
    num_spatial_parts=4,
    balance=None,
    num_classes=10,
):
    if (depth - 2) % 6 != 0:
        raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")

    layers = OrderedDict()
    name = 0
    in_filters = 3
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    num_layers = int((depth / 2) + 1)

    _, end_layer = get_start_end_layer_index(
        num_layers, balance, mp_size, local_rank=spatial_size - 1
    )
    print("end_layer:", end_layer)

    # inputs = Input(shape=input_shape)
    layers[str(name)] = resnet_layer_spatial(
        local_rank, spatial_size, num_spatial_parts, in_num_filters=in_filters
    )

    name += 1
    # in_filters = num_filters

    in_filters = num_filters
    # return nn.Sequential(layers)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2

            if name >= end_layer:
                layers[str(name)] = make_cell_v1(
                    stack, res_block, strides, in_filters, num_filters
                )
            else:
                layers[str(name)] = make_cell_v1_spatial(
                    local_rank,
                    spatial_size,
                    num_spatial_parts,
                    stack,
                    res_block,
                    strides,
                    in_filters,
                    num_filters,
                )
            name += 1
            in_filters = num_filters
        num_filters *= 2

    layers[str(name)] = end_part_v1(
        kernel_size=8,
        batch_size=input_shape[0],
        num_filters=int(num_filters / 2),
        image_size=input_shape[2],
    )
    return nn.Sequential(layers)


class make_cell_v2_spatial(nn.Module):
    def __init__(
        self,
        local_rank,
        spatial_size,
        num_spatial_parts,
        resblock,
        strides,
        in_filters,
        out_filters1,
        out_filters2,
        activation,
        batch_normalization,
        slice_method="square",
    ):
        super(make_cell_v2_spatial, self).__init__()
        self.r1 = resnet_layer_spatial(
            local_rank,
            spatial_size,
            num_spatial_parts,
            in_num_filters=in_filters,
            num_filters=out_filters1,
            strides=strides,
            activation=activation,
            batch_normalization=batch_normalization,
            conv_first=False,
            slice_method=slice_method,
        )
        self.r2 = resnet_layer_spatial(
            local_rank,
            spatial_size,
            num_spatial_parts,
            in_num_filters=out_filters1,
            num_filters=out_filters1,
            conv_first=False,
            slice_method=slice_method,
        )
        self.r3 = resnet_layer_spatial(
            local_rank,
            spatial_size,
            num_spatial_parts,
            in_num_filters=out_filters1,
            num_filters=out_filters2,
            kernel_size=1,
            conv_first=False,
            slice_method=slice_method,
        )
        self.resblock = resblock
        if resblock == 0:
            self.r4 = resnet_layer_spatial(
                local_rank,
                spatial_size,
                num_spatial_parts,
                in_num_filters=in_filters,
                num_filters=out_filters2,
                strides=strides,
                activation=None,
                batch_normalization=False,
                kernel_size=1,
                slice_method=slice_method,
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

        # Check for vertical and horzontal slicing
        shapes = y.shape
        if shapes[2] != shapes[3]:
            print("ERROR: YOU ARE IN TROUBLE (SHAPES ARE UNEQUAL)")

        y = self.r2(y)
        y = self.r3(y)
        if self.resblock == 0:
            x = self.r4(x)

        x = x + y
        # x = F.relu(x)
        return x


class end_part_v2(nn.Module):
    def __init__(self, kernel_size, batch_size, num_filters, image_size):
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
        self.fc1 = nn.Linear(self.flatten_size, 10)

    def forward(self, x):
        x = F.relu(self.batch_last(x))
        x = self.pool(x)

        x = x.view(-1, self.flatten_size)
        x = F.softmax(self.fc1(x))

        return x


def get_resnet_v2(
    input_shape,
    depth,
    local_rank,
    mp_size,
    spatial_size=1,
    num_spatial_parts=4,
    balance=None,
    num_classes=10,
    fused_layers=1,
    slice_method="square",
):
    # fused layers parameter does not mean anything here , only for D2 design
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")

    layers = OrderedDict()
    name = 0
    in_filters = 3
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    num_layers = num_res_blocks * 3 + 2

    _, end_layer = get_start_end_layer_index(
        num_layers, balance, mp_size, local_rank=spatial_size - 1
    )
    print("end_layer:", end_layer)

    # inputs = Input(shape=input_shape)
    # layers[str(name)] = resnet_layer(in_num_filters=in_filters,conv_first=True)
    layers[str(name)] = resnet_layer_spatial(
        local_rank,
        spatial_size,
        num_spatial_parts,
        in_num_filters=in_filters,
        slice_method=slice_method,
    )
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
            if name >= end_layer:
                layers[str(name)] = make_cell_v2(
                    res_block,
                    strides,
                    in_filters,
                    num_filters_in,
                    num_filters_out,
                    activation,
                    batch_normalization,
                )
            else:
                layers[str(name)] = make_cell_v2_spatial(
                    local_rank,
                    spatial_size,
                    num_spatial_parts,
                    res_block,
                    strides,
                    in_filters,
                    num_filters_in,
                    num_filters_out,
                    activation,
                    batch_normalization,
                    slice_method=slice_method,
                )
            name += 1
            in_filters = num_filters_out
        num_filters_in = num_filters_out

    layers[str(name)] = end_part_v2(
        kernel_size=8,
        batch_size=input_shape[0],
        num_filters=int(num_filters_in),
        image_size=input_shape[2],
    )
    return nn.Sequential(layers)


# model = get_resnet_v1((16,3,32,32),20)
