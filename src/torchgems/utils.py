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


def isPowerTwo(num):
    return not (num & (num - 1))


# Get the depth of ResNet model based on version and number of ResNet Blocks
# This parameter will used for ResNet model architecture.
def get_depth(version, n):
    if version == 1:
        return n * 6 + 2
    elif version == 2:
        return n * 9 + 2


def print_model_size(model, rank, inverse):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"model size at rank {rank} Inverse {inverse} : {size_all_mb:.3f}MB")


def get_gpu_memory(rank):
    import subprocess as sp

    output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]
    COMMAND = "nvidia-smi --query-gpu=memory.total --format=csv"
    try:
        memory_total_info = output_to_list(
            sp.check_output(COMMAND.split(), stderr=sp.STDOUT)
        )[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )
    print(f"***********Total GPU Memory on Rank {rank} : {memory_total_info}*******")


def load_fake_dataset(batch_size, channel, image_size1, image_size2):
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = torchvision.datasets.FakeData(
        size=10 * batch_size,
        image_size=(3, image_size1, image_size2),
        num_classes=10,
        transform=transform,
        target_transform=None,
        random_offset=0,
    )
    dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return dataloader
