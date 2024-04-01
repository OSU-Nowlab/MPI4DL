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

import os
import torch
import subprocess

def isPowerTwo(num):
    return not (num & (num - 1))


# Get the depth of ResNet model based on version and number of ResNet Blocks
# This parameter will used for ResNet model architecture.
def get_depth(version, n):
    if version == 1:
        return n * 6 + 2
    elif version == 2:
        return n * 9 + 2

def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default

def get_xdist_worker_id():
    xdist_worker = os.environ.get('PYTEST_XDIST_WORKER', None)
    if xdist_worker is not None:
        xdist_worker_id = xdist_worker.replace('gw', '')
        return int(xdist_worker_id)
    return None

def set_accelerator_visible():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    xdist_worker_id = get_xdist_worker_id()
    if xdist_worker_id is None:
        xdist_worker_id = 0
    if cuda_visible is None:
        # CUDA_VISIBLE_DEVICES is not set, discover it using accelerator specific command instead
        nvidia_smi = subprocess.check_output(['nvidia-smi', '--list-gpus'])
        num_accelerators = len(nvidia_smi.decode('utf-8').strip().split('\n'))
        cuda_visible = ",".join(map(str, range(num_accelerators)))

    # rotate list based on xdist worker id, example below
    # wid=0 -> ['0', '1', '2', '3']
    # wid=1 -> ['1', '2', '3', '0']
    # wid=2 -> ['2', '3', '0', '1']
    # wid=3 -> ['3', '0', '1', '2']
    dev_id_list = cuda_visible.split(",")
    dev_id_list = dev_id_list[xdist_worker_id:] + dev_id_list[:xdist_worker_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(dev_id_list)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

def initialize_cuda():
    my_local_rank = env2int(
        ["MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "MVP_COMM_WORLD_LOCAL_RANK"],
        0,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(my_local_rank % 4)
    torch.cuda.init()

def set_mpi_dist_environemnt(master_addr = None):
    if master_addr is not None:
        os.environ['MASTER_ADDR'] = master_addr
    # local_rank = env2int(
    #     ['LOCAL_RANK', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK', 'SLURM_LOCALID', 'MVP_COMM_WORLD_LOCAL_RANK'])
    local_rank = env2int(
        ["MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "MVP_COMM_WORLD_LOCAL_RANK"])
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(local_rank)
    rank = env2int(['RANK', 'MPI_RANKID', 'OMPI_COMM_WORLD_RANK', 'MV2_COMM_WORLD_RANK', 'SLURM_PROCID', 'MVP_COMM_WORLD_RANK'])
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(rank)
    world_size = env2int(['WORLD_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'SLURM_NPROCS', 'MVP_COMM_WORLD_SIZE'])
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(world_size)
    print(f"LOCAL_VAL => local_rank: {local_rank}, rank : {rank}, world_size : {world_size}")
    print(f"ENV_VAR => local_rank: {os.environ['LOCAL_RANK']}, rank : {os.environ['RANK']}, world_size : {os.environ['WORLD_SIZE']}")

