# Installation Guide for MVAPICH2-GDR

**To install MVAPICH2-GDR, refer https://mvapich.cse.ohio-state.edu/userguide/gdr/**


<div align="center">
  <b>OR</b> 
</div>

**You can follow below instructions to install MVAPICH2-GDR from RPMs**


### Get the approriate RPMs for your system

In this case, we used MOFED 5.5 RPMs
```bash
wget https://mvapich.cse.ohio-state.edu/download/mvapich/gdr/2.3.7/mofed5.5/mvapich2-gdr-cuda11.6.mofed5.5.gnu8.5.0-2.3.7-1.el8.x86_64.rpm
```
### Unpack the rpm
```bash
rpm2cpio mvapich2-gdr-cuda11.6.mofed5.5.gnu8.5.0-2.3.7-1.el8.x86_64.rpm | cpio -id
```

### Note the path of your rpm. It should ook as follows:
```
<directory>/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0
```

### Add the rpm to your path (this needs to be run every time you start a new job)

```bash
export RPM_HOME=<directory>/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0
export PATH=$RPM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$RPM_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$RPM_HOME/include:$CPATH
```

### Load the gcc and cuda versions for our rpm
```bash
module load cuda/11.6 gcc/8.5.0
```

### Update the compiler paths to be absolute instead of relative 
```bash
vi $RPM_HOME/bin/mpicc
```

#### Incorrect paths look like:
```
prefix=/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0
exec_prefix=/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0
sysconfdir=/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0/etc
includedir=/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0/include
libdir=/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0/lib64
```
#### Change these with the value of $RPM_HOME prepended like:
```
prefix=<directory>/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0
exec_prefix=<directory>/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0
sysconfdir=<directory>/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0/etc
includedir=<directory>/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0/include
libdir=<directory>/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0/opt/mvapich2/gdr/2.3.7/no-mpittool/no-openacc/cuda11.6/mofed5.5/mpirun/gnu8.5.0/lib64
```


### Check installation by running "osu micro benchmarks (OMB)" benchmark

OMB performs common MPI operations like allreduce, bcast, send/recv, etc. OMB is located in **$RPM_HOME/libexec**

#### Run an allreduce benchmark on 2 GPUs.

following output is expected 

```
[gulhane.2@a100-01 libexec]$ $MV2_HOME/bin/mpirun_rsh --export-all -np 2 a100-01 a100-01 MV2_USE_CUDA=1 osu-micro-benchmarks/mpi/collective/osu_allreduce -d
 cuda
[a100-01.cluster:mpi_rank_0][rdma_param_handle_heterogeneity] All nodes involved in the job were detected to be homogeneous in terms of processors and interconnects. Setting MV2_HOMOGENEOUS_CLUSTER=1 can improve job startup performance on such systems. The following link has more details on enhancing job startup performance. http://mvapich.cse.ohio-state.edu/performance/job-startup/.
[a100-01.cluster:mpi_rank_0][rdma_param_handle_heterogeneity] To suppress this warning, please set MV2_SUPPRESS_JOB_STARTUP_PERFORMANCE_WARNING to 1

# OSU MPI-CUDA Allreduce Latency Test v5.9
# Size       Avg Latency(us)
4                       1.98
8                       2.04
16                      1.60
[a100-01.cluster:mpi_rank_0][dreg_register] [Performance Impact Warning]: Entries are being evicted from the InfiniBand registration cache. This can lead to degraded performance. Consider increasing MV2_NDREG_ENTRIES_MAX (current value: 16384) and MV2_NDREG_ENTRIES (current value: 8196)
[a100-01.cluster:mpi_rank_1][dreg_register] [Performance Impact Warning]: Entries are being evicted from the InfiniBand registration cache. This can lead to degraded performance. Consider increasing MV2_NDREG_ENTRIES_MAX (current value: 16384) and MV2_NDREG_ENTRIES (current value: 8196)
32                     13.30
64                     13.47
128                    13.48
256                    15.15
512                    13.82
1024                   13.93
2048                   14.31
4096                   15.40
8192                   16.65
16384                  19.72
32768                 456.73
65536                 503.67
131072                500.02
262144                457.37
524288                631.29
1048576               630.67
```
