# https://github.com/f4exb/sdrangel/issues/524
apt-get install software-properties-common
apt-get update
add-apt-repository ppa:ubuntu-toolchain-r/test
apt-get update
apt install gcc-10

apt-get install mpich
apt install libopenmpi-dev
pip install mpi4py

apt install -y --allow-downgrades libnccl2=2.12.12-1+cuda11.0 libnccl-dev=2.12.12-1+cuda11.0
pip install protobuf==3.20.1
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod

