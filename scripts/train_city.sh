#!/bin/bash

#$ -l rt_F=2
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd


source /etc/profile.d/modules.sh

module load python/3.6/3.6.5
module load cuda/10.2/10.2.89
module load cudnn/7.6/7.6.5
module load nccl/2.5/2.5.6-1
module load openmpi/4.0.3
module load gcc/7.4.0

export PATH="~/anaconda3/bin:${PATH}"
source activate ddp
cd /groups1/gaa50131/user/nakashima/ICRA2021/seg


NUM_NODES=${NHOSTS}
NUM_GPUS_PER_NODE=4
NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})

MPIOPTS="-np ${NUM_PROCS} -map-by ppr:${NUM_GPUS_PER_NODE}:node -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include bond0"

name=cityscapes
root=/fs1/groups1/gaa50131/user/nakashima/datasets/cityscapes
mpirun ${MPIOPTS} python -B train.py dataset_name=$name dataroot=$root validation=False imgnet_pretrained=False transfer_learning=False batch_size=8 crop_size=720 lr=0.0125 epochs=500 save_epoch_freq=100 log_step_freq=400
