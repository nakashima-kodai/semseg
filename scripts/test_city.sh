#!/bin/bash

#$ -l rt_G.small=1
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

name=cityscapes
root=/fs1/groups1/gaa50131/user/nakashima/datasets/cityscapes
ckpt_pth=outputs/2020-08-08/23-49-19/deeplabv3_r50_bs008_lr1.0e-03_ep050.pth
python -B test.py dataset_name=$name dataroot=$root phase=test image_preprocess=fixed label_preprocess=fixed validation=False ckpt_pth=$ckpt_pth batch_size=1
