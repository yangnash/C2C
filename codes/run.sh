#!/bin/bash
source /root/.bashrc
############可选，查看申请的显卡状态###############
#nvidia-smi
##############必选，进⼊配置好的conda虚拟环境--orvit#############
# miniconda 路径
# 让conda命令⽣效
source /root/miniconda3/bin/activate
conda activate pytorch
#cd /opt/data/private/exp/C2CP/codes
cd /opt/data/private/bishe/C2C/codes
#  CUDA_VISIBLE_DEVICES=0 python -u train.py --config config/c2c_vm/c2c_vanilla_tsm.yml 2>&1 | tee ./outputserplus1.txt
CUDA_VISIBLE_DEVICES=0 python -u train.py --config config/c2c_vm/c2c_vanilla_tsm.yml 2>&1 | tee ./tsm.txt