#!/bin/sh

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="10.141.8.84" --master_port=9876 tools/train_net.py --config-file configs/fcos/fcos_X_101_32x8d_FPN_2x_lsvt.yaml

# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="10.141.8.84" --master_port=9876 tools/train_net.py --config-file configs/fcos/fcos_X_101_32x8d_FPN_2x_lsvt.yaml

# python -m torch.distributed.launch --nproc_per_node=8 tools/test_net.py --config-file configs/fcos/fcos_R_50_FPN_1x.yaml MODEL.WEIGHT models/FCOS_R_50_FPN_1x.pth