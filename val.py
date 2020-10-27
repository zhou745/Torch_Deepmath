import TorchDeepmath as td
import torch
from tqdm import tqdm
import os
import random
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']="2"
# import argparse

if __name__ == '__main__':
    # args = parser.parse_args()

    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_0hop_small_nodrop_human_no_norm.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_0hop_small_nodrop_human_norm.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_small_no_hard.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_0hop_small_remove_relu.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_small_remove_relu.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_16hop_large_remove_relu.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_0hop_small_remove_relu_lr8.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_small_remove_relu_lr8.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_small_remove_relu_lr4.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_0hop_small_remove_relu_lr4.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_2hop_small_remove_relu_lr4.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_2hop_small_remove_relu_lr8.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_2hop_small_remove_relu_lr8_auc100.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_1hop_small_remove_relu_lr8.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_1hop_small_remove_relu_lr8_no_drop.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_2hop_small_remove_relu_lr8_no_drop.npy"
    load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_small_remove_relu_lr8_no_drop.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_16hop_small_remove_relu_lr8_no_drop.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_2hop_small_remove_relu_lr8_drop.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_4hop_small_remove_relu_lr8_drop.npy"
    args = np.load(load_path,allow_pickle=True).tolist()

    random.seed(0)
    torch.manual_seed(0)

    model_size =0
    
    
    args.path_goal = "/mnt/cache/zhoujingqiu/data/data_goal_valid_small.npy"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_2hop_small_remove_relu_lr8/model_epoch60"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_1hop_small_remove_relu_lr8/model_epoch20"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_1hop_small_remove_relu_lr8_no_drop/model_epoch175"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_2hop_small_remove_relu_lr8_no_drop/model_epoch100"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_2hop_small_remove_relu_lr8_drop/model_epoch100"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_4hop_small_remove_relu_lr8_drop/model_epoch100"
    args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_12hop_small_remove_relu_lr8_no_drop/model_epoch105"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_16hop_small_remove_relu_lr8_no_drop/model_epoch15"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_2hop_small_remove_relu_lr4/model_epoch180"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_2hop_small_remove_relu_lr8_auc100/model_epoch10"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_0hop_small_remove_relu_lr4/model_epoch90"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_0hop_small_remove_relu_lr8/model_epoch370"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_12hop_small_remove_relu_lr4/model_epoch80"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_12hop_small_remove_relu_lr8/model_epoch55"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_12hop_small_remove_relu/model_epoch305"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_0hop_small_nodrop_human_no_norm/model_epoch210"
    # args.load_name = "/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_0hop_small_nodrop_human_norm/model_epoch295"
    # args.load_name = "/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_12hop_small_no_hard/model_epoch105"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_0hop_small_remove_relu/model_epoch835"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_16hop_large_remove_relu/model_epoch90"
    print(args,flush=True)
    # dataset = td.Data.dataset.GNN_dataset("../data/data_goal_lr.npy","../data/data_thm_lr.npy",{'neg_per_pos':neg_per_pos,'neg_hard_per_pos':neg_hard_per_pos})
    dataset = td.Data.dataset.GNN_dataset(args)

    if model_size == 0:
        model = td.Model.GNN.GNN_net(args)
    else:
        raise RuntimeError('unknown model')

    td.Train.Train_GNN.ValLoop(dataset,model,args)
    
