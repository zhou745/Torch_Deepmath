import TorchDeepmath as td
import torch
from tqdm import tqdm
import os
import random
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']="2"
# import argparse

# parser = argparse.ArgumentParser(description="Training Hyperparameters")
# #dataset parameter
# parser.add_argument('--neg_hard_per_pos', type=int,default=1)
# parser.add_argument('--neg_per_pos', type=int,default=15)
# # parser.add_argument('--path_goal', default="/mnt/cache/zhoujingqiu/data/goal_human_synth.npy")
# parser.add_argument('--path_goal', default="/mnt/cache/zhoujingqiu/data/data_goal_valid_small.npy")
# # parser.add_argument('--path_goal', default="/mnt/cache/zhoujingqiu/data/data_goal_lr_small0.npy")
# parser.add_argument('--path_thm', default="/mnt/cache/zhoujingqiu/data/data_thm_lr.npy")
#training parameter
# parser.add_argument('--batch_size', type=int,default=128)
# parser.add_argument('--world_size', type=int,default=8)
# parser.add_argument('--num_node', type=int,default=1)
# parser.add_argument('--decay_rate', type=int,default=10)
# parser.add_argument('--lr', type=float,default=1e-4)
# parser.add_argument('--lr_decay', type=float,default=0.98)
# #ckpt parameter
# parser.add_argument('--save_name', default="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_0hop_small_nodrop/model_epoch")
# parser.add_argument('--load_name', default=None)
# parser.add_argument('--save_frequency', type=int,default=5)
# #model parameter
# parser.add_argument('--goal_voc_length', type=int,default=1109)
# parser.add_argument('--goal_voc_embedsize', type=int,default=128)
# parser.add_argument('--thm_voc_length', type=int,default=1193)
# parser.add_argument('--thm_voc_embedsize', type=int,default=128)
# parser.add_argument('--num_hops', type=int,default=0)
# parser.add_argument('--score_weight', type=float,default=0.2)
# parser.add_argument('--tactic_weight', type=float,default=1.0)
# parser.add_argument('--auc_weight', type=int,default=4.0)
# parser.add_argument('--gnn_layer_size', type=list,default=[256,128])
# parser.add_argument('--neck_layer_size', type=list,default=[512,1024])
# parser.add_argument('--tac_layer_size', type=list,default=[512,256,41])
# parser.add_argument('--thm_layer_size', type=int,default=[1024,512,1])
# parser.add_argument('--neck_layer_size', type=list,default=[512,4096])
# parser.add_argument('--tac_layer_size', type=list,default=[1024,256,41])
# parser.add_argument('--thm_layer_size', type=int,default=[2048,256,1])

if __name__ == '__main__':
    # args = parser.parse_args()

    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_0hop_small_nodrop_human_no_norm.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_0hop_small_nodrop_human_norm.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_small_no_hard.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_0hop_small_remove_relu.npy"
    load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_small_remove_relu.npy"
    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_16hop_large_remove_relu.npy"
    args = np.load(load_path,allow_pickle=True).tolist()

    random.seed(0)
    torch.manual_seed(0)

    model_size =0
    
    
    args.path_goal = "/mnt/cache/zhoujingqiu/data/data_goal_valid_small.npy"
    args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_12hop_small_remove_relu/model_epoch305"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_0hop_small_nodrop_human_no_norm/model_epoch210"
    # args.load_name = "/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_0hop_small_nodrop_human_norm/model_epoch295"
    # args.load_name = "/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_12hop_small_no_hard/model_epoch105"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_0hop_small_remove_relu/model_epoch835"
    # args.load_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_16hop_large_remove_relu/model_epoch55"
    print(args,flush=True)
    # dataset = td.Data.dataset.GNN_dataset("../data/data_goal_lr.npy","../data/data_thm_lr.npy",{'neg_per_pos':neg_per_pos,'neg_hard_per_pos':neg_hard_per_pos})
    dataset = td.Data.dataset.GNN_dataset(args)

    if model_size == 0:
        model = td.Model.GNN.GNN_net(args)
    else:
        raise RuntimeError('unknown model')

    td.Train.Train_GNN.ValLoop(dataset,model,args)
    
