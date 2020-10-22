import TorchDeepmath as td
import torch
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description="Training Hyperparameters")
#dataset parameter
parser.add_argument('--neg_hard_per_pos', type=int,default=1)
parser.add_argument('--neg_per_pos', type=int,default=15)
parser.add_argument('--path_goal', default="/mnt/cache/zhoujingqiu/data/goal_human.npy")
# parser.add_argument('--path_goal', default="/mnt/cache/zhoujingqiu/data/goal_human_synth.npy")
# parser.add_argument('--path_goal', default="/mnt/cache/zhoujingqiu/data/data_goal_lr.npy")
parser.add_argument('--path_thm', default="/mnt/cache/zhoujingqiu/data/data_thm_lr.npy")
#training parameter
parser.add_argument('--batch_size', type=int,default=128)
parser.add_argument('--world_size', type=int,default=8)
parser.add_argument('--num_node', type=int,default=1)
parser.add_argument('--decay_rate', type=int,default=10)
parser.add_argument('--lr', type=float,default=1e-4)
parser.add_argument('--lr_decay', type=float,default=0.98)
parser.add_argument('--num_worker', type=int,default=4)
#ckpt parameter
parser.add_argument('--save_name', default="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_0hop_small_nodrop_human_test/model_epoch")
parser.add_argument('--load_name', default=None)
parser.add_argument('--save_frequency', type=int,default=5)
#model parameter
parser.add_argument('--goal_voc_length', type=int,default=1109)
parser.add_argument('--goal_voc_embedsize', type=int,default=128)
parser.add_argument('--thm_voc_length', type=int,default=1193)
parser.add_argument('--thm_voc_embedsize', type=int,default=128)
parser.add_argument('--num_hops', type=int,default=0)
parser.add_argument('--score_weight', type=float,default=0.2)
parser.add_argument('--tactic_weight', type=float,default=1.0)
parser.add_argument('--auc_weight', type=float,default=4.0)
parser.add_argument('--gnn_layer_size', type=list,default=[256,128])
parser.add_argument('--neck_layer_size', type=list,default=[512,1024])
parser.add_argument('--tac_layer_size', type=list,default=[512,256,41])
parser.add_argument('--thm_layer_size', type=int,default=[1024,512,1])
# parser.add_argument('--neck_layer_size', type=list,default=[512,4096])
# parser.add_argument('--tac_layer_size', type=list,default=[1024,256,41])
# parser.add_argument('--thm_layer_size', type=int,default=[2048,256,1])

if __name__ == '__main__':
    args = parser.parse_args()
    model_type =0

    # # path_goal = "/mnt/cache/zhoujingqiu/data/goal_human_synth.npy"
    # # path_goal = "/mnt/cache/zhoujingqiu/data/data_human_synth_1_3_lr.npy"
    # path_goal = "/mnt/cache/zhoujingqiu/data/data_goal_lr.npy"
    # path_thm = "/mnt/cache/zhoujingqiu/data/data_thm_lr.npy"

    dataset = td.Data.dataset.GNN_dataset(args)
    
    if model_type == 0:
        model = td.Model.GNN.GNN_net(args)
    else:
        raise RuntimeError('unknown model')
    
    td.Train.Train_GNN.TrainLoop(dataset,model,args)
