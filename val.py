import TorchDeepmath as td
import torch
from tqdm import tqdm
import os
import random
os.environ['CUDA_VISIBLE_DEVICES']="7"



if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    neg_hard_per_pos = 1
    neg_per_pos = 15
    bactch_size = 16
    word_size = 1
    address = '682563'

    model_size =0
    save_name = "/mnt/cache/share_data/zhoujq/ckpt/exp_init/model_epoch318"
    # save_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_load318/model_epoch80"
    # save_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr4/model_epoch274"
    # save_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr2/model_epoch98"

    # dataset = td.Data.dataset.GNN_dataset("../data/data_goal_lr.npy","../data/data_thm_lr.npy",{'neg_per_pos':neg_per_pos,'neg_hard_per_pos':neg_hard_per_pos})
    dataset = td.Data.dataset.GNN_dataset("../data/data_goal_lr_small0.npy","../data/data_thm_lr.npy",{'neg_per_pos':neg_per_pos,'neg_hard_per_pos':neg_hard_per_pos})

    if model_size == 0:
        model = td.Model.GNN.GNN_net({
            'goal_voc_length':1109,
            'goal_voc_embedsize':128,
            'thm_voc_length':1193,
            'thm_voc_embedsize':128,
            'num_hops':12,
            'score_weight':0.2,
            'tactic_weight':1.0,
            'auc_weight':4.0,
            'neg_per_pos':neg_per_pos,
            'bactch_size':bactch_size,
            'word_size':word_size
        })
    else:
        model = td.Model.GNN_zhou.GNN_net({
            'goal_voc_length':1109,
            'goal_voc_embedsize':128,
            'thm_voc_length':1193,
            'thm_voc_embedsize':128,
            'num_hops':16,
            'score_weight':0.2,
            'tactic_weight':1.0,
            'auc_weight':4.0,
            'neg_per_pos':neg_per_pos,
            'bactch_size':bactch_size,
            'word_size':word_size
        })

    td.Train.Train_GNN.ValLoop(dataset,model,save_name)
    
