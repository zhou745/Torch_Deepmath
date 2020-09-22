import TorchDeepmath as td
import torch
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES']="5"



if __name__ == '__main__':

    neg_hard_per_pos = 1
    neg_per_pos = 15
    bactch_size = 1
    word_size = 1
    save_name="/mnt/cache/share_data/zhoujingqiu/ckpt/exp5/model_epoch0"

    dataset = td.Data.dataset.GNN_dataset("../data/data_goal2.npy","../data/data_thm2.npy",{'neg_per_pos':neg_per_pos,'neg_hard_per_pos':neg_hard_per_pos})

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

    td.Train.Train_GNN.ValLoop(dataset,model,save_name)
