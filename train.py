import TorchDeepmath as td
import torch
from tqdm import tqdm
import os




if __name__ == '__main__':

    neg_hard_per_pos = 1
    neg_per_pos = 15
    bactch_size = 128
    word_size = 8
    save_name="/mnt/cache/share_data/zhoujq/ckpt/exp_pclr_large_human_synth/model_epoch"
    load_name = None
    #  load_name = "/mnt/cache/share_data/zhoujq/ckpt/exp_pclr3/model_epoch137"
    # load_name = "/mnt/cache/share_data/zhoujq/ckpt/exp_pclr0/savelast"
    #  load_name = "/mnt/cache/share_data/zhoujq/ckpt/exp_init/model_epoch318"
    decay_rate = 10
    num_node = 1

    model_type =1

    path_goal = "/mnt/cache/zhoujingqiu/data/data_human_synth_1_3_lr.npy"
    #  path_goal = "/mnt/cache/zhoujingqiu/data/data_goal_lr.npy"
    path_thm = "/mnt/cache/zhoujingqiu/data/data_thm_lr.npy"

    dataset = td.Data.dataset.GNN_dataset(path_goal,path_thm,{'neg_per_pos':neg_per_pos,'neg_hard_per_pos':neg_hard_per_pos})
    
    if model_type == 0:
       model = td.Model.GNN.GNN_net({
            'goal_voc_length':1109,
            'goal_voc_embedsize':128,
            'thm_voc_length':1193,
            'thm_voc_embedsize':128,
            'num_hops':12,
            'score_weight':0.2,
            'tactic_weight':1.0,
            'auc_weight':40.0,
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

    td.Train.Train_GNN.TrainLoop(dataset,model,word_size,bactch_size,save_name,num_node,load_name=load_name,
                                                                                        decay_rate=decay_rate)
