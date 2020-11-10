import TorchDeepmath as td
import torch
from tqdm import tqdm
import os
import random
import numpy as np
import argparse
import time
import pandas as pd

parser = argparse.ArgumentParser(description="Path Hyperparameters")
parser.add_argument('--load_path',type=str, default=None)
parser.add_argument('--gpu', type=str,default=None)

if __name__ == '__main__':
    tactic_accuracy_list = []
    theorem_accuracy_list = []
    combine_accuracy_list = []
    df = pd.DataFrame({'Tac top5 acc': tactic_accuracy_list, 'Thm top1 acc': theorem_accuracy_list,
                       'Sample accuracy': combine_accuracy_list})
    df = df.T

    path_arg = parser.parse_args()
    assert path_arg.load_path is not None
    assert path_arg.gpu is not None
    os.environ['CUDA_VISIBLE_DEVICES']=path_arg.gpu
    args = np.load(path_arg.load_path,allow_pickle=True).tolist()
    

    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_small_remove_relu_lr1_noshare_xavier.npy"

    random.seed(0)
    torch.manual_seed(0)

    args.batch_size = 16
    args.path_goal = "/mnt/cache/zhoujingqiu/data/data_goal_valid_small_fang.npy"
    args.path_thm = "/mnt/cache/zhoujingqiu/data/data_thm_lr_fang.npy"
    model_type = args.model_type if hasattr(args,"model_type") else 0

    if hasattr(args,'mask_token'):
        args.mask_token=False
    # args.path_goal = "/mnt/cache/zhoujingqiu/data/data_goal_valid_small.npy"
    _,dir_name = os.path.split(path_arg.load_path)
    dir_name = dir_name.replace(".npy","")
    file_name = "/mnt/cache/share_data/zhoujq/ckpt/"+dir_name+"/model_epoch"
    # file_name = "/mnt/lustre/share_data/fangrongyao/ckpt/exp_hop4_lr4_batch128_layernorm_gnnres_No1019/model_epoch"

    idx = 150
    step = 5
    # current_file = file_name+str(idx)
    current_file = file_name+str(idx)

    exp_name = "tmp"
    csv_folder = 'val_result/{:}.csv'.format(exp_name)
    if model_type == 0:
        model = td.Model.GNN.GNN_net(args)
    elif model_type==1:
        model = td.Model.GNN_experimental.GNN_net(args)
    else:
        raise RuntimeError('unknown model')
    while True:
        if os.path.isfile(current_file):
            args.load_name=current_file
            print(args,flush=True)
            dataset = td.Data.dataset.GNN_dataset(args)

            tac,thm,sample=td.Train.Train_GNN.ValLoop(dataset,model,args)

            df.loc['Tac top5 acc', str(idx)] = '{:.6f}'.format(tac)
            df.loc['Thm top1 acc', str(idx)] = '{:.6f}'.format(thm)
            df.loc['Sample accuracy', str(idx)] = '{:.6f}'.format(sample)
            df.to_csv(csv_folder)

            idx+=step
            # current_file = file_name+str(idx)
            current_file = file_name+str(idx)
        else:
            time.sleep(1)

    
