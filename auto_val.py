import TorchDeepmath as td
import torch
from tqdm import tqdm
import os
import random
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description="Path Hyperparameters")
parser.add_argument('--load_path',type=str, default=None)
parser.add_argument('--gpu', type=str,default=None)

if __name__ == '__main__':
    path_arg = parser.parse_args()
    assert path_arg.load_path is not None
    assert path_arg.gpu is not None
    os.environ['CUDA_VISIBLE_DEVICES']=path_arg.gpu
    args = np.load(path_arg.load_path,allow_pickle=True).tolist()
    

    # load_path = "/mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_small_remove_relu_lr1_noshare_xavier.npy"

    random.seed(0)
    torch.manual_seed(0)
    model_size =0

    args.batch_size = 16
    args.path_goal = "/mnt/cache/zhoujingqiu/data/data_goal_valid_small_fang.npy"
    args.path_thm = "/mnt/cache/zhoujingqiu/data/data_thm_lr_fang.npy"
    # args.path_goal = "/mnt/cache/zhoujingqiu/data/data_goal_valid_small.npy"
    _,dir_name = os.path.split(path_arg.load_path)
    dir_name = dir_name.replace(".npy","")
    file_name = "/mnt/cache/share_data/zhoujq/ckpt/"+dir_name+"/model_epoch"

    idx = 0
    step = 5
    current_file = file_name+str(idx)
    while True:

        if os.path.isfile(current_file):
            args.load_name=current_file
            print(args,flush=True)
            dataset = td.Data.dataset.GNN_dataset(args)

            if model_size == 0:
                model = td.Model.GNN.GNN_net(args)
            else:
                raise RuntimeError('unknown model')

            td.Train.Train_GNN.ValLoop(dataset,model,args)

            idx+=step
            current_file = file_name+str(idx)
        else:
            time.sleep(60)

    
