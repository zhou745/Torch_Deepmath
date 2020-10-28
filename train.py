import TorchDeepmath as td
import torch
from tqdm import tqdm
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Config Parameter")
#dataset parameter
parser.add_argument('--configs', default=None)


if __name__ == '__main__':
    config_args = parser.parse_args()
    model_type =0

    args= np.load(config_args.configs,allow_pickle=True).tolist()
    print("running with parameters")
    print(args,flush=True)
    dataset = td.Data.dataset.GNN_dataset(args)
    
    if model_type == 0:
        model = td.Model.GNN.GNN_net(args)
    else:
        raise RuntimeError('unknown model')
    
    td.Train.Train_GNN.TrainLoop(dataset,model,args)
            