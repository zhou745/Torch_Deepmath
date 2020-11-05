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

    args= np.load(config_args.configs,allow_pickle=True).tolist()
    model_type = args.model_type if hasattr(args,'model_type') else 0
    print("running with parameters")
    print(args,flush=True)
    dataset = td.Data.dataset.GNN_dataset(args)
    
    if model_type == 0:
        model = td.Model.GNN.GNN_net(args)
    elif model_type == 1:
        model = td.Model.GNN_experimental.GNN_net(args)
    else:
        raise RuntimeError('unknown model')
    
    td.Train.Train_GNN.TrainLoop(dataset,model,args)
            