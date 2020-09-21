import torch
import torch.optim as optim
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from TorchDeepmath.Data.dataset import Batch_collect


def setup(rank, world_size,name='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(name, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def model_parallel(rank, world_size,dataset,model,batch_size,save_name):
    # device = 'cuda'
    setup(rank, world_size)

    model_new = model.to(rank)
    model_new = DDP(model_new,device_ids=[rank],find_unused_parameters=True)
    # model_new.to(device)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size//world_size,
                                                        collate_fn=Batch_collect,
                                                        sampler=sampler,
                                                        shuffle=True)
    # for parameter in model_new.parameters():
    #     print(parameter)
    model_new.train()
    optimizer=optim.Adam(model_new.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    while True:   
        idx = 0          
        for item in tqdm(data_loader):
            
            loss,tactic_loss,score_loss,auc_loss = model_new(item)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_out = loss.detach()
            tactic_out = tactic_loss.detach()
            score_out = score_loss.detach()
            auc_out = auc_loss.detach()

            if rank==0:
                loss_all = torch.distributed.all_reduce(loss_out, async_op=False)
                loss_tac = torch.distributed.all_reduce(tactic_out, async_op=False)
                loss_sco = torch.distributed.all_reduce(score_out, async_op=False)
                loss_auc = torch.distributed.all_reduce(auc_out, async_op=False)
                print("the current loss is "+str(loss_all.item())+
                        " tactic "+str(loss_tac.item())+
                        " score "+str(loss_sco.item())+
                        " auc "+str(loss_auc.item()),flush=True)
        if rank==0:
            model_new.save(save_name+str(idx))
        idx+=1



        

    cleanup()


def TrainLoop(dataset,model,world_size,batch_size,save_name):
    if world_size>1:
        train_multi_gpu(dataset,model,world_size,batch_size,save_name)



def train_multi_gpu(dataset,model,world_size,batch_size,save_name):

    mp.spawn(model_parallel,args=(world_size,dataset,model,batch_size,save_name),
                            nprocs=world_size,
                            join=True)
    
