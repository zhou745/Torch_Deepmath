import torch
import torch.optim as optim
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import os
from TorchDeepmath.Data.dataset import Batch_collect
from collections import OrderedDict
import socket
import time

def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

def setup(rank, dist_url,world_size,name='nccl'):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = address
    # os.environ['MASTER_PORT'] = '78902'
    # initialize the process group
    dist.init_process_group(name, init_method=dist_url,rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def model_parallel(rank,pid,dist_url,dataset,model,args):
    # device = 'cuda'
    gpu = rank
    rank = pid*8+gpu
    print("current rank is %d"%(rank),flush=True)
    setup(rank,dist_url, args.world_size)

    model_new = model.to(gpu)
    if args.load_name is not None:
        state_dict = torch.load(args.load_name,map_location=torch.device('cuda:'+str(gpu)))
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' not in k:
                continue
            else:
                while 'module' in k:
                    k = k[7:]
                new_state_dict[k] = v
        model_new.load_state_dict(new_state_dict)
        print("ckpt loaded!",flush=True)

    model_new = DDP(model_new,device_ids=[gpu])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True, seed=123456)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size//args.world_size,
                                                            collate_fn=Batch_collect,
                                                            sampler=sampler,
                                                            num_workers=args.num_worker)
    print("loader build finished",flush=True) 
    # swa_model = AveragedModel(model_new,avg_fn = lambda ap, mp, nv:0.01*mp+0.99*ap)
    swa_model = AveragedModel(model_new,avg_fn = lambda ap, mp, nv:0.0001*mp+0.9999*ap)
    # swa_model = AveragedModel(model_new,avg_fn = lambda ap, mp, nv:0.05*mp+0.95*ap)

    if rank==0:
        dir_pth,file_name = os.path.split(args.save_name)
        if not os.path.exists(dir_pth):
            os.mkdir(dir_pth)
            print("create directory "+dir_pth,flush=True)

        f = open(args.save_name+"log","w")
        f.close()
        np.save(args.save_name+"_config",args)
    # model_new.to(device)
    #borad cast the initial weight as the same
    for parameter in model_new.parameters():
        torch.distributed.broadcast(parameter,0,async_op=False)
    
    idx = 0 
    model_new.train(True)
    optimizer=optim.Adam(model_new.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-3, weight_decay=0, amsgrad=False)
    while True:
        print("set epoch ",flush=True)
        sampler.set_epoch(idx)
        print("set epoch finished",flush=True)
        dist.barrier()
        step = 0
        for item in tqdm(data_loader):
            # pass
            tactic_loss,score_loss,auc_loss,reg_loss = model_new(item)

            loss=tactic_loss+ score_loss+auc_loss+reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #reduce the loss
            loss_out = loss.detach()
            tactic_out = tactic_loss.detach()
            score_out = score_loss.detach()
            auc_out = auc_loss.detach()
            torch.distributed.all_reduce(loss_out, async_op=False)
            torch.distributed.all_reduce(tactic_out, async_op=False)
            torch.distributed.all_reduce(score_out, async_op=False)
            torch.distributed.all_reduce(auc_out, async_op=False)

            swa_model.update_parameters(model_new)

            if rank==0:
                log_str = str(score_out.item())+" "+str(tactic_out.item())+" "+str(auc_out.item())+"\n"
                f = open(args.save_name+"log","a")
                f.write(log_str)
                f.close()
                print("At epoch "+str(idx)+" step "+str(step)+" the current loss is "+str(loss_out.item())+
                        " tactic "+str(tactic_out.item())+
                        " score "+str(score_out.item())+
                        " auc "+str(auc_out.item()),flush=True)
                step+=1
        dist.barrier()
        for parameter in model_new.parameters():
            torch.distributed.broadcast(parameter,0,async_op=False)

        if idx%args.decay_rate == (args.decay_rate-1):
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*args.lr_decay
        if rank==0 and idx%args.save_frequency==0:
            torch.save(swa_model.state_dict(), args.save_name+str(idx))
            torch.save(model_new.state_dict(), args.save_name+str(idx)+"_master_copy")
        idx=idx+1  

    cleanup()


def TrainLoop(dataset,model,args):
    #write the host file
    pid = int(os.environ["SLURM_PROCID"])
    jobid = os.environ["SLURM_JOBID"]
    hostfile = "dist_url_" + jobid  + ".txt"
    
    ip = socket.gethostbyname(socket.gethostname())
    port = find_free_port()
    dist_url = None

    if pid == 0:
        dist_url = "tcp://{}:{}".format(ip, port)
        with open(hostfile, "w") as f:
            f.write(dist_url)
    else:
        while not os.path.exists(hostfile):
                time.sleep(1)
        with open(hostfile, "r") as f:
            dist_url = f.read()
    
    train_multi_gpu(dataset,model,pid,dist_url,args)

def ValLoop(dataset,model,args):
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=1,
                                                        collate_fn=Batch_collect,
                                                        num_workers=4)

    state_dict = torch.load(args.load_name,map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' not in k:
            continue
        else:
            while 'module' in k:
                k = k[7:]
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    device = torch.device('cuda:0')
    model.to(device)
    model.train(False)
    # print(torch.sum(torch.abs(model.goal_embed.tokenvectors[8,:])),flush=True)
    # print(torch.sum(torch.abs(model.thm_embed.tokenvectors[8,:])),flush=True)
    # model.eval()

    N_all = 0
    N_true_tac = 0
    N_true_sco = 0
    N_true_sample = 0

    for item in tqdm(data_loader):
        for k in item.keys():
            item[k]=item[k].to(device)
        # print("----------------------------------",flush=True)
        # print(item['thm_edge_p_indicate'],flush=True)
        # print("==================================",flush=True)
        # print(item['thm_edge_c_indicate'],flush=True)
        result = model(item)

        tactic = result['tactic_scores']
        score = result['logits'].view(-1)
        # print(torch.sigmoid(score),flush=True)


        gt = item['tac_id'].item()
        tac_sco,tac_topk = torch.topk(tactic,5)
        #print(tac_sco,flush=True)
        # print(tac_topk,flush=True)
        _,score_top1 = torch.topk(score,1)
        
        if gt in tac_topk:
            N_true_tac+=1

        if score_top1==0:
            N_true_sco+=1

        if score_top1==0 and gt in tac_topk:
            N_true_sample+=1
        N_all+=1
    print("tac acuracy is %f"%(N_true_tac/N_all),flush=True)
    print("score acuracy is %f"%(N_true_sco/N_all),flush=True)
    print("sample acuracy is %f"%(N_true_sample/N_all),flush=True)



def train_multi_gpu(dataset,model,pid,dist_url,args):

    mp.spawn(model_parallel,args=(pid,dist_url,dataset,model,args),
                            nprocs=args.world_size//args.num_node,
                            join=True)
    
