import torch
import torch.optim as optim
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from TorchDeepmath.Data.dataset import Batch_collect
from collections import OrderedDict


def setup(rank, address,world_size,name='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = address
    # os.environ['MASTER_PORT'] = '78902'
    # initialize the process group
    dist.init_process_group(name, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def model_parallel(rank, world_size,dataset,model,batch_size,save_name,address):
    # device = 'cuda'
    print("current rank is %d"%(rank),flush=True)
    setup(rank,address, world_size)

    model_new = model.to(rank)
    # state_dict = torch.load("/mnt/cache/share_data/zhoujingqiu/ckpt/exp_pclr_3/model_epoch58",map_location=torch.device('cuda:'+str(rank)))
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     if 'module' not in k:
    #         continue
    #     else:
    #         name = k[14:] # remove `module.`
    #         new_state_dict[name] = v
    # model_new.load_state_dict(new_state_dict)
    # print("ckpt loaded!",flush=True)

    model_new = DDP(model_new,device_ids=[rank])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True, seed=123456)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size//world_size,
                                                            collate_fn=Batch_collect,
                                                            sampler=sampler)
    print("loader build finished",flush=True) 
    # swa_model = AveragedModel(model_new,avg_fn = lambda ap, mp, nv:0.01*mp+0.99*ap)
    swa_model = AveragedModel(model_new,avg_fn = lambda ap, mp, nv:0.0001*mp+0.9999*ap)
    # swa_model = AveragedModel(model_new,avg_fn = lambda ap, mp, nv:0.05*mp+0.95*ap)

    # dir_pth = save_name.rstrip('/model_epoch')
    # if not os.path.exists(dir_pth):
    #     os.mkdir(dir_pth)

    if rank==0:
        f = open(save_name+"log","w")
        f.close()
    # model_new.to(device)
    #borad cast the initial weight as the same
    for parameter in model_new.parameters():
        torch.distributed.broadcast(parameter,0,async_op=False)
    
    idx = 0 
    model_new.train(True)
    optimizer=optim.Adam(model_new.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-3, weight_decay=0, amsgrad=False)
    while True:
        print("set epoch ",flush=True)
        sampler.set_epoch(idx)
        print("set epoch finished",flush=True)
        dist.barrier()
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
                # for p in model_new.parameters():

                #     print(p.grad,flush=True)
                log_str = str(score_out.item())+" "+str(tactic_out.item())+" "+str(auc_out.item())+"\n"
                f = open(save_name+"log","a")
                f.write(log_str)
                f.close()
                print("At epoch "+str(idx)+" the current loss is "+str(loss_out.item())+
                        " tactic "+str(tactic_out.item())+
                        " score "+str(score_out.item())+
                        " auc "+str(auc_out.item()),flush=True)
        dist.barrier()

        if idx%35 == 34:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.98
        if rank==0:
            torch.save(swa_model.state_dict(), save_name+str(idx))
            torch.save(model_new.state_dict(), save_name+str(idx)+"_no_mean")
        idx=idx+1  

    cleanup()


def TrainLoop(dataset,model,world_size,batch_size,save_name,address):
    if world_size>0:
        train_multi_gpu(dataset,model,world_size,batch_size,save_name,address)

def ValLoop(dataset,model,save_name):
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=1,
                                                        collate_fn=Batch_collect,
                                                        num_workers=4)

    state_dict = torch.load(save_name,map_location=torch.device('cpu'))
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

    N_all = 0
    N_true_tac = 0
    N_true_sco = 0
    N_true_sample = 0
    for item in tqdm(data_loader):
        for k in item.keys():
            item[k]=item[k].to(device)
        result = model(item)

        tactic = result['tactic_scores']
        score = result['logits'].view(-1)
        # print(torch.sigmoid(score),flush=True)


        gt = item['tac_id'].item()
        tac_sco,tac_topk = torch.topk(tactic,1)
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
    print("tac acuracy is %f"%(N_true_tac/N_all))
    print("score acuracy is %f"%(N_true_sco/N_all))
    print("sample acuracy is %f"%(N_true_sample/N_all))



def train_multi_gpu(dataset,model,world_size,batch_size,save_name,address):

    mp.spawn(model_parallel,args=(world_size,dataset,model,batch_size,save_name,address),
                            nprocs=world_size,
                            join=True)
    
