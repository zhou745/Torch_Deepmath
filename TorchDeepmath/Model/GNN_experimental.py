import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_scatter
import numpy as np
import torch.distributed as dist

init_a = -0.01
init_b = 0.01

p1 = 0.5
p2 = 0.3

class Toynet(nn.Module):
    def __init__(self):
        super(Toynet, self).__init__()
        self.fc1 = nn.Linear(256,256)
    
    def forward(self, x):
        # print(x['thm_token'].shape,flush=True)
        # print(x['length_list_g'].shape,flush=True)
        return torch.tensor([0.])

class Tokenstore(nn.Module):
    def __init__(self,voc_length,
                      voc_embedsize=128,
                      use_embed = False,
                      max_norm = None,
                      init_methd = None):
        super(Tokenstore,self).__init__()
        
        self.voc_length = voc_length
        self.voc_embedsize = voc_embedsize
        self.use_embed = use_embed
        self.start_index = None
        self.end_index = None
        self.init_methd = init_methd
        
        if not self.use_embed:
            self.register_parameter(name='tokenvectors', param=torch.nn.Parameter(torch.zeros([self.voc_length+2,self.voc_embedsize],
                                                                                            dtype=torch.float32)))
        else:
            self.tokenvectors=torch.nn.Embedding(self.voc_length+2,self.voc_embedsize,max_norm=max_norm)
        
        for p in self.parameters():
            if self.init_methd==None:
                pass
            elif self.init_methd=='uniform':
                init.uniform_(p, a=init_a, b=init_b)
            elif self.init_methd=='xavier':
                init.xavier_normal_(p)
            else:
                raise RuntimeError('unknown init method')


    def forward(self,token_idx):
        if not self.use_embed:
            batch_tokens = self.tokenvectors[token_idx,:]
        else:
            batch_tokens = self.tokenvectors(token_idx)
        return(batch_tokens)
    
    def get_start_token(self,device):
        if not self.use_embed:
            start_token = self.tokenvectors[-2,:]
        else:
            if self.start_index == None:
                self.start_index=torch.tensor(self.voc_length,dtype=torch.int64).to(device)
            start_token = self.tokenvectors(self.start_index)
        return(start_token)

    def get_end_token(self,device):
        if not self.use_embed:
            end_token = self.tokenvectors[-1,:]
        else:
            if self.end_index == None:
                self.end_index=torch.tensor(self.voc_length+1,dtype=torch.int64).to(device)
            end_token = self.tokenvectors(self.end_index)
        return(end_token) 

class GNN_noshare_v3(nn.Module):
    def __init__(self,num_hops,embed_size,layer_size=[256,128],use_bn=True,init_methd=None):
        super(GNN_noshare_v3,self).__init__()

        self.num_hops = num_hops
        self.embed_size = embed_size
        self.use_bn = use_bn
        self.init_methd=init_methd

        if self.num_hops>0:
            self.MLP_V = MLP(self.embed_size,layer_size,self.use_bn,self.init_methd)
        if self.num_hops>1:
            self.MLP_E = MLP(1,layer_size,self.use_bn,self.init_methd)
            
            self.MLP_p_list = nn.ModuleList([MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd) for i in range(self.num_hops-1)])
            self.MLP_c_list = nn.ModuleList([MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd) for i in range(self.num_hops-1)])
            self.MLP_aggr_list = nn.ModuleList([MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd) for i in range(self.num_hops-1)])

    def forward(self,batch_token,edge_p_node,edge_c_node,edge_p_indicate,edge_c_indicate,
                     p_mask,c_mask,start_token,end_token):
        # print(batch_token,flush=True)

        Num_node = batch_token.shape[0]
        if self.num_hops>0:
            hidden_state = self.MLP_V(batch_token)
        else:
            hidden_state = batch_token
       
        #compute the edge initial embed
        if self.num_hops>1:
            edge_p = self.MLP_E(edge_p_indicate.view(-1,1))
            edge_c = self.MLP_E(edge_c_indicate.view(-1,1))

        #main gnn loops
        for hop in range(self.num_hops-1):
            #gather node
            edge_p_node_batch = hidden_state[edge_p_node,:]
            edge_c_node_batch = hidden_state[edge_c_node,:]

            #concat the input
            edge_p_input = torch.cat([edge_c_node_batch,edge_p_node_batch, edge_p],-1)
            edge_c_input = torch.cat([edge_p_node_batch,edge_c_node_batch, edge_c],-1)
            
            S_p = self.MLP_p_list[hop](edge_p_input)
            S_c = self.MLP_c_list[hop](edge_c_input)

            S_p = torch_scatter.scatter_mean(S_p,edge_p_node,dim=0,dim_size=Num_node)
            S_c = torch_scatter.scatter_mean(S_c,edge_c_node,dim=0,dim_size=Num_node)

            S_p=S_p + p_mask.view(-1,1)*start_token
            S_c=S_c + c_mask.view(-1,1)*end_token

            x_aggr = torch.cat([hidden_state,S_p,S_c],-1)
            hidden_state = hidden_state+self.MLP_aggr_list[hop](x_aggr)
            hidden_state = F.relu(hidden_state)

        return(hidden_state)

class GNN_noshare_v2(nn.Module):
    def __init__(self,num_hops,embed_size,layer_size=[256,128],use_bn=True,init_methd=None):
        super(GNN_noshare_v2,self).__init__()

        self.num_hops = num_hops
        self.embed_size = embed_size
        self.use_bn = use_bn
        self.init_methd=init_methd

        if self.num_hops>0:
            self.MLP_V = MLP(self.embed_size,layer_size,self.use_bn,self.init_methd)
        if self.num_hops>1:
            self.MLP_E = MLP(1,layer_size,self.use_bn,self.init_methd)
            
            self.MLP_p_list = nn.ModuleList([MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd) for i in range(self.num_hops-1)])
            self.MLP_c_list = nn.ModuleList([MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd) for i in range(self.num_hops-1)])
            self.MLP_aggr_list = nn.ModuleList([MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd) for i in range(self.num_hops-1)])

    def forward(self,batch_token,edge_p_node,edge_c_node,edge_p_indicate,edge_c_indicate,
                     p_mask,c_mask,start_token,end_token):
        # print(batch_token,flush=True)

        Num_node = batch_token.shape[0]
        if self.num_hops>0:
            hidden_state = batch_token+self.MLP_V(batch_token)
        else:
            hidden_state = batch_token
       
        #compute the edge initial embed
        if self.num_hops>1:
            edge_p = self.MLP_E(edge_p_indicate.view(-1,1))
            edge_c = self.MLP_E(edge_c_indicate.view(-1,1))

        #main gnn loops
        for hop in range(self.num_hops-1):
            #gather node
            edge_p_node_batch = hidden_state[edge_p_node,:]
            edge_c_node_batch = hidden_state[edge_c_node,:]

            #concat the input
            edge_p_input = torch.cat([edge_c_node_batch,edge_p_node_batch, edge_p],-1)
            edge_c_input = torch.cat([edge_p_node_batch,edge_c_node_batch, edge_c],-1)
            
            S_p = self.MLP_p_list[hop](edge_p_input)
            S_c = self.MLP_c_list[hop](edge_c_input)

            S_p = torch_scatter.scatter_mean(S_p,edge_p_node,dim=0,dim_size=Num_node)
            S_c = torch_scatter.scatter_mean(S_c,edge_c_node,dim=0,dim_size=Num_node)

            S_p=S_p + p_mask.view(-1,1)*start_token
            S_c=S_c + c_mask.view(-1,1)*end_token

            x_aggr = torch.cat([hidden_state,S_p,S_c],-1)
            hidden_state = hidden_state+self.MLP_aggr_list[hop](x_aggr)
            hidden_state = F.relu(hidden_state)

        return(hidden_state)

class GNN_noshare(nn.Module):
    def __init__(self,num_hops,embed_size,layer_size=[256,128],use_bn=True,init_methd=None):
        super(GNN_noshare,self).__init__()

        self.num_hops = num_hops
        self.embed_size = embed_size
        self.use_bn = use_bn
        self.init_methd=init_methd

        if self.num_hops>0:
            self.MLP_V = MLP(self.embed_size,layer_size,self.use_bn,self.init_methd)
        if self.num_hops>1:
            self.MLP_E = MLP(1,layer_size,self.use_bn,self.init_methd)
            
            self.MLP_p_list = nn.ModuleList([MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd) for i in range(self.num_hops-1)])
            self.MLP_c_list = nn.ModuleList([MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd) for i in range(self.num_hops-1)])
            self.MLP_aggr = MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd)

    def forward(self,batch_token,edge_p_node,edge_c_node,edge_p_indicate,edge_c_indicate,
                     p_mask,c_mask,start_token,end_token):
        # print(batch_token,flush=True)

        Num_node = batch_token.shape[0]
        if self.num_hops>0:
            hidden_state = batch_token+self.MLP_V(batch_token)
        else:
            hidden_state = batch_token
       
        #compute the edge initial embed
        if self.num_hops>1:
            edge_p = self.MLP_E(edge_p_indicate.view(-1,1))
            edge_c = self.MLP_E(edge_c_indicate.view(-1,1))

        #main gnn loops
        for hop in range(self.num_hops-1):
            #gather node
            edge_p_node_batch = hidden_state[edge_p_node,:]
            edge_c_node_batch = hidden_state[edge_c_node,:]

            #concat the input
            edge_p_input = torch.cat([edge_c_node_batch,edge_p_node_batch, edge_p],-1)
            edge_c_input = torch.cat([edge_p_node_batch,edge_c_node_batch, edge_c],-1)
            
            S_p = self.MLP_p_list[hop](edge_p_input)
            S_c = self.MLP_c_list[hop](edge_c_input)

            S_p = torch_scatter.scatter_mean(S_p,edge_p_node,dim=0,dim_size=Num_node)
            S_c = torch_scatter.scatter_mean(S_c,edge_c_node,dim=0,dim_size=Num_node)

            S_p=S_p + p_mask.view(-1,1)*start_token
            S_c=S_c + c_mask.view(-1,1)*end_token

            x_aggr = torch.cat([hidden_state,S_p,S_c],-1)
            hidden_state = hidden_state+self.MLP_aggr(x_aggr)
            hidden_state = F.relu(hidden_state)

        return(hidden_state)

class GNN_res(nn.Module):
    def __init__(self,num_hops,embed_size,layer_size=[256,128],use_bn=True,init_methd=None):
        super(GNN_res,self).__init__()

        self.num_hops = num_hops
        self.embed_size = embed_size
        self.use_bn = use_bn
        self.init_methd=init_methd

        if self.num_hops>0:
            self.MLP_V = MLP(self.embed_size,layer_size,self.use_bn,self.init_methd)
        if self.num_hops>1:
            self.MLP_E = MLP(1,layer_size,self.use_bn,self.init_methd)
            self.MLP_p = MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd)
            self.MLP_c = MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd)
            self.MLP_aggr = MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd)
    
    def forward(self,batch_token,edge_p_node,edge_c_node,edge_p_indicate,edge_c_indicate,
                     p_mask,c_mask,start_token,end_token):
        # print(batch_token,flush=True)

        Num_node = batch_token.shape[0]
        if self.num_hops>0:
            hidden_state = batch_token+self.MLP_V(batch_token)
        else:
            hidden_state = batch_token
       
        #compute the edge initial embed
        if self.num_hops>1:
            edge_p = self.MLP_E(edge_p_indicate.view(-1,1))
            edge_c = self.MLP_E(edge_c_indicate.view(-1,1))

        #main gnn loops
        for hop in range(self.num_hops-1):
            #gather node
            edge_p_node_batch = hidden_state[edge_p_node,:]
            edge_c_node_batch = hidden_state[edge_c_node,:]

            #concat the input
            edge_p_input = torch.cat([edge_c_node_batch,edge_p_node_batch, edge_p],-1)
            edge_c_input = torch.cat([edge_p_node_batch,edge_c_node_batch, edge_c],-1)
            
            S_p = self.MLP_p(edge_p_input)
            S_c = self.MLP_c(edge_c_input)

            S_p = torch_scatter.scatter_mean(S_p,edge_p_node,dim=0,dim_size=Num_node)
            S_c = torch_scatter.scatter_mean(S_c,edge_c_node,dim=0,dim_size=Num_node)

            S_p=S_p + p_mask.view(-1,1)*start_token
            S_c=S_c + c_mask.view(-1,1)*end_token

            x_aggr = torch.cat([hidden_state,S_p,S_c],-1)
            hidden_state = hidden_state+self.MLP_aggr(x_aggr)
            hidden_state = F.relu(hidden_state)

        return(hidden_state)

class MLP(nn.Module):
    def __init__(self,input_size,layer_size=[256,128],use_bn=True,init_methd=None):
        super(MLP,self).__init__()
        self.use_bn = use_bn
        self.init_methd=init_methd

        if self.use_bn:
            self.model = nn.Sequential(
                nn.Linear(input_size,layer_size[0]),
                nn.LayerNorm(layer_size[0]),
                nn.ReLU(),
                nn.Linear(layer_size[0],layer_size[1]),
                nn.LayerNorm(layer_size[1]),
                nn.ReLU(),
                nn.Dropout(p=p1)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_size,layer_size[0]),
                nn.ReLU(),
                nn.Linear(layer_size[0],layer_size[1]),
                nn.ReLU(),
                nn.Dropout(p=p1)
            )
        
        for m in self.model:
            if isinstance(m,nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                print("Normalize layer in mlp",flush=True)
            elif isinstance(m,nn.Linear):
                init.zeros_(m.bias)
                if self.init_methd==None or self.init_methd=="uniform":
                    init.uniform_(m.weight, a=init_a, b=init_b)
                elif self.init_methd=="xavier":
                    init.xavier_normal_(m.weight)
                else:
                    raise RuntimeError("unknow init type")
            elif isinstance(m,nn.ReLU) or isinstance(m,nn.Dropout):
                pass
            else:
                raise RuntimeError("unknow parameter type")
    
    def forward(self,input):
        return(self.model(input))

class GNN(nn.Module):
    def __init__(self,num_hops,embed_size,layer_size=[256,128],use_bn=True,init_methd=None):
        super(GNN,self).__init__()

        self.num_hops = num_hops
        self.embed_size = embed_size
        self.use_bn = use_bn
        self.init_methd=init_methd

        if self.num_hops>0:
            self.MLP_V = MLP(self.embed_size,layer_size,self.use_bn,self.init_methd)
        if self.num_hops>1:
            self.MLP_E = MLP(1,layer_size,self.use_bn,self.init_methd)
            self.MLP_p = MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd)
            self.MLP_c = MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd)
            self.MLP_aggr = MLP(3*self.embed_size,layer_size,self.use_bn,self.init_methd)

    
    def forward(self,batch_token,edge_p_node,edge_c_node,edge_p_indicate,edge_c_indicate,
                     p_mask,c_mask,start_token,end_token):
        # print(batch_token,flush=True)

        Num_node = batch_token.shape[0]
        if self.num_hops>0:
            hidden_state = self.MLP_V(batch_token)
        else:
            hidden_state = batch_token
       
        #compute the edge initial embed
        if self.num_hops>1:
            edge_p = self.MLP_E(edge_p_indicate.view(-1,1))
            edge_c = self.MLP_E(edge_c_indicate.view(-1,1))

        #main gnn loops
        for hop in range(self.num_hops-1):
            #gather node
            edge_p_node_batch = hidden_state[edge_p_node,:]
            edge_c_node_batch = hidden_state[edge_c_node,:]

            #concat the input
            edge_p_input = torch.cat([edge_c_node_batch,edge_p_node_batch, edge_p],-1)
            edge_c_input = torch.cat([edge_p_node_batch,edge_c_node_batch, edge_c],-1)
            
            S_p = self.MLP_p(edge_p_input)
            S_c = self.MLP_c(edge_c_input)

            S_p = torch_scatter.scatter_mean(S_p,edge_p_node,dim=0,dim_size=Num_node)
            S_c = torch_scatter.scatter_mean(S_c,edge_c_node,dim=0,dim_size=Num_node)

            S_p=S_p + p_mask.view(-1,1)*start_token
            S_c=S_c + c_mask.view(-1,1)*end_token

            x_aggr = torch.cat([hidden_state,S_p,S_c],-1)
            hidden_state = hidden_state+self.MLP_aggr(x_aggr)
            hidden_state = F.relu(hidden_state)

        return(hidden_state)

class Neck_exp(nn.Module):
    def __init__(self,embed_size,filters=[512,1024],use_bn=True,init_methd=None):
        super(Neck_exp,self).__init__()

        self.embed_size = embed_size
        self.filters=filters
        self.use_bn = use_bn
        self.init_methd=init_methd
       
        if self.use_bn:
            self.model = nn.Sequential(
                nn.Linear(self.embed_size,self.filters[0]),
                nn.LayerNorm(self.filters[0]),
                nn.ReLU(),
                nn.Linear(self.filters[0],self.filters[1]),
                nn.LayerNorm(self.filters[1])
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(self.embed_size,self.filters[0]),
                nn.ReLU(),
                nn.Linear(self.filters[0],self.filters[1])
            )
        
        for m in self.model:
            if isinstance(m,nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                print("Normalize layer in neck",flush=True)
            elif isinstance(m,nn.Linear):
                init.zeros_(m.bias)
                if self.init_methd==None or self.init_methd=="uniform":
                    init.uniform_(m.weight, a=init_a, b=init_b)
                elif self.init_methd=="xavier":
                    init.xavier_normal_(m.weight)
                else:
                    raise RuntimeError("unknow init type")
            elif isinstance(m,nn.ReLU) or isinstance(m,nn.Dropout):
                pass
            else:
                raise RuntimeError("unknow parameter type")

    def forward(self,batch_gnn_embed,gather_idx,num_graph):
        # print(batch_gnn_embed,flush=True)
        batch_neck_allnode = self.model(batch_gnn_embed)

        batch_neck_graph,idx_ = torch_scatter.scatter_max(batch_neck_allnode,gather_idx,dim=0,dim_size=num_graph)
        return(batch_neck_graph)


class Neck_bow(nn.Module):
    def __init__(self):
        super(Neck_bow,self).__init__()

    def forward(self,batch_gnn_embed,gather_idx,num_graph):
        # print(batch_gnn_embed,flush=True)
        batch_neck_graph,idx_ = torch_scatter.scatter_max(batch_gnn_embed,gather_idx,dim=0,dim_size=num_graph)

        return(batch_neck_graph)


class Neck(nn.Module):
    def __init__(self,embed_size,filters=[512,1024],use_bn = True,init_methd=None):
        super(Neck,self).__init__()

        self.embed_size = embed_size
        self.filters=filters
        self.use_bn = use_bn
        self.init_methd=init_methd
        
        if self.use_bn:
            self.model = nn.Sequential(
                nn.Linear(self.embed_size,self.filters[0]),
                nn.LayerNorm(self.filters[0]),
                nn.ReLU(),
                nn.Linear(self.filters[0],self.filters[1]),
                nn.LayerNorm(self.filters[1]),
                nn.ReLU()
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(self.embed_size,self.filters[0]),
                nn.ReLU(),
                nn.Linear(self.filters[0],self.filters[1]),
                nn.ReLU()
            )

        for m in self.model:
            if isinstance(m,nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                print("Normalize layer in neck",flush=True)
            elif isinstance(m,nn.Linear):
                init.zeros_(m.bias)
                if self.init_methd==None or self.init_methd=="uniform":
                    init.uniform_(m.weight, a=init_a, b=init_b)
                elif self.init_methd=="xavier":
                    init.xavier_normal_(m.weight)
                else:
                    raise RuntimeError("unknow init type")
            elif isinstance(m,nn.ReLU) or isinstance(m,nn.Dropout):
                pass
            else:
                raise RuntimeError("unknow parameter type")

    def forward(self,batch_gnn_embed,gather_idx,num_graph):
        # print(batch_gnn_embed,flush=True)
        batch_neck_allnode = self.model(batch_gnn_embed)

        batch_neck_graph,idx_ = torch_scatter.scatter_max(batch_neck_allnode,gather_idx,dim=0,dim_size=num_graph)
        # print(batch_neck_allnode,flush=True)
        # print(batch_neck_graph,flush=True) 
        # print(idx_,flush=True)
        return(batch_neck_graph)
        

class Tactic_Classifier(nn.Module):
    def __init__(self,input_size = 1024,
                      hidden_layers=[512,256,41],
                      use_bn = True,
                      init_methd=None):
        super(Tactic_Classifier,self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.use_bn = use_bn
        self.init_methd = init_methd
        if self.use_bn:
            self.model = nn.Sequential(
                nn.Dropout(p=p2),
                nn.Linear(self.input_size,self.hidden_layers[0]),
                nn.LayerNorm(self.hidden_layers[0]),
                nn.ReLU(),
                nn.Dropout(p=p2),
                nn.Linear(self.hidden_layers[0],self.hidden_layers[1]),
                nn.LayerNorm(self.hidden_layers[1]),
                nn.ReLU(),
                nn.Dropout(p=p2),
                nn.Linear(self.hidden_layers[1],self.hidden_layers[2])
            )
        else:
            self.model = nn.Sequential(
                nn.Dropout(p=p2),
                nn.Linear(self.input_size,self.hidden_layers[0]),
                nn.ReLU(),
                nn.Dropout(p=p2),
                nn.Linear(self.hidden_layers[0],self.hidden_layers[1]),
                nn.ReLU(),
                nn.Dropout(p=p2),
                nn.Linear(self.hidden_layers[1],self.hidden_layers[2])
            )

        for m in self.model:
            if isinstance(m,nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                print("Normalize layer in tac",flush=True)
            elif isinstance(m,nn.Linear):
                init.zeros_(m.bias)
                if self.init_methd==None or self.init_methd=="uniform":
                    init.uniform_(m.weight, a=init_a, b=init_b)
                elif self.init_methd=="xavier":
                    init.xavier_normal_(m.weight)
                else:
                    raise RuntimeError("unknow init type")
            elif isinstance(m,nn.ReLU) or isinstance(m,nn.Dropout):
                pass
            else:
                raise RuntimeError("unknow parameter type")
    
    def forward(self,batch_goal_neck):
        return(self.model(batch_goal_neck))

class Theom_logit(nn.Module):
    def __init__(self,embed_size=1024,
                      hidden_layers=[1024,512,1],
                      use_bn = True,
                      init_methd = None):
        super(Theom_logit,self).__init__()
        self.embed_size = embed_size
        self.hidden_layers = hidden_layers
        self.use_bn = use_bn
        self.init_methd = init_methd
        
        if self.use_bn:
            self.model = nn.Sequential(
                nn.Dropout(p=p2),
                nn.Linear(self.embed_size*3,self.hidden_layers[0]),
                nn.LayerNorm(self.hidden_layers[0]),
                nn.ReLU(),
                nn.Dropout(p=p2),
                nn.Linear(self.hidden_layers[0],self.hidden_layers[1]),
                nn.LayerNorm(self.hidden_layers[1]),
                nn.ReLU(),
                nn.Dropout(p=p2),
                nn.Linear(self.hidden_layers[1],self.hidden_layers[2])
            )
        else:
                self.model = nn.Sequential(
                nn.Dropout(p=p2),
                nn.Linear(self.embed_size*3,self.hidden_layers[0]),
                nn.ReLU(),
                nn.Dropout(p=p2),
                nn.Linear(self.hidden_layers[0],self.hidden_layers[1]),
                nn.ReLU(),
                nn.Dropout(p=p2),
                nn.Linear(self.hidden_layers[1],self.hidden_layers[2])
            )

        for m in self.model:
            if isinstance(m,nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                print("Normalize layer in thm",flush=True)
            elif isinstance(m,nn.Linear):
                init.zeros_(m.bias)
                if self.init_methd==None or self.init_methd=="uniform":
                    init.uniform_(m.weight, a=init_a, b=init_b)
                elif self.init_methd=="xavier":
                    init.xavier_normal_(m.weight)
                else:
                    raise RuntimeError("unknow init type")
            elif isinstance(m,nn.ReLU) or isinstance(m,nn.Dropout):
                pass
            else:
                raise RuntimeError("unknow parameter type")

    def forward(self,batch_goal,batch_thm):
        batch_goal_reshaped = batch_goal.view(-1,1,self.embed_size)
        batch_thm_reshaped = batch_thm.view(1,-1,self.embed_size)

        batch_element_mul = batch_goal_reshaped*batch_thm_reshaped
        num_goal,num_thm,_ = batch_element_mul.shape

        batch_goal_repeat = torch.cat(num_thm*[batch_goal_reshaped],1)
        batch_thm_repeat = torch.cat(num_goal*[batch_thm_reshaped],0)

        goal_thm = torch.cat([batch_goal_repeat,batch_thm_repeat,batch_element_mul],2)
        goal_thm = goal_thm.view(-1,self.embed_size*3)
        socre_logits = self.model(goal_thm).view(num_goal,num_thm,-1)
        return(socre_logits)            
        
class GNN_net(nn.Module):
    def __init__(self,args):
        super(GNN_net,self).__init__()
        self.embed_init = args.embed_init if hasattr(args, 'embed_init') else None
        self.weight_init = args.weight_init if hasattr(args, 'weight_init') else None

        self.goal_embed=Tokenstore(args.goal_voc_length,args.goal_voc_embedsize,args.use_embed,args.max_norm,self.embed_init)
        self.thm_embed=Tokenstore(args.thm_voc_length,args.thm_voc_embedsize,args.use_embed,args.max_norm,self.embed_init)

        self.num_hops = args.num_hops

        self.gnn_usebn = args.gnn_usebn if hasattr(args, 'gnn_usebn') else False
        self.neck_usebn = args.neck_usebn if hasattr(args, 'neck_usebn') else False
        self.tac_usebn = args.tac_usebn if hasattr(args, 'tac_usebn') else False
        self.thm_usebn = args.thm_usebn if hasattr(args, 'thm_usebn') else False

        self.mask_token_rate = args.mask_token_rate if hasattr(args,'mask_token_rate') else 0.
        self.mask_token = args.mask_token if hasattr(args,'mask_token') else False


        if hasattr(args, 'gnn_module'):
            if args.gnn_module == "GNN":
                self.GNN_goal = GNN(args.num_hops,args.goal_voc_embedsize,args.gnn_layer_size,self.gnn_usebn,self.weight_init)
                self.GNN_thm = GNN(args.num_hops,args.thm_voc_embedsize,args.gnn_layer_size,self.gnn_usebn,self.weight_init)
            elif args.gnn_module == "GNN_res":
                self.GNN_goal = GNN_res(args.num_hops,args.goal_voc_embedsize,args.gnn_layer_size,self.gnn_usebn,self.weight_init)
                self.GNN_thm = GNN_res(args.num_hops,args.thm_voc_embedsize,args.gnn_layer_size,self.gnn_usebn,self.weight_init)
            elif args.gnn_module == "GNN_noshare":
                self.GNN_goal = GNN_noshare(args.num_hops,args.goal_voc_embedsize,args.gnn_layer_size,self.gnn_usebn,self.weight_init)
                self.GNN_thm = GNN_noshare(args.num_hops,args.thm_voc_embedsize,args.gnn_layer_size,self.gnn_usebn,self.weight_init)
            elif args.gnn_module == "GNN_noshare_v2":
                self.GNN_goal = GNN_noshare_v2(args.num_hops,args.goal_voc_embedsize,args.gnn_layer_size,self.gnn_usebn,self.weight_init)
                self.GNN_thm = GNN_noshare_v2(args.num_hops,args.thm_voc_embedsize,args.gnn_layer_size,self.gnn_usebn,self.weight_init)
            elif args.gnn_module == "GNN_noshare_v3":
                self.GNN_goal = GNN_noshare_v3(args.num_hops,args.goal_voc_embedsize,args.gnn_layer_size,self.gnn_usebn,self.weight_init)
                self.GNN_thm = GNN_noshare_v3(args.num_hops,args.thm_voc_embedsize,args.gnn_layer_size,self.gnn_usebn,self.weight_init)
            else:
                raise RuntimeError('unknown gnn type')
        else:
            self.GNN_goal = GNN(args.num_hops,args.goal_voc_embedsize,args.gnn_layer_size,self.gnn_usebn,self.weight_init)
            self.GNN_thm = GNN(args.num_hops,args.thm_voc_embedsize,args.gnn_layer_size,self.gnn_usebn,self.weight_init)
        
        if hasattr(args, 'neck_module'):
            if args.neck_module == "neck":
                self.neck_goal = Neck(args.gnn_layer_size[-1],args.neck_layer_size,self.neck_usebn,self.weight_init)
                self.neck_thm = Neck(args.gnn_layer_size[-1],args.neck_layer_size,self.neck_usebn,self.weight_init)
            elif args.neck_module == "neck_exp":
                self.neck_goal = Neck_exp(args.gnn_layer_size[-1],args.neck_layer_size,self.neck_usebn,self.weight_init)
                self.neck_thm = Neck_exp(args.gnn_layer_size[-1],args.neck_layer_size,self.neck_usebn,self.weight_init)
            elif args.neck_module == "neck_bow":
                self.neck_goal = Neck_bow()
                self.neck_thm = Neck_bow()                 
            else:
                raise RuntimeError('unknown neck type')
        else:
            self.neck_goal = Neck(args.gnn_layer_size[-1],args.neck_layer_size,self.neck_usebn,self.weight_init)
            self.neck_thm = Neck(args.gnn_layer_size[-1],args.neck_layer_size,self.neck_usebn,self.weight_init)


        self.tactic_head = Tactic_Classifier(args.neck_layer_size[-1],args.tac_layer_size,self.tac_usebn,self.weight_init)
        self.logit_head = Theom_logit(args.neck_layer_size[-1],args.thm_layer_size,self.thm_usebn,self.weight_init)

        self.score_weight = args.score_weight
        self.tactic_weight = args.tactic_weight
        self.auc_weight = args.auc_weight

        self.neg_per_pos = args.neg_per_pos
        self.batch_size = args.batch_size
        self.world_size = args.world_size
        
    def regulizer(self):
        loss=0.
        for p in self.parameters():
            loss+=p.norm()
        return(loss)

    def aucloss(self,logits):
        #compute pos 
        num_goal,num_thm,_ = logits.shape
        offset_num = num_thm//num_goal
        assert num_thm%num_goal==0
        logits_flat = logits.view(-1)
        tmp = np.zeros([logits_flat.shape[0]],dtype=np.float32)
        tmp_goal = np.zeros([logits_flat.shape[0]],dtype=np.float32)
        for idx in range(num_goal):
            tmp[idx*num_thm+idx*offset_num]=1.
            tmp_goal[idx*num_thm:idx*num_thm+idx*offset_num]=1.
            tmp_goal[idx*num_thm+idx*offset_num+1:idx*num_thm+num_thm]=1.

        device = logits.device

        choose_pos = torch.tensor(tmp>0.5).to(device)
        choose_neg = torch.tensor(tmp<0.5).to(device)
        choose_neg_goal = torch.tensor(tmp_goal>0.5).to(device)


        pos_logits = logits_flat[choose_pos].view(-1,1)
        neg_logits = logits_flat[choose_neg].view(1,-1)

        pos_logits_goal = pos_logits.expand(-1,num_thm-1)
        neg_logits_goal = logits_flat[choose_neg_goal].view(-1,num_thm-1)
        
        delta = pos_logits-neg_logits
        delta_goal = pos_logits_goal-neg_logits_goal

        delta = torch.clamp(delta, min=-80., max=800.)
        delta_goal = torch.clamp(delta_goal, min=-80., max=800.)

        auc_loss = torch.log((torch.exp(-delta)+1)).view(-1)
        auc_loss_goal = torch.log((torch.exp(-delta_goal)+1)).view(-1)
        
        auc_loss_all = torch.mean(torch.cat((auc_loss,auc_loss_goal),-1))

        return(auc_loss_all)

    def loss(self,tactic_scores,logits,gt_tactic):
        batch_current = gt_tactic.shape[0]
        tmp = np.zeros([batch_current,batch_current*(self.neg_per_pos+1),1],dtype=np.float32)
        for i in range(batch_current):
            tmp[i,i*(self.neg_per_pos+1),0]=1.
        
        device = tactic_scores.device
        logits_gt = torch.tensor(tmp).to(device)

        tactic_loss =F.cross_entropy(tactic_scores,gt_tactic,reduction='mean')

        score_loss = F.binary_cross_entropy(torch.sigmoid(logits),logits_gt,reduction='mean')

        auc_loss = self.aucloss(logits)

        score_loss = score_loss*self.score_weight
        tactic_loss = tactic_loss*self.tactic_weight
        auc_loss = auc_loss*self.auc_weight

        reg_loss = 0.
        return(tactic_loss,score_loss,auc_loss,reg_loss)
    
    def forward(self,input):
        if self.mask_token:
            # print("mask used",flush=True)
            mask = np.random.uniform(0.,1)<0.5
            if mask:
                device = input['goal_token'].device
                goal_num = input['goal_token'].shape[0]
                thm_num = input['thm_token'].shape[0]

                idx_goal = int(goal_num*self.mask_token_rate)
                idx_thm = int(thm_num*self.mask_token_rate)
                mask_id_goal = torch.randperm(goal_num,device=device)[:idx_goal]
                mask_id_thm = torch.randperm(thm_num,device=device)[:idx_thm]

                input['goal_token'][mask_id_goal]=1
                input['thm_token'][mask_id_thm]=1

        batch_goal_token = self.goal_embed(input['goal_token'])
        batch_thm_token = self.thm_embed(input['thm_token'])

        device = batch_goal_token.device

        goal_start_token = self.goal_embed.get_start_token(device).view(1,-1)
        goal_end_token = self.goal_embed.get_end_token(device).view(1,-1)

        thm_start_token = self.thm_embed.get_start_token(device).view(1,-1)
        thm_end_token = self.thm_embed.get_end_token(device).view(1,-1)

        goal_hidden_state = self.GNN_goal(batch_goal_token,
                                            input['goal_edge_p_node'],input['goal_edge_c_node'],
                                            input['goal_edge_p_indicate'],input['goal_edge_c_indicate'],
                                            input['goal_p_mask'],input['goal_c_mask'],goal_start_token,goal_end_token)

        thm_hidden_state = self.GNN_thm(batch_thm_token,
                                            input['thm_edge_p_node'],input['thm_edge_c_node'],
                                            input['thm_edge_p_indicate'],input['thm_edge_c_indicate'],
                                            input['thm_p_mask'],input['thm_c_mask'],thm_start_token,thm_end_token)

        goal_neck_state = self.neck_goal(goal_hidden_state,input['idx_gather_goal'],input['length_list_g'].shape[0])
        thm_neck_state = self.neck_thm(thm_hidden_state,input['idx_gather_thm'],input['length_list_t'].shape[0])

        tactic_scores = self.tactic_head(goal_neck_state)
        logits = self.logit_head(goal_neck_state,thm_neck_state)

        if self.training:
            return(self.loss(tactic_scores,logits,input['tac_id']))
        else:
            return({'tactic_scores':F.softmax(tactic_scores,dim=1),'logits':logits})


        

