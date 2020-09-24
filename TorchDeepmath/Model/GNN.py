import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_scatter
import numpy as np
import torch.distributed as dist

init_a = -0.1
init_b = 0.1

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
                      initializer=None):
        super(Tokenstore,self).__init__()
        
        self.voc_length = voc_length
        self.voc_embedsize = voc_embedsize
        self.register_parameter(name='tokenvectors', param=torch.nn.Parameter(torch.zeros([self.voc_length+2,self.voc_embedsize],
                                                                                           dtype=torch.float32)))

        if initializer==None:
            init.uniform_(self.tokenvectors, a=init_a, b=init_b)
    
    def forward(self,token_idx):
        batch_tokens = self.tokenvectors[token_idx,:]
        return(batch_tokens)

class MLP(nn.Module):
    def __init__(self,input_size,layer_size=[256,128]):
        super(MLP,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size,layer_size[0]),
            nn.ReLU(),
            nn.Linear(layer_size[0],layer_size[1]),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        
        for p in self.model.parameters():
            init.uniform_(p, a=init_a, b=init_b)
        
    
    def forward(self,input):
        return(self.model(input))

class GNN(nn.Module):
    def __init__(self,num_hops,embed_size):
        super(GNN,self).__init__()

        self.num_hops = num_hops
        self.embed_size = embed_size

        self.MLP_V = MLP(self.embed_size)
        self.MLP_E = MLP(1)
        self.MLP_p = MLP(3*self.embed_size)
        self.MLP_c = MLP(3*self.embed_size)
        self.MLP_aggr = MLP(3*self.embed_size)
    
    def forward(self,batch_token,self_idx_batch,parent_idx_batch,
                     root_mask,leaf_mask,start_token,end_token):
        # print(batch_token,flush=True)

        Num_node = batch_token.shape[0]
        Num_edge = self_idx_batch.shape[0]
        #compute the edge initial embed
        device = batch_token.device
        edge_in = self.MLP_E(torch.tensor([[1.]],dtype=torch.float32).to(device))
        edge_out = self.MLP_E(torch.tensor([[0.]],dtype=torch.float32).to(device))

        edge_in = edge_in.expand(Num_edge,-1)
        edge_out = edge_out.expand(Num_edge,-1)
        hidden_state = self.MLP_V(batch_token)
        # print(batch_token,flush=True)
        # print(hidden_state,flush=True)
        
        #main gnn loops
        for hop in range(self.num_hops):
            #gather self
            node_self_batch = hidden_state[self_idx_batch,:]
            node_parent_batch = hidden_state[parent_idx_batch,:]
     
            #concat the input
            input_parent_batch = torch.cat([node_parent_batch,node_self_batch, edge_out],-1)
            input_child_batch = torch.cat([node_self_batch,node_parent_batch, edge_in],-1)

            
            S_p = self.MLP_p(input_parent_batch)
            S_c = self.MLP_c(input_child_batch)

            S_p = torch_scatter.scatter_mean(S_p,self_idx_batch,dim=0,dim_size=Num_node)
            S_c = torch_scatter.scatter_mean(S_c,parent_idx_batch,dim=0,dim_size=Num_node)
            # print(parent_idx_batch,flush=True)
            # print(self_idx_batch,flush=True)
            # print(parent_idx_batch.shape[0],flush=True)
            S_p=S_p + root_mask.view(-1,1)*start_token
            S_c=S_c + leaf_mask.view(-1,1)*end_token

            x_aggr = torch.cat([hidden_state,S_p,S_c],-1)
            hidden_state = hidden_state+self.MLP_aggr(x_aggr)

        return(hidden_state)

class Neck(nn.Module):
    def __init__(self,embed_size,filters=[512,1024]):
        super(Neck,self).__init__()

        self.embed_size = embed_size
        self.filters=filters
        
       
        self.model = nn.Sequential(
            nn.Linear(self.embed_size,self.filters[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.filters[0],self.filters[1]),
            nn.ReLU()
        )

        for p in self.model.parameters():
            init.uniform_(p, a=init_a, b=init_b)

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
                      hidden_layers=[512,256,41]):
        super(Tactic_Classifier,self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers

        self.model = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[0],self.hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_layers[1],self.hidden_layers[2])
        )

        for p in self.model.parameters():
            init.uniform_(p, a=init_a, b=init_b)
    
    def forward(self,batch_goal_neck):
        return(self.model(batch_goal_neck))

class Theom_logit(nn.Module):
    def __init__(self,embed_size=1024,
                      hidden_layers=[1024,512,1]):
        super(Theom_logit,self).__init__()
        self.embed_size = embed_size
        self.hidden_layers = hidden_layers
      
        self.model = nn.Sequential(
            nn.Linear(self.embed_size*3,self.hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_layers[0],self.hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[1],self.hidden_layers[2])
        )

        for p in self.model.parameters():
            init.uniform_(p, a=init_a, b=init_b)

    def forward(self,batch_goal,batch_thm):
        batch_goal_reshaped = batch_goal.view(-1,1,self.embed_size)
        batch_thm_reshaped = batch_thm.view(1,-1,self.embed_size)

        batch_element_mul = batch_goal_reshaped*batch_thm_reshaped
        num_goal,num_thm,_ = batch_element_mul.shape

        batch_goal_repeat = torch.cat(num_thm*[batch_goal_reshaped],1)
        batch_thm_repeat = torch.cat(num_goal*[batch_thm_reshaped],0)

        goal_thm = torch.cat([batch_goal_repeat,batch_thm_repeat,batch_element_mul],2)
        socre_logits = self.model(goal_thm)
        return(socre_logits)            
        
class GNN_net(nn.Module):
    def __init__(self,params):
        super(GNN_net,self).__init__()

        self.goal_embed=Tokenstore(params['goal_voc_length'],params['goal_voc_embedsize'])
        self.thm_embed=Tokenstore(params['thm_voc_length'],params['thm_voc_embedsize'])

        self.GNN_goal = GNN(params['num_hops'],params['goal_voc_embedsize'])
        self.GNN_thm = GNN(params['num_hops'],params['thm_voc_embedsize'])

        self.neck_goal = Neck(params['goal_voc_embedsize'])
        self.neck_thm = Neck(params['thm_voc_embedsize'])

        self.tactic_head = Tactic_Classifier()
        self.logit_head = Theom_logit()

        self.score_weight = params['score_weight']
        self.tactic_weight = params['tactic_weight']
        self.auc_weight = params['auc_weight']

        self.neg_per_pos = params['neg_per_pos']
        self.bactch_size = params['bactch_size']
        self.word_size = params['word_size']

        #pre compute the gt for logits
        # tmp = np.zeros([self.bactch_size,self.bactch_size*(params['neg_per_pos']+1),1],dtype=np.int64)
        # for i in range(self.bactch_size):
        #     tmp[i,i*(params['neg_per_pos']+1),0]=1

        # self.logits_gt = torch.tensor(tmp)

        #get auc pos gather idx
        
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
        for idx in range(num_goal):
            tmp[idx*num_thm+idx*offset_num]=1.
        
        device = logits.device

        choose_pos = torch.tensor(tmp>0.5).to(device)
        choose_neg = torch.tensor(tmp<0.5).to(device)

        pos_logits = logits_flat[choose_pos].view(-1,1)
        neg_logits = logits_flat[choose_neg].view(1,-1)
        
        delta = pos_logits-neg_logits
        #delta = F.leaky_relu(pos_logits-neg_logits, negative_slope=0.01, inplace=False)

        delta = torch.clamp(delta, min=-80., max=80.)
        auc_loss = torch.mean(torch.log((torch.exp(-delta)+1)))
        # auc_loss = torch.mean(-torch.log(torch.sigmoid(delta)*(1-1e-4)+1e-20))

        #raise positive logits by force
        # auc_loss+= torch.mean(1-torch.sigmoid(pos_logits))
        # auc_loss+= torch.mean(torch.sigmoid(neg_logits))*0.2

        return(auc_loss)

    def loss(self,tactic_scores,logits,gt_tactic):
        batch_current = gt_tactic.shape[0]
        tmp = np.zeros([batch_current,batch_current*(self.neg_per_pos+1),1],dtype=np.float32)
        for i in range(batch_current):
            tmp[i,i*(self.neg_per_pos+1),0]=1.
        
        device = tactic_scores.device
        logits_gt = torch.tensor(tmp).to(device)
        # print(tactic_scores,flush=True)

        tactic_loss =F.cross_entropy(tactic_scores,gt_tactic,reduction='mean')
        # batch,channel = tactic_scores.shape

        # gt_numpy = np.zeros([batch,channel],dtype=np.float32)
        # for b in range(batch):
        #     gt_numpy[b,gt_tactic[b].item()]=1.
        # gt_torch = torch.tensor(gt_numpy).to(device)
        # tactic_loss =F.binary_cross_entropy(torch.sigmoid(tactic_scores),gt_torch,reduction='mean')

        score_loss = F.binary_cross_entropy(torch.sigmoid(logits),logits_gt,reduction='mean')

        auc_loss = self.aucloss(logits)

        score_loss = score_loss*self.score_weight
        tactic_loss = tactic_loss*self.tactic_weight
        auc_loss = auc_loss*self.auc_weight

        # print(tactic_loss,flush=True)
        # loss = 0.
        # loss = score_loss+tactic_loss+auc_loss
        # reg_loss =self.regulizer()*1e-4

        reg_loss = 0.
        return(tactic_loss,score_loss,auc_loss,reg_loss)
    
    def forward(self,input):
        batch_goal_token = self.goal_embed(input['goal_token'])
        batch_thm_token = self.thm_embed(input['thm_token'])

        goal_start_token = self.goal_embed.tokenvectors[-2,:].view(1,-1)
        goal_end_token = self.goal_embed.tokenvectors[-1,:].view(1,-1)

        thm_start_token = self.thm_embed.tokenvectors[-2,:].view(1,-1)
        thm_end_token = self.thm_embed.tokenvectors[-1,:].view(1,-1)
        # print(type(batch_goal_token),flush=True)
        goal_hidden_state = self.GNN_goal(batch_goal_token,input['goal_self_index_p'],
                                          input['goal_parent_idx'],input['goal_root_mask'],
                                          input['goal_leaf_mask'],goal_start_token,goal_end_token)

        thm_hidden_state = self.GNN_thm(batch_thm_token,input['thm_self_index_p'],
                                        input['thm_parent_idx'],input['thm_root_mask'],
                                        input['thm_leaf_mask'],thm_start_token,thm_end_token)

        # thm_hidden_state = self.GNN_goal(batch_thm_token,input['thm_self_index_p'],
        #                                 input['thm_parent_idx'],input['thm_root_mask'],
        #                                 input['thm_leaf_mask'],thm_start_token,thm_end_token)


        goal_neck_state = self.neck_goal(goal_hidden_state,input['idx_gather_goal'],input['length_list_g'].shape[0])
        thm_neck_state = self.neck_thm(thm_hidden_state,input['idx_gather_thm'],input['length_list_t'].shape[0])

        tactic_scores = self.tactic_head(goal_neck_state)
        logits = self.logit_head(goal_neck_state,thm_neck_state)

        if self.training:
            return(self.loss(tactic_scores,logits,input['tac_id']))
        else:
            return({'tactic_scores':F.softmax(tactic_scores,dim=1),'logits':logits})


        

