from torch.utils.data import Dataset, DataLoader
from torch import as_tensor
import numpy as np
import random

class GNN_dataset(Dataset):
    def __init__(self,goal_path,neg_thm_path,params):

        self.goal_list = np.load(goal_path,allow_pickle=True).tolist()
        self.neg_thm_list = np.load(neg_thm_path,allow_pickle=True).tolist()

        self.len_goal = len(self.goal_list)
        self.len_neg_thm = len(self.neg_thm_list)
        self.params = params

    def _sample_one_pos(self,thms_list):
        
        assert len(thms_list)>0

        return(random.sample(thms_list,1))
    
    def _sample_neg(self,hard_neg_thm_list):
        
        if len(hard_neg_thm_list)==0:
            return(random.sample(self.neg_thm_list,self.params['neg_per_pos']))
        elif len(hard_neg_thm_list)<self.params['neg_hard_per_pos']:
            neg_sample = random.sample(self.neg_thm_list,self.params['neg_per_pos']-len(hard_neg_thm_list))
            return(hard_neg_thm_list+neg_sample)
        else:
            neg_sample = random.sample(self.neg_thm_list,self.params['neg_per_pos']-self.params['neg_hard_per_pos'])
            neg_hard_sample = random.sample(hard_neg_thm_list,self.params['neg_hard_per_pos'])
            return(neg_hard_sample+neg_sample)

    def __getitem__(self,index):
        goal_token = self.goal_list[index]['goal']['token']
        goal_self_index_p = self.goal_list[index]['goal']['self_index_p']
        goal_parent_idx = self.goal_list[index]['goal']['parent_idx']
        goal_root_mask = self.goal_list[index]['goal']['root_mask']
        goal_leaf_mask = self.goal_list[index]['goal']['leaf_mask']

        tac_id = self.goal_list[index]['tac_id']
        # print(type(self.goal_list[index]['thms']),flush=True)
        thm_pos = self._sample_one_pos(self.goal_list[index]['thms'])
        thm_neg = self._sample_neg(self.goal_list[index]['thms_hard_negatives'])
        # print(thm_pos,flush=True)
        # print(thm_neg,flush=True)
        thms = thm_pos+thm_neg
        thm_token_l = []
        thm_self_index_p_l = []
        thm_parent_idx_l = []
        thm_root_mask_l = []
        thm_leaf_mask_l = []
        
        offset = 0
        length_list_t = []
        for thm in thms:
            thm_token_l.append(thm['token'])
            thm_self_index_p_l.append(thm['self_index_p']+offset)
            thm_parent_idx_l.append(thm['parent_idx']+offset)
            thm_root_mask_l.append(thm['root_mask'])
            thm_leaf_mask_l.append(thm['leaf_mask'])
            offset +=thm['token'].shape[0]
            length_list_t.append(thm['token'].shape[0])

        thm_token = np.concatenate(thm_token_l,axis=0)
        thm_self_index_p = np.concatenate(thm_self_index_p_l,axis=0)
        thm_parent_idx = np.concatenate(thm_parent_idx_l,axis=0)
        thm_root_mask = np.concatenate(thm_root_mask_l,axis=0)
        thm_leaf_mask = np.concatenate(thm_leaf_mask_l,axis=0)

        # neg_items = random.sample(self.neg_thm_list,self.params['neg_per_pos'])
        output = {
            'goal_token':goal_token,
            'goal_self_index_p':goal_self_index_p,
            'goal_parent_idx':goal_parent_idx,
            'goal_root_mask':goal_root_mask,
            'goal_leaf_mask':goal_leaf_mask,
            'thm_token':thm_token,
            'thm_self_index_p':thm_self_index_p,
            'thm_parent_idx':thm_parent_idx,
            'thm_root_mask':thm_root_mask,
            'thm_leaf_mask':thm_leaf_mask,
            'tac_id':tac_id,
            'length_list_t':length_list_t
        }
        return(output)

    
    def __len__(self):
        return(self.len_goal)

def get_gather_idx(length_list):
    node_num = np.sum(length_list)
    idx_gather = np.zeros([node_num],dtype=np.int64)
    idx = 0
    offset=0
    for item in length_list:
        idx_gather[offset:item]=idx
        idx+=1
        offset+=item
    return(idx_gather)


def Batch_collect(batch):
    output = {}

    goal_token_l = []
    goal_self_index_p_l = []
    goal_parent_idx_l = []
    goal_root_mask_l =[]
    goal_leaf_mask_l =[]
    thm_token_l =[]
    thm_self_index_p_l = []
    thm_parent_idx_l = []
    thm_root_mask_l =[]
    thm_leaf_mask_l = []
    tac_id_l =[]
    
    offset_g = 0
    offset_t = 0
    
    length_list_g = []
    length_list_t = []
    for b in batch:
        goal_token_l.append(b['goal_token'])
        goal_self_index_p_l.append(b['goal_self_index_p']+offset_g)
        goal_parent_idx_l.append(b['goal_parent_idx']+offset_g)
        goal_root_mask_l.append(b['goal_root_mask'])
        goal_leaf_mask_l.append(b['goal_leaf_mask'])
        thm_token_l.append(b['thm_token'])
        thm_self_index_p_l.append(b['thm_self_index_p']+offset_t)
        thm_parent_idx_l.append(b['thm_parent_idx']+offset_t)
        thm_root_mask_l.append(b['thm_root_mask'])
        thm_leaf_mask_l.append(b['thm_leaf_mask'])
        tac_id_l.append([b['tac_id']])
        offset_g+=b['goal_token'].shape[0]
        offset_t+=b['thm_token'].shape[0]
        length_list_t +=b['length_list_t']
        length_list_g.append(b['goal_token'].shape[0])

    goal_token = np.concatenate(goal_token_l,axis=0)
    goal_self_index_p = np.concatenate(goal_self_index_p_l,axis=0)
    goal_parent_idx = np.concatenate(goal_parent_idx_l,axis=0)
    goal_root_mask = np.concatenate(goal_root_mask_l,axis=0)
    goal_leaf_mask = np.concatenate(goal_leaf_mask_l,axis=0)

    thm_token = np.concatenate(thm_token_l,axis=0)
    thm_self_index_p = np.concatenate(thm_self_index_p_l,axis=0)
    thm_parent_idx = np.concatenate(thm_parent_idx_l,axis=0)
    thm_root_mask = np.concatenate(thm_root_mask_l,axis=0)
    thm_leaf_mask = np.concatenate(thm_leaf_mask_l,axis=0)
    
    length_list_t = np.array(length_list_t)
    length_list_g = np.array(length_list_g)


    tac_id = np.concatenate(tac_id_l,axis=0)
    #compute gather idx
    idx_gather_goal = get_gather_idx(length_list_g)
    idx_gather_thm = get_gather_idx(length_list_t)
    
    output = {
        'goal_token':as_tensor(goal_token),
        'goal_self_index_p':as_tensor(goal_self_index_p),
        'goal_parent_idx':as_tensor(goal_parent_idx),
        'goal_root_mask':as_tensor(goal_root_mask),
        'goal_leaf_mask':as_tensor(goal_leaf_mask),
        'thm_token':as_tensor(thm_token),
        'thm_self_index_p':as_tensor(thm_self_index_p),
        'thm_parent_idx':as_tensor(thm_parent_idx),
        'thm_root_mask':as_tensor(thm_root_mask),
        'thm_leaf_mask':as_tensor(thm_leaf_mask),
        'tac_id':as_tensor(tac_id),
        'idx_gather_goal':as_tensor(idx_gather_goal),
        'idx_gather_thm':as_tensor(idx_gather_thm),
        'length_list_g':as_tensor(length_list_g),
        'length_list_t':as_tensor(length_list_t)
    }
    return(output)