"""Provides data for HOL Tactics + Parameters.

Example usage:

mode = tf.estimator.ModeKeys.TRAIN
dataset = data.get_holparam_dataset(mode=mode, dataset_dir=dataset_dir)
input_fn = data.get_input_fn(dataset=dataset, mode=mode, params=params,
                             shuffle_queue=10000,
                             repeat=False)
features, labels = input_fn()
"""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import functools
import os
from TorchDeepmath.Utils import sexpress2graph
import numpy as np
import tensorflow as tf
# import pyarrow as pa
from multiprocessing import Process, Queue

#deserialization of the tfrecord file.   
def parse(item,source):
    out_decode=tf.io.parse_single_example(
        item,
        features={
            # Subgoal features
            # goal: the consequent term of the subgoal as a string.
            'goal': tf.io.FixedLenFeature((), tf.string, default_value=''),
            # goal_asl: list of hypotheses of the subgoal.
            'goal_asl': tf.io.VarLenFeature(dtype=tf.string),
            # Parameterized tactic applied to the subgoal
            # tactic: string name of tactic that is applied to this subgoal.
            'tactic': tf.io.FixedLenFeature((), tf.string, default_value=''),
            # tac_id: integer id of tactic.
            'tac_id': tf.io.FixedLenFeature((), tf.int64, default_value=-1),
            # thms: list of tactic arguments of type thm.
            'thms': tf.io.VarLenFeature(dtype=tf.string),
            # thms_hard_negatives: list of hard negative theorem parameter
            # arguments
            'thms_hard_negatives': tf.io.VarLenFeature(dtype=tf.string),
        })
    
    for key in ('goal_asl', 'thms', 'thms_hard_negatives'):
        if key in out_decode:
            out_decode[key] = tf.sparse.to_dense(out_decode[key], default_value='')
    return(out_decode)

#read directly from tfrecord files, put them into a multiprocess queue
def GetData_from_TF(files,Queue_raw,num_worker):
    dataset = tf.data.TFRecordDataset(files).map(lambda value: (value, 0))
    dataset = dataset.map(parse)
    idx = 0
    for item in dataset:
        idx+=1
        if idx<37000:
            continue
        if idx>38000:
            break
        Queue_raw.put(item)
    
    #signal all worker that job is down
    for idx in range(num_worker):
        Queue_raw.put("done")
    print("finished read all data from tf record!",flush=True)

def Compute_goal_data(Queue_raw,Queue_new,voc_goal,voc_thm):

    while True:
        item = Queue_raw.get()
        if item == "done":
            Queue_new.put("done")
            break
        else:

            goal_sexp = str(item["goal"].numpy(),encoding="utf-8")
            goal_dict = Conver2Graph(goal_sexp,voc_goal)

            thms_dict_list = []
            thms_sexp = item['thms'].numpy()
            for thm in thms_sexp:
                thms_dict_list.append(Conver2Graph(str(thm,encoding="utf-8"),voc_thm))
            if not thms_dict_list:
                thms_dict_list.append(Conver2Graph("",voc_thm))
            
            neg_thms_dict_list = []
            neg_thms_sexp = item['thms_hard_negatives'].numpy()
            for neg_thm in neg_thms_sexp:
                neg_thms_dict_list.append(Conver2Graph(str(neg_thm,encoding="utf-8"),voc_thm))
            
            output = {
                'goal':goal_dict,
                'thms':thms_dict_list,
                'thms_hard_negatives':neg_thms_dict_list,
                'tac_id':item['tac_id'].numpy()
            }
            Queue_new.put(output)
    
    print("one worker finished its job!",flush=True)

def Get_data_from_file(files,Queue_raw,num_worker):
    dataset = tf.data.TextLineDataset(files)
    idx=0
    for item in dataset:

        idx+=1
        if idx<1000:
            continue
        if idx>2000:
            break
        Queue_raw.put(item)
    
    #signal all worker that job is down
    for idx in range(num_worker):
        Queue_raw.put("done")
    print("finished read all data from tf record!",flush=True)

def Compute_thm_data(Queue_raw,Queue_new,voc_thm):
    while True:
        item = Queue_raw.get()
        if item == "done":
            Queue_new.put("done")
            break
        else:
            thm_sexp = item.numpy()
            thm_dict=Conver2Graph(str(thm_sexp,encoding="utf-8"),voc_thm)
            
            Queue_new.put(thm_dict)
    
    print("one worker finished its job!",flush=True)

def get_voc(path):
    f=open(path)
    content = f.readlines()
    f.close()
    voc_dict = {}
    for index in range(len(content)):
        voc_dict.update({content[index].rstrip("\n"):index})

    return(voc_dict)

def text_gen(dataset_dir,save_file,num_worker=64):
    path = os.path.join(dataset_dir, 'thms_ls.train')
    files = tf.io.gfile.glob(path)  
    Queue_raw = Queue(512)
    Queue_new = Queue(512)

    path_voc_thm = dataset_dir+"/vocab_thms_ls.txt"
    thm_voc_dict = get_voc(path_voc_thm)

    process_pool = []
    process_pool.append(Process(target=Get_data_from_file,args=(files,Queue_raw,num_worker)))
    for idx in range(num_worker):
        process_pool.append(Process(target=Compute_thm_data,args=(Queue_raw,Queue_new,thm_voc_dict)))

    for p in process_pool:
        p.start()

    # dataset = Get_dataset(path,descript)
    finished = 0
    index=0
    reocrd_list = []

    while finished<num_worker:

        item = Queue_new.get()
        if item == "done":
            finished+=1
        else:
            index+=1
            reocrd_list.append(item)
            if index%1000==0:
                print(index,flush=True)
    
    # buf = pa.serialize(reocrd_list).to_buffer()

    # f = pa.output_stream(save_file)
    # f.write(buf.to_pybytes())
    # f.close()
    # f = open(save_file,"w")
    # json.dump(reocrd_list, f)
    # f.close()
    np.save(save_file,reocrd_list)
    print("find %d in total"%(index),flush=True)

def tfgen(dataset_dir,save_file,num_worker=64):
    path = dataset_dir+"/train/tfexamples*"
    files = tf.io.gfile.glob(path)  
    Queue_raw = Queue(512)
    Queue_new = Queue(512)

    path_voc_goal = dataset_dir+"/vocab_goal_ls.txt"
    path_voc_thm = dataset_dir+"/vocab_thms_ls.txt"

    goal_voc_dict = get_voc(path_voc_goal)
    thm_voc_dict = get_voc(path_voc_thm)

    process_pool = []
    process_pool.append(Process(target=GetData_from_TF,args=(files,Queue_raw,num_worker)))
    for idx in range(num_worker):
        process_pool.append(Process(target=Compute_goal_data,args=(Queue_raw,Queue_new,goal_voc_dict,thm_voc_dict)))

    for p in process_pool:
        p.start()

    # dataset = Get_dataset(path,descript)
    finished = 0
    index=0
    reocrd_list = []

    while finished<num_worker:

        item = Queue_new.get()
        if item == "done":
            finished+=1
        else:
            index+=1
            reocrd_list.append(item)
            if index%1000==0:
                print(index,flush=True)
    
    # f = open(save_file,"w")
    # buf = pa.serialize(reocrd_list)
    # f = pa.output_stream(save_file)
    # # json.dump(reocrd_list, f)
    # f.write(buf)
    # f.close()
    np.save(save_file,reocrd_list)
    print("find %d in total"%(index),flush=True)


def Conver2Graph(sexp,voc_dict):
    #add samplor for negative examples
    graph = sexpress2graph.GNN_tree(sexp)
    graph_token = graph.tokens
    #convert token to idx
    graph_token_idx = []
    for token in graph_token:
        if token == "<OBSC>":
            print("find oen",flush=True)
            
        if token not in voc_dict.keys():
            token = "<OBSC>"
        graph_token_idx.append(voc_dict[token])

    graph_adj_matrix = graph.get_adjcent_maxtrix()
    graph_length = len(graph.tokens)
    
    
    graph_self_idx_p,graph_parent_idx,graph_root_mask,graph_leaf_mask = get_gather_idx(graph_adj_matrix,graph_length)

    # graph_para = {
    #     'token':graph_token_idx,
    #     "self_index_p":graph_self_idx_p.tolist(),
    #     "parent_idx":graph_parent_idx.tolist(),
    #     "root_mask":graph_root_mask.tolist(),
    #     "leaf_mask":graph_leaf_mask.tolist()
    # }
    graph_para = {
        'token':np.array(graph_token_idx),
        "self_index_p":graph_self_idx_p,
        "parent_idx":graph_parent_idx,
        "root_mask":graph_root_mask,
        "leaf_mask":graph_leaf_mask
    }

    return(graph_para)

def get_gather_idx(graph_adj_matrix, graph_length):

    #compute the new length list
    p_idx_list = np.array(np.where(graph_adj_matrix == 1)).T
    
    root_mask = (np.sum(graph_adj_matrix == 1,axis=1)<0.5).astype(np.float32)
    leaf_mask = (np.sum(graph_adj_matrix == -1,axis=1)<0.5).astype(np.float32)
    
  
    p_gather_idx = np.zeros([len(p_idx_list)],dtype=np.int64)
    self_p_gather_idx = np.zeros([len(p_idx_list)],dtype=np.int64)
    
    offset_p = 0
    for cordinate in p_idx_list:
        self_idx, p_idx = cordinate
        self_p_gather_idx[offset_p] = self_idx
        p_gather_idx[offset_p] = p_idx
        offset_p+=1

    return self_p_gather_idx,p_gather_idx,root_mask,leaf_mask