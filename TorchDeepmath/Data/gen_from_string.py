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
# import pyarrow as pa
from multiprocessing import Process, Queue


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

    # graph_adj_matrix = graph.get_adjcent_maxtrix()
    # graph_length = len(graph.tokens)
    
    
    # graph_self_idx_p,graph_parent_idx,graph_root_mask,graph_leaf_mask = get_gather_idx(graph_adj_matrix,graph_length)
    idx_list = get_gather_idx_left_right_child(graph)

    graph_para = {
        'token':np.array(graph_token_idx,dtype=np.int64),
        "edge_p_node":idx_list[0],
        "edge_c_node":idx_list[1],
        "edge_p_indicate":idx_list[2],
        "edge_c_indicate":idx_list[3],
        "p_mask": idx_list[4],
        "c_mask": idx_list[5]
    }
    # graph_para = {
    #     'token':np.array(graph_token_idx),
    #     "self_index_p":graph_self_idx_p,
    #     "parent_idx":graph_parent_idx,
    #     "root_mask":graph_root_mask,
    #     "leaf_mask":graph_leaf_mask
    # }

    return(graph_para)

def get_gather_idx_left_right_child(graph):
    #loop through all node to get edges
    node_list = graph.nodes
    edge_p_node, edge_c_node = [], []
    edge_p_indicate, edge_c_indicate = [], []

    p_mask = np.array([1. for i in range(len(node_list))],dtype=np.float32)
    c_mask = np.array([1. for i in range(len(node_list))],dtype=np.float32)

    for node in node_list:
        assert len(node.child_index)<=2

        if len(node.child_index) >= 1:
            edge_p_node.append(node.index)
            edge_c_node.append(node.child[0].index)
            edge_p_indicate.append(0.)
            edge_c_indicate.append(0.)

            assert node.child_position[0]==0
            p_mask[node.index]=0.
            c_mask[node.child[0].index]=0.

        if len(node.child_index) == 2:
            edge_p_node.append(node.index)
            edge_c_node.append(node.child[1].index)
            edge_p_indicate.append(1.)
            edge_c_indicate.append(1.)

            assert node.child_position[1]==1
            p_mask[node.index]=0.
            c_mask[node.child[1].index]=0.
    
    edge_p_node = np.array(edge_p_node,dtype=np.int64)
    edge_c_node = np.array(edge_c_node,dtype=np.int64)
    edge_p_indicate = np.array(edge_p_indicate,dtype=np.float32)
    edge_c_indicate = np.array(edge_c_indicate,dtype=np.float32)

    return edge_p_node,edge_c_node,edge_p_indicate,edge_c_indicate,p_mask,c_mask



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
