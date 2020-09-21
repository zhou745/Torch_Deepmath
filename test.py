import TorchDeepmath as td
import torch
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
# td.Data.gen_from_tfrecord.tfgen("/mnt/lustre/zhoujingqiu/deephol-data/deepmath/deephol/proofs/human/",
#                                 "/mnt/cache/zhoujingqiu/data_goal2")
# td.Data.gen_from_tfrecord.text_gen("/mnt/lustre/zhoujingqiu/deephol-data/deepmath/deephol/proofs/human/",
#                                    "/mnt/cache/zhoujingqiu/data_thm2")

# for item in tqdm(data_loader):
#     # print(item,flush=True)
#     print("load one!",flush=True)



if __name__ == '__main__':
    # f1 = open("/mnt/lustre/zhoujingqiu/deephol-data/deepmath/deephol/proofs/human/vocab_goal_ls.txt")
    # f2 = open("/mnt/lustre/zhoujingqiu/deephol-data/deepmath/deephol/proofs/human/vocab_thms_ls.txt")

    # content1 = f1.readlines()
    # content2 = f2.readlines()

    # f1.close()
    # f2.close()

    # print(len(content1),flush=True)
    # print(len(content2),flush=True)
    neg_hard_per_pos = 1
    neg_per_pos = 15
    bactch_size = 128
    word_size = 2
    save_name="/mnt/cache/share_data/zhoujingqiu/ckpt/exp5/model_epoch"

    dataset = td.Data.dataset.GNN_dataset("../data/data_goal2.npy","../data/data_thm2.npy",{'neg_per_pos':neg_per_pos,'neg_hard_per_pos':neg_hard_per_pos})

    model = td.Model.GNN.GNN_net({
        'goal_voc_length':1109,
        'goal_voc_embedsize':128,
        'thm_voc_length':1193,
        'thm_voc_embedsize':128,
        'num_hops':12,
        'score_weight':0.2,
        'tactic_weight':1.0,
        'auc_weight':4.0,
        'neg_per_pos':neg_per_pos,
        'bactch_size':bactch_size,
        'word_size':word_size
    })

    td.Train.Train_GNN.TrainLoop(dataset,model,word_size,bactch_size,save_name)
