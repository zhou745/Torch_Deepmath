import numpy as np
from tqdm import tqdm

goal_path="../data/data_goal.npy"
neg_thm_path = "../data/data_thm.npy"
goal_voc_length=1109
thm_voc_length = 1193
goal_list = np.load(goal_path,allow_pickle=True).tolist()
thm_list = np.load(neg_thm_path,allow_pickle=True).tolist()

thm_voc_stat = [0 for i in range(thm_voc_length)]
goal_voc_stat = [0 for i in range(goal_voc_length)]

for item in tqdm(goal_list):
    for idx in item['goal']['token']:
        goal_voc_stat[idx]+=1


for idx in range(goal_voc_length):
    if goal_voc_stat[idx]==0:
        print(idx)

print("-----------------------")

for item in tqdm(thm_list):
    for idx in item['token']:
        thm_voc_stat[idx]+=1


for idx in range(thm_voc_length):
    if thm_voc_stat[idx]==0:
        print(idx)
