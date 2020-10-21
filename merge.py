import numpy as np
import random

human_path =  "/mnt/cache/zhoujingqiu/data/data_goal_lr.npy"
synth_path = "/mnt/cache/zhoujingqiu/data/data_goal_syn_lr.npy"
save_file = "/mnt/cache/zhoujingqiu/data/data_human_synth_1_3_lr"

human = np.load(human_path,allow_pickle=True).tolist()
synth = np.load(synth_path,allow_pickle=True).tolist()
print('load finished',flush=True)
total_num = len(synth)
sample_num = total_num//3
sampled = random.sample(synth,sample_num)
print('sampled',flush=True)
merged = human+sampled
print('merged',flush=True)

np.save(save_file,merged)
