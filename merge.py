import numpy as np
import random

human_path =  "/mnt/cache/zhoujingqiu/data/data_goal_lr.npy"
synth_path = "/mnt/cache/zhoujingqiu/data/data_goal_syn_lr.npy"
save_file = "/mnt/cache/zhoujingqiu/data/data_goal_valid_small"
thm_path = "/mnt/cache/zhoujingqiu/data/data_thm_lr.npy"

valid_path = "/mnt/cache/zhoujingqiu/data/data_goal_valid.npy"
human_synth_list = "/mnt/cache/zhoujingqiu/data/goal_human_synth.npy"

# human = np.load(human_path,allow_pickle=True).tolist()
# synth = np.load(synth_path,allow_pickle=True).tolist()
# valid = np.load(valid_path,allow_pickle=True).tolist()
# thm = np.load(thm_path,allow_pickle=True).tolist()
list_name = np.load(human_synth_list,allow_pickle=True).tolist()
print(list_name[0],flush=True)
# print('load finished',flush=True)
# total_num = len(valid)
# sample_num = total_num//20
# sampled = random.sample(valid,sample_num)
# print('sampled',flush=True)
# # merged = human+sampled
# # print('merged',flush=True)

# np.save(save_file,sampled)
