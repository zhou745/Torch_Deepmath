import os
import numpy as np
from tqdm import tqdm

save_name = "goal_human"
prefix = "/mnt/cache/zhoujingqiu/data"
file_path = [#"data_goal_synth",
            "data_goal_human"]

file_list = []
for item in file_path:
    file_list.append(os.listdir(os.path.join(prefix,item)))
print("files find done!")
#save in one list 
final_list = []
for idx_i in range(len(file_list)):
    for idx_j in tqdm(range(len(file_list[idx_i]))):
        file_list[idx_i][idx_j]=prefix+"/"+file_path[idx_i]+"/"+file_list[idx_i][idx_j]

    final_list+=file_list[idx_i]

print("merge done!")
np.save(os.path.join(prefix,save_name),final_list)
print("done!")