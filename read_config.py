import numpy as np
save_name = "/mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_lr1_uni_embed_torch_nobn_4gpu_mvmean0002.npy"
args = np.load(save_name,allow_pickle=True).tolist()

# print(getattr(args, 'neck_module'))
print(args)
# print(hasattr(args, 'neck_module'))
# print(hasattr(args, 'neck_layer_size'))
