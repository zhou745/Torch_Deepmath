import numpy as np
save_name = "/mnt/cache/zhoujingqiu/configs/exp_pclr_0hop_small_remove_relu.npy"
args = np.load(save_name,allow_pickle=True).tolist()

# print(getattr(args, 'neck_module'))
print(args)
# print(hasattr(args, 'neck_module'))
# print(hasattr(args, 'neck_layer_size'))
