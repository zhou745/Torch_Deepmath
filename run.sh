srun --partition=vc_research_2 --gres=gpu:8 -N1 -x SH-IDC1-10-198-8-125 --job-name=12hop_16lr --kill-on-bad-exit=1 python train.py --configs /mnt/cache/zhoujingqiu/configs/exp_pclr_0hop_small_remove_relu_lr1_noshare_xavier_resneck.npy
