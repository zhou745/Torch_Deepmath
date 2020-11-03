srun --partition=vc_research_2 --gres=gpu:8 -N1 -x SH-IDC1-10-198-8-125 --job-name=2hop --kill-on-bad-exit=1 python train.py \
     --configs /mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_lr1_uni_embed_xavier_nobn_8gpu_mvmean0001_noshare.npy
