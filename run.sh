srun --partition=vc_research_2 --gres=gpu:8 -N1 -x SH-IDC1-10-198-8-[125,126] --job-name=2hop --kill-on-bad-exit=1 python train.py \
     --configs /mnt/cache/zhoujingqiu/configs/exp_pclr_2hop_lr1_uni_embed_pytorch_layernorm_8gpu_mvmean0001_noshare.npy
