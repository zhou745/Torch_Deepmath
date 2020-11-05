srun --partition=vc_research_2 --gres=gpu:0 -w SH-IDC1-10-198-8-128  --job-name=GNN --kill-on-bad-exit=1 python auto_val.py \
   --load_path /mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_lr1_uni_embed_pytorch_layernorm_8gpu_mvmean0001_noshare.npy --gpu 4
