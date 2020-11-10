srun --partition=vc_research_2 --gres=gpu:0 -w SH-IDC1-10-198-8-127  --job-name=GNN --kill-on-bad-exit=1 python auto_val.py \
   --load_path /mnt/lustre/share_data/fangrongyao/ckpt/exp_hop4_lr4_batch128_layernorm_gnnres_No1019/model_epoch_config.npy --gpu 4
