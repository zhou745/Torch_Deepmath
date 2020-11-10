srun --partition=vc_research_2 --gres=gpu:8 -N1 --job-name=load270 --kill-on-bad-exit=1 python train.py \
     --configs /mnt/cache/zhoujingqiu/configs/exp_pclr_12hop_lr1_uni_embed_pytorch_layernorm_8gpu_mvmean0001_noshare_v2.npy
