srun --partition=vc_research_2 --gres=gpu:8 -N1  --job-name=GNN318 --kill-on-bad-exit=1 python train.py
