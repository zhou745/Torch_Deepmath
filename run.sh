srun --partition=vc_research_2 --gres=gpu:8 -N1 -x SH-IDC1-10-198-8-125 --job-name=GNN0hop_small --kill-on-bad-exit=1 python train.py
