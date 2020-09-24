srun --partition=vc_research --gres=gpu:0 -w SH-IDC1-10-198-6-128  --job-name=GNN --kill-on-bad-exit=1 python gen_data.py
