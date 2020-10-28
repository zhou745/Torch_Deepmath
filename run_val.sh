srun --partition=vc_research_2 --gres=gpu:0 -w SH-IDC1-10-198-8-125  --job-name=GNN --kill-on-bad-exit=1 python val.py
