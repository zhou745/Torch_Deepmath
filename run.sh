srun --partition=vc_research_2 --gres=gpu:8 -N1  --job-name=GNN_synth --kill-on-bad-exit=1 python train.py
