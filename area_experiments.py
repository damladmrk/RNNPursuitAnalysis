# arena_experiments.py
import torch
import numpy as np
from model import PursuitRNN
from environment import PursuitEnvironment
from analysis import evaluate_model, print_summary, make_multiple_videos

model = PursuitRNN(N=1024, dt=0.02, v_max=1.125)
ckpt = torch.load('checkpoints/model_best.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()

arenas = [1.0, 2.0, 3.0, 4.0]

for L in arenas:
    for center_bias in [True, False]:
        tag = f"L{L}m_{'center' if center_bias else 'free'}"
        print(f"\n{'='*40}")
        print(f"Arena: {L}m | center_bias: {center_bias}")
        env = PursuitEnvironment(L=L, T=50, dt=0.02, v_max=1.125, center_bias=center_bias)
        rt_dists, ct_dists = evaluate_model(model, env, n_trials=500)
        print_summary(rt_dists, ct_dists)
        make_multiple_videos(model, env, n_RT=2, n_CT=2,
                            save_dir=f'videos/{tag}')