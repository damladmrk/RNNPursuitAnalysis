# make_videos.py
import torch
from model import PursuitRNN
from environment import PursuitEnvironment
from analysis import make_multiple_videos

# model upload
model = PursuitRNN(N=1024, dt=0.02, v_max=1.125)
ckpt = torch.load('checkpoints/model_best.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()

# environment
env = PursuitEnvironment(L=1.0, T=50, dt=0.02, v_max=1.125)

# videos
make_multiple_videos(model, env, n_RT=3, n_CT=3, save_dir='videos')