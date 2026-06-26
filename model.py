# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PursuitRNN(nn.Module):
    """
    Full-rank RNN for pursuit 

    Units!!!!!:
        u_seq   : m/step  
        v_RNN   : m/s     (W_out)
        step    : v * dt  = m/step 
        v_max   : 1.125 m/s → * dt = 0.0225 m/step

    Weights:
        W_back : (N, 4)   
        W_in   : (N, 2)   
        W_rec  : (N, N)   
        W_out  : (2, N)   
        b      : (N,)     

    Dynamics:
        r(0)   = W_back @ [z_RNN(0), z_target(0)]      
        r(t+1) = ReLU(W_rec @ r(t) + W_in @ u(t) + b)
        o(t)   = W_out @ r(t)
        theta  = o[0]
        v      = sigmoid(o[1]) * v_max                  
        z_RNN(t+1) = z_RNN(t) + v * dt * [cos θ, sin θ]
    """

    def __init__(self, N=1024, dt=0.02, v_max=1.125):
        super().__init__()
        self.N     = N
        self.dt    = dt
        self.v_max = v_max  # m/s — * dt = 0.0225 m/step

        # weights — empty, initialize later
        self.W_back = nn.Parameter(torch.empty(N, 4))
        self.W_in   = nn.Parameter(torch.empty(N, 2))
        self.W_rec  = nn.Parameter(torch.empty(N, N))
        self.W_out  = nn.Parameter(torch.empty(2, N))
        self.b      = nn.Parameter(torch.zeros(N))

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform ???????????
        nn.init.xavier_uniform_(self.W_back)
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)

        # W_rec: edge of chaos — N(0, 1/sqrt(N))
        nn.init.normal_(self.W_rec, mean=0.0, std=1.0 / math.sqrt(self.N))

        nn.init.zeros_(self.b)

    def forward(self, z_rnn_0, z_target_0, u_seq):
        """
        Args:
            z_rnn_0    : (batch, 2) 
            z_target_0 : (batch, 2) 
            u_seq      : (T, batch, 2) 

        Returns:
            z_rnn_final : (batch, 2)
            r_seq       : (T, batch, N)
            v_seq       : (T, batch)
        """
        T     = u_seq.shape[0]
        z_rnn = z_rnn_0.clone()

        # r(0) = W_back @ [z_RNN(0), z_target(0)] 
        init_pos = torch.cat([z_rnn_0, z_target_0], dim=1)  # (batch, 4)
        r = init_pos @ self.W_back.T                  # (batch, N)

        r_seq = []
        v_seq = []

        for t in range(T):
            # r(t+1) = ReLU(W_rec @ r(t) + W_in @ u(t) + b)
            r = torch.relu(r @ self.W_rec.T + u_seq[t] @ self.W_in.T + self.b)
            r_seq.append(r)

            # o(t) = W_out @ r(t)
            o     = r @ self.W_out.T       # (batch, 2)
            theta = o[:, 0]                # [-inf, inf] ~ from example

            # speed: w/sigmoid squeee bw/[0, v_max] 
            v = torch.sigmoid(o[:, 1]) * self.v_max  # m/s
            v_seq.append(v)

            # update position: z(t+1) = z(t) + v * dt * [cos θ, sin θ]
            step   = v * self.dt  # m/adım
            dx     = step * torch.cos(theta)
            dy     = step * torch.sin(theta)
            z_rnn  = z_rnn + torch.stack([dx, dy], dim=1)

        r_seq = torch.stack(r_seq, dim=0)  # (T, batch, N)
        v_seq = torch.stack(v_seq, dim=0)  # (T, batch)

        return z_rnn, r_seq, v_seq

# ─────────────────────────────────────────────
#  LOSS 
# ─────────────────────────────────────────────

def pursuit_loss(z_rnn_final, z_target_final):
    """
    L_end = ||z_RNN(T) - z_target(T)||^2
    """
    diff    = z_rnn_final - z_target_final   # (batch, 2)
    dist_sq = torch.sum(diff**2, dim=1)      # (batch,)
    return torch.mean(dist_sq)


def loss_with_energy(z_rnn_final, z_target_final, r_seq, v_seq,
                     lambda_r=0.001, lambda_v=0.01):
    """
    cost of burning energy in the brain and muscles ?????

    L = L_end + lambda_r * L_neural + lambda_v * L_energy

    L_end    : distance to target — pursuit error
    L_neural : neuron energy — firing loss
    L_energy : speed penalty — muscle energy / fatigue
    """
    # 1 
    diff      = z_rnn_final - z_target_final
    dist_loss = torch.mean(torch.sum(diff**2, dim=1))
    # 2
    neural_loss = torch.mean(r_seq**2)
    # 3 
    energy_loss = torch.mean(v_seq**2)

    return dist_loss + lambda_r * neural_loss + lambda_v * energy_loss