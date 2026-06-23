# RNN model itself

import math
import torch
import torch.nn as nn


class PursuitRNN(nn.Module):
    """
    dimentions for wights:
        - W_back  : (N, 4)      starting pos to model
        - W_in    : (N, 2)      input weights
        - W_rec   : (N, N)      hidden layer weights in memory
        - W_out   : (2, N)      output state to (theta, v)
    
    calculations:
        r(0)   = ReLU(W_back (x) [z_RNN(0), z_target(0)])
        r(t+1) = ReLU(W_rec (x) r(t) + W_in (x) u(t))
        o(t)   = W_out (x) r(t)  ->  (theta, v)
        z_RNN  = z_RNN + v * dt * [cos(theta), sin(theta)]
    """

    def __init__(self, N=1024, dt=0.02, v_max=1.125):
        super().__init__()
        self.N = N
        self.dt = dt
        self.v_max = v_max

        # --- weights ---
        # W_back ~ N(0, 1) 
        self.W_back = nn.Parameter(torch.empty(N, 4))

        # W_in ~ N(0, 1) 
        self.W_in = nn.Parameter(torch.empty(N, 2))

        # W_rec: recurrent weights
        self.W_rec = nn.Parameter(torch.empty(N, N))

        # no initialization for W_out 
        self.W_out = nn.Parameter(torch.empty(2, N))

        # Bias 
        self.b = nn.Parameter(torch.zeros(N))

        # ?????
        nn.init.xavier_uniform_(self.W_back)
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)
        nn.init.normal_(self.W_rec, mean=0, std=1/math.sqrt(N))

    def forward(self, z_rnn_0, z_target_0, u_seq):
        """
        args:
            z_rnn_0    : (batch, 2) - RNN-agent initial pos
            z_target_0 : (batch, 2) - target initial pos
            u_seq      : (T, batch, 2) - target speed sequence [vx


        returns:
            z_rnn_final : (batch, 2) - RNN's final pos
            r_seq       : (T, batch, N) - to analyze
        """
        T = u_seq.shape[0]
        batch = u_seq.shape[1]
 
        # r(0) = W_back @ [z_RNN(0), z_target(0)]
        init_pos = torch.cat([z_rnn_0, z_target_0], dim=1)  # (batch, 4)
        r = init_pos @ self.W_back.T  # (batch, N)

        # pos updates on this
        z_rnn = z_rnn_0.clone()

        # hidden states
        r_seq = []

        for t in range(T):
            # r(t+1) = ReLU(W_rec @ r(t) + W_in @ u(t) + b)
            pre_act = r @ self.W_rec.T + u_seq[t] @ self.W_in.T + self.b
            r = torch.relu(pre_act)  # (batch, N)
            r_seq.append(r)

            # --- Output ---
            # o(t) = W_out @ r(t) -> (theta, v)
            o = r @ self.W_out.T    
            theta = o[:, 0]         
            v = o[:, 1]             

            # v_max = 1.125
            v = torch.sigmoid(o[:, 1]) * self.v_max

            # --- pos update ---
            # z_RNN(t+1) = z_RNN(t) + v * dt * [cos(theta), sin(theta)]
            dx = v * torch.cos(theta) * self.dt  # (batch,)
            dy = v * torch.sin(theta) * self.dt  # (batch,)
            delta_z = torch.stack([dx, dy], dim=1)  # (batch, 2)
            z_rnn = z_rnn + delta_z

        r_seq = torch.stack(r_seq, dim=0)  # (T, batch, N)

        return z_rnn, r_seq


def pursuit_loss(z_rnn_final, z_target_final):
    """
    L_end = ||z_RNN(T) - z_target(T)||^2
    take mean over the batch

    --- add the tired loss term later ---
    
    """
    diff = z_rnn_final - z_target_final          # (batch, 2)
    dist_sq = torch.sum(diff ** 2, dim=1)        # (batch,)
    return torch.mean(dist_sq) 