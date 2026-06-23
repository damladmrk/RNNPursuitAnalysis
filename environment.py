#trajectories

from turtle import pos

import torch
import math

class PursuitEnvironment:
    """
    RT and CT trajectory 

    params:
        L     : area (1.0m)
        T     : tot time (50)
        dt    : time step (0.02s)
        v_max : max speed (1.125m/dt)
    """

    def __init__(self, L=1.0, T=50, dt=0.02, v_max=1.125):
        self.L     = L
        self.T     = T
        self.dt    = dt
        self.v_max = v_max
        self.half  = L / 2.0  # [-0.5, 0.5]

    # random trajectory (RT)
    def sample_RT(self, batch_size):
        """
        random trajectory

        what does it do:
            - wall avoidance
            - smooth path -> momentum/bias 

        Args:
            batch_size : int - trial num

        Returns:
            z_rnn_0       : (batch, 2) - RNN agent starting pos
            z_target_0    : (batch, 2) - target starting pos
            u_seq         : (T, batch, 2) - target speed sequence [vx, vy]
            z_target_final: (batch, 2) - target final position
        """
        sigma_RT = 11.52
        std_per_step = sigma_RT * self.dt  # 11.52 * 0.02 = 0.2304 rad
 
        # --- starting positions: randomly inside the arena ---
        z_rnn_0    = (torch.rand(batch_size, 2) - 0.5) * self.L
        z_target_0 = (torch.rand(batch_size, 2) - 0.5) * self.L
 
        # --- starting direction: random [-pi, pi] ---
        theta = (torch.rand(batch_size) * 2 - 1) * math.pi  # (batch,)
 
        # --- starting speed: uniform [v_max/2, v_max] ---
        speed = torch.rand(batch_size) * (self.v_max / 2) + (self.v_max / 2)
 
        # --- generate trajectory ---
        z_target = z_target_0.clone()
        u_seq    = []
 
        for t in range(self.T):
            # angle update
            theta = theta + torch.randn(batch_size) * std_per_step
 
            # wall avoidance
            theta = self.wall_avoidance(z_target, theta)
 
            # speed update 
            speed = torch.clamp(
                speed + torch.randn(batch_size) * 0.001,
                self.v_max / 4,
                self.v_max
            )
 
            # speed vector: u(t) = [vx, vy]
            vx = speed * torch.cos(theta)
            vy = speed * torch.sin(theta)
            u  = torch.stack([vx, vy], dim=1)  # (batch, 2)
            u_seq.append(u)
 
            # position update
            z_target = z_target + u * self.dt
            z_target = self.clip_to_arena(z_target)
 
        u_seq         = torch.stack(u_seq, dim=0)  # (T, batch, 2)
        z_target_final = z_target
 
        return z_rnn_0, z_target_0, u_seq, z_target_final
 
    # characteristic trajectory (CT)
    def sample_CT(self, batch_size):
        """
        characteristic trajectory 

        what does it do:
            1) choose one of the 4 starting points 
            2) move towards the target wall (small direction changes)
            3) turn +/-90 degrees when approaching the wall
            4) move along the wall towards x=0
            -> L-shaped path is created (Fig. 1F)

        Args:
            batch_size : int

        Returns:
            z_rnn_0       : (batch, 2)
            z_target_0    : (batch, 2)
            u_seq         : (T, batch, 2)
            z_target_final: (batch, 2)
        """
        sigma_CT = 1.0
        std_per_step = sigma_CT * self.dt  # 1.0 * 0.02 = 0.02 rad
        q = self.L / 4.0                   # L/4 = 0.25
 
        # (z_target_0, z_rnn_0, theta_0)
        configs = [
            ([ q,  q], [ 0,  q], 0.0       ),  
            ([-q,  q], [-q,  0], math.pi   ),  
            ([ q, -q], [ q,  0], 0.0       ),  
            ([-q, -q], [ 0, -q], math.pi   ),  
        ]
 
        # randomly choose one of the 4 configs 
        choice = torch.randint(0, 4, (batch_size,))
 
        z_target_0 = torch.zeros(batch_size, 2)
        z_rnn_0 = torch.zeros(batch_size, 2)
        theta = torch.zeros(batch_size)
        turned = torch.zeros(batch_size, dtype=torch.bool)  
 
        for i, (zt, zr, th) in enumerate(configs):
            mask = (choice == i)
            z_target_0[mask] = torch.tensor(zt, dtype=torch.float32)
            z_rnn_0[mask]    = torch.tensor(zr, dtype=torch.float32)
            theta[mask]      = th
 
        # --- trajectory ---
        z_target = z_target_0.clone()
        speed    = torch.full((batch_size,), self.v_max * 0.8)
        u_seq    = []
 
        for t in range(self.T):
            theta = theta + torch.randn(batch_size) * std_per_step

            limit = self.half - 0.08
            near_wall = (z_target.abs().max(dim=1).values > limit) & ~turned
 
            # when approaching the wall, turn +/-90 degrees
            upper_right = (z_target[:, 0] > 0) & (z_target[:, 1] > 0)
            lower_left  = (z_target[:, 0] < 0) & (z_target[:, 1] < 0)
            turn_positive = upper_right | lower_left
 
            # torch.where(condition, 1, 0) -> 1 if condition is True, else 0
            turn_amount = torch.where(turn_positive,torch.tensor(math.pi / 2),torch.tensor(-math.pi / 2))
 
            theta  = torch.where(near_wall, theta + turn_amount, theta)
            turned = turned | near_wall
 
            vx = speed * torch.cos(theta)
            vy = speed * torch.sin(theta)
            u  = torch.stack([vx, vy], dim=1)
            u_seq.append(u)
 
            # update pos
            z_target = z_target + u * self.dt
            z_target = self.clip_to_arena(z_target)
 
        u_seq          = torch.stack(u_seq, dim=0)  # (T, batch, 2)
        z_target_final = z_target
 
        return z_rnn_0, z_target_0, u_seq, z_target_final
 
    # sample batch
    def sample_batch(self, batch_size, percent_CT=0.25):
        """
        %75 RT + %25 CT batch

        Returns:
            z_rnn_0       : (batch, 2)
            z_target_0    : (batch, 2)
            u_seq         : (T, batch, 2)
            z_target_final: (batch, 2)
        """
        n_CT = int(batch_size * percent_CT)
        n_RT = batch_size - n_CT
 
        rt_data = self.sample_RT(n_RT)
        ct_data = self.sample_CT(n_CT)
 
        # concatenate RT and CT data
        z_rnn_0    = torch.cat([rt_data[0], ct_data[0]], dim=0)
        z_target_0 = torch.cat([rt_data[1], ct_data[1]], dim=0)
        u_seq      = torch.cat([rt_data[2], ct_data[2]], dim=1)  # dim=1: batch
        z_target_f = torch.cat([rt_data[3], ct_data[3]], dim=0)
 
        # shuffle the batch
        idx = torch.randperm(batch_size)
        z_rnn_0    = z_rnn_0[idx]
        z_target_0 = z_target_0[idx]
        u_seq      = u_seq[:, idx, :]  
        z_target_f = z_target_f[idx]
 
        return z_rnn_0, z_target_0, u_seq, z_target_f
    
    # ============  helpers  ==============

    # pos in the area
    def clip_to_arena(self, pos):
        return torch.clamp(pos, -self.half, self.half)

    # to provide wall avoidance bias, clip to arena
    def wall_avoidance(self, pos, theta, margin=0.05):
        limit = self.half - margin
 
        # x wall
        near_x = pos[:, 0].abs() > limit
        theta = torch.where(near_x, math.pi - theta, theta)
 
        # y wall
        near_y = pos[:, 1].abs() > limit
        theta = torch.where(near_y, -theta, theta)
 
        return theta