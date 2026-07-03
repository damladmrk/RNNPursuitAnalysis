# environment.py
# RT and CT trajectory generation

import torch
import numpy as np
import math


class PursuitEnvironment:
    """
    RT and CT trajectory generation.

    Speed measure: m/step
    u_seq[t] = [vx, vy] where vx,vy in m/step

    params:
        L        : area (1.0m)
        T        : total timesteps (50)
        dt       : time step (0.02s)
        v_max    : maximum speed (0.0225 m/step)
    """

    def __init__(self, L=1.0, T=50, dt=0.02, v_max=0.0225, center_bias=False):
        self.L        = L
        self.T        = T
        self.dt       = dt
        self.v_max    = v_max
        self.center_bias = center_bias
        self.half     = L / 2.0

        # Rayleigh scale ???????
        self.b  = 0.13 * 2 * math.pi   # ~0.817 m/s
        self.border_region = 0.03      # close to wall region (m)

        # Dönüş std
        self.sigma_RT = 5.76 * 2   # rad/s → * dt 
        self.sigma_CT = 1.0        # rad/s → * dt 

    # ─────────────────────────────────────────
    #  RT
    # ─────────────────────────────────────────

    def sample_RT(self, batch_size):
        """
        Random Trajectory.

        Returns:
            z_rnn_0       : (batch, 2)
            z_target_0    : (batch, 2)
            u_seq         : (T, batch, 2)  — m/step
            z_target_final: (batch, 2)
        """
        # initial poss
        if self.center_bias:
        # start in a small region around the center (does not affect in 1m arenas)
            z_rnn_0    = (torch.rand(batch_size, 2) - 0.5) * 1.0
            z_target_0 = (torch.rand(batch_size, 2) - 0.5) * 1.0
        else:
            # whole arena; start between [-L/2, L/2] 
            z_rnn_0    = (torch.rand(batch_size, 2) - 0.5) * self.L
            z_target_0 = (torch.rand(batch_size, 2) - 0.5) * self.L

        # initial angle
        hd = np.random.uniform(0, 2 * math.pi, batch_size)

        random_vel  = np.random.rayleigh(self.b, (batch_size, self.T))
        random_turn = np.random.normal(0, self.sigma_RT, (batch_size, self.T))

        z_target = z_target_0.numpy().copy()
        u_seq    = []

        for t in range(self.T):
            v = random_vel[:, t]          # m/s
            turn_angle = np.zeros(batch_size)

            # wall avoidance
            is_near, wall_turn = self.wall_avoidance_RT(z_target, hd)
            turn_angle[is_near] = wall_turn[is_near]
            v[is_near] *= 0.25  # slow down if near wall

            # update angle 
            turn_angle += self.dt * random_turn[:, t]
            hd         += turn_angle

            # take step
            step = v * self.dt  # m/s * s = m
            step = np.clip(step, 0, self.v_max)

            vx = step * np.cos(hd)
            vy = step * np.sin(hd)
            u  = np.stack([vx, vy], axis=1)  # (batch, 2)
            u_seq.append(u)

            z_target = z_target + u
            z_target = np.clip(z_target, -self.half, self.half)

        u_seq          = torch.tensor(np.stack(u_seq, axis=0), dtype=torch.float32)  # (T, batch, 2)
        z_target_final = torch.tensor(z_target, dtype=torch.float32)

        return z_rnn_0, z_target_0, u_seq, z_target_final

    # ─────────────────────────────────────────
    #  CT
    # ─────────────────────────────────────────

    def sample_CT(self, batch_size):
        """
        Characteristic Trajectory 

        Returns:
            z_rnn_0       : (batch, 2)
            z_target_0    : (batch, 2)
            u_seq         : (T, batch, 2)
            z_target_final: (batch, 2)
        """
        # q = self.L / 4.0  ~ changed due to area experiments/start close to center
        q = 0.25

        # (z_target_0, z_rnn_0, theta_0)
        configs = [
            ([-q, -q], [ 0,  q], 0.0      ),  
            ([ q, -q], [-q,  0], math.pi  ),  
            ([-q,  q], [ q,  0], 0.0      ),  
            ([ q,  q], [ 0, -q], math.pi  ),  
        ]

        choice = np.random.randint(0, 4, batch_size)

        z_target_0 = np.zeros((batch_size, 2))
        z_rnn_0    = np.zeros((batch_size, 2))
        hd         = np.zeros(batch_size)

        for i, (zt, zr, th) in enumerate(configs):
            mask = (choice == i)
            z_target_0[mask] = zt
            z_rnn_0[mask] = zr
            hd[mask] = th

        # target speed sequence
        random_vel  = np.random.rayleigh(self.b, (batch_size, self.T))
        random_turn = np.random.normal(0, self.sigma_CT, (batch_size, self.T))

        z_target = z_target_0.copy()
        wall_id  = np.zeros(batch_size)  # ???????
        u_seq    = []

        for t in range(self.T):
            v          = random_vel[:, t]
            turn_angle = np.zeros(batch_size)

            # CT wall avoidance
            is_near, wall_turn, wall_id = self.wall_avoidance_CT(
                z_target, hd, wall_id
            )
            turn_angle[is_near] = wall_turn[is_near]
            v[is_near] *= 0.25

            # update angle
            turn_angle += self.dt * random_turn[:, t]
            hd         += turn_angle

            # take step
            step = v * self.dt
            step = np.clip(step, 0, self.v_max)

            vx = step * np.cos(hd)
            vy = step * np.sin(hd)
            u  = np.stack([vx, vy], axis=1)
            u_seq.append(u)

            z_target = z_target + u
            z_target = np.clip(z_target, -self.half, self.half)

        u_seq          = torch.tensor(np.stack(u_seq, axis=0), dtype=torch.float32)
        z_target_final = torch.tensor(z_target, dtype=torch.float32)

        return (
            torch.tensor(z_rnn_0, dtype=torch.float32),
            torch.tensor(z_target_0, dtype=torch.float32),
            u_seq, z_target_final,
        )

    # ─────────────────────────────────────────
    #  BATCH
    # ─────────────────────────────────────────

    def sample_batch(self, batch_size, percent_CT=0.25):
        """75% RT + 25% CT karışık batch."""
        n_CT = int(batch_size * percent_CT)
        n_RT = batch_size - n_CT

        rt = self.sample_RT(n_RT)
        ct = self.sample_CT(n_CT)

        z_rnn_0    = torch.cat([rt[0], ct[0]], dim=0)
        z_target_0 = torch.cat([rt[1], ct[1]], dim=0)
        u_seq      = torch.cat([rt[2], ct[2]], dim=1)
        z_target_f = torch.cat([rt[3], ct[3]], dim=0)

        idx        = torch.randperm(batch_size)
        return (
            z_rnn_0[idx],
            z_target_0[idx],
            u_seq[:, idx, :],
            z_target_f[idx],
        )

    # ─────────────────────────────────────────
    #  HELPERS 
    # ─────────────────────────────────────────

    def wall_avoidance_RT(self, pos, hd):
        """RT wall avoidance."""
        x, y   = pos[:, 0], pos[:, 1]
        half   = self.half
        border = self.border_region

        dists  = [half - x, half - y, half + x, half + y]
        d_wall = np.min(dists, axis=0)
        angles = np.array([0, math.pi/2, math.pi, 3*math.pi/2])
        theta  = angles[np.argmin(dists, axis=0)]

        hd_mod = np.mod(hd, 2 * math.pi)
        a_wall = hd_mod - theta
        a_wall = np.mod(a_wall + math.pi, 2 * math.pi) - math.pi

        is_near   = (d_wall < border) & (np.abs(a_wall) < math.pi / 2)
        turn      = np.zeros_like(hd)
        turn[is_near] = np.sign(a_wall[is_near]) * (math.pi/2 - np.abs(a_wall[is_near]))

        return is_near, turn

    def wall_avoidance_CT(self, pos, hd, wall_id):
        """
        CT wall avoidance
        """
        x, y   = pos[:, 0], pos[:, 1]
        half   = self.half
        border = self.border_region

        dists  = [half - x, half - y, half + x, half + y]
        d_wall = np.min(dists, axis=0)
        angles = np.array([0, math.pi/2, math.pi, 3*math.pi/2])
        theta  = angles[np.argmin(dists, axis=0)]

        hd_mod = np.mod(hd, 2 * math.pi)
        a_wall = hd_mod - theta
        a_wall = np.mod(a_wall + math.pi, 2 * math.pi) - math.pi

        is_near = (d_wall < border) & (np.abs(a_wall) < math.pi/2) & (wall_id < 5)

        turn = np.zeros_like(hd)
        # if upper-right ve lower-left → -pi/2; else → +pi/2
        turn[is_near & (x > 0) & (y > 0)] = -1 * (math.pi/2 - np.abs(a_wall[is_near & (x > 0) & (y > 0)]))
        turn[is_near & (x < 0) & (y < 0)] = -1 * (math.pi/2 - np.abs(a_wall[is_near & (x < 0) & (y < 0)]))
        turn[is_near & (x < 0) & (y > 0)] =  1 * (math.pi/2 - np.abs(a_wall[is_near & (x < 0) & (y > 0)]))
        turn[is_near & (x > 0) & (y < 0)] =  1 * (math.pi/2 - np.abs(a_wall[is_near & (x > 0) & (y < 0)]))

        # wall_id update
        wall_id[is_near & (wall_id == 0)] = 1
        wall_id[wall_id > 0] += 1

        return is_near, turn, wall_id