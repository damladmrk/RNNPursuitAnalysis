# Analysis results for the model

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from environment import PursuitEnvironment

# ─────────────────────────────────────────────
#  1. TRAJECTORY SAMPLING
# ─────────────────────────────────────────────

def sample_trajectories(env, n_RT=10, n_CT=10, seed=37):
    """
    Sample RT and CT trajectories for visualization

    Returns:
        rt_data : (z_rnn_0, z_target_0, u_seq, z_target_final) for RTs
        ct_data : (z_rnn_0, z_target_0, u_seq, z_target_final) for CTs
    """
    torch.manual_seed(seed) # set random seed for reproducibility
    rt_data = env.sample_RT(n_RT)
    ct_data = env.sample_CT(n_CT)
    return rt_data, ct_data


def reconstruct_target_trajectory(z_target_0, u_seq, dt):
    T = u_seq.shape[0]
    traj = torch.zeros(T + 1, u_seq.shape[1], 2)
    traj[0] = z_target_0
    pos = z_target_0.clone()
    for t in range(T):
        pos = pos + u_seq[t]  
        pos = torch.clamp(pos, -0.5, 0.5)
        traj[t + 1] = pos
    return traj


def run_model_trajectory(model, z_rnn_0, z_target_0, u_seq):
    """
    Run model forward and return full RNN-agent trajectory.

    Returns:
        rnn_traj : (T+1, batch, 2)
    """
    model.eval()
    with torch.no_grad():
        T = u_seq.shape[0]
        batch = u_seq.shape[1]

        init_pos = torch.cat([z_rnn_0, z_target_0], dim=1)
        r = init_pos @ model.W_back.T

        z_rnn = z_rnn_0.clone()
        rnn_traj = torch.zeros(T + 1, batch, 2)
        rnn_traj[0] = z_rnn

        for t in range(T):
            r = torch.relu(r @ model.W_rec.T + u_seq[t] @ model.W_in.T + model.b)
            o = r @ model.W_out.T
            theta = o[:, 0]
            v = torch.clamp(o[:, 1], 0.0, model.v_max)
            delta = v.unsqueeze(1) * torch.stack(
                [torch.cos(theta), torch.sin(theta)], dim=1
            ) * model.dt
            z_rnn = z_rnn + delta
            rnn_traj[t + 1] = z_rnn

    return rnn_traj


# ─────────────────────────────────────────────
#  2. TRAJECTORY PLOTS
# ─────────────────────────────────────────────

def plot_fig1_trajectories(env, model=None, n_trials=4, seed=37, save_path=None):
    """
    Shows RT and CT example trajectories.
    If model is provided, also plots RNN-agent trajectories.

    Args:
        env       : PursuitEnvironment
        model     : PursuitRNN (optional) — if None, only target trajectories shown
        n_trials  : number of example trials per type
        seed      : random seed
        save_path : if given, saves figure to this path
    """
    torch.manual_seed(seed)
    rt_data = env.sample_RT(n_trials)
    ct_data = env.sample_CT(n_trials)

    fig, axes = plt.subplots(2, n_trials, figsize=(3 * n_trials, 6))
    fig.suptitle('Example Trajectories', fontsize=13, fontweight='bold')

    for col, (data, label) in enumerate([(rt_data, 'RT'), (ct_data, 'CT')]):
        z_rnn_0, z_target_0, u_seq, z_target_final = data
        target_traj = reconstruct_target_trajectory(z_target_0, u_seq, env.dt)

        if model is not None:
            rnn_traj = run_model_trajectory(model, z_rnn_0, z_target_0, u_seq)

        for i in range(n_trials):
            ax = axes[col, i]
            ax.set_xlim(-0.55, 0.55)
            ax.set_ylim(-0.55, 0.55)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            # area border
            rect = plt.Rectangle((-0.5, -0.5), 1.0, 1.0, fill=False, edgecolor='gray', linewidth=1)
            ax.add_patch(rect)

            # target trajectory
            tx = target_traj[:, i, 0].numpy()
            ty = target_traj[:, i, 1].numpy()
            ax.plot(tx, ty, 'k-', linewidth=1.5, label='Target', zorder=2)

            # start / end markers
            ax.plot(tx[0], ty[0], 'k^', markersize=6, zorder=3)   # start
            ax.plot(tx[-1], ty[-1], 'ks', markersize=6, zorder=3)  # end

            # RNN trajectory
            if model is not None:
                rx = rnn_traj[:, i, 0].numpy()
                ry = rnn_traj[:, i, 1].numpy()
                ax.plot(rx, ry, 'r-', linewidth=1.5, label='RNN', zorder=2)
                ax.plot(rx[0], ry[0], 'r^', markersize=6, zorder=3)
                ax.plot(rx[-1], ry[-1], 'rs', markersize=6, zorder=3)
                dist = np.sqrt((rx[-1] - tx[-1])**2 + (ry[-1] - ty[-1])**2)
                ax.set_title(f'{label} — {dist:.2f}m', fontsize=8)
            else:
                ax.set_title(f'{label} trial {i+1}', fontsize=8)

    # legend on first plot
    if model is not None:
        axes[0, 0].legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────
#  3. TRAINING LOSS CURVE
# ─────────────────────────────────────────────

def plot_loss_curve(losses, save_path=None):
    """
    Plot training loss over epochs (Fig 1B style).

    Args:
        losses    : list of mean epoch losses
        save_path : optional save path
    """
    epochs = np.arange(1, len(losses) + 1)
    dists = np.sqrt(losses)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, losses, 'r-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (m²)')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, dists, 'r-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Distance to target (m)')
    axes[1].set_title('Distance to Target')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
#  4. MODEL EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, env, n_trials=1000, seed=21):
    """
    Evaluate trained model on RT and CT trials.
    Returns distribution of final distances (Fig 1C, 1D style).

    Returns:
        rt_dists : (n_trials,) final distances on RT trials
        ct_dists : (n_trials,) final distances on CT trials
    """
    torch.manual_seed(seed)
    model.eval()

    with torch.no_grad():
        # RT
        z_rnn_0, z_target_0, u_seq, z_target_final = env.sample_RT(n_trials)
        rnn_traj = run_model_trajectory(model, z_rnn_0, z_target_0, u_seq)
        z_rnn_final_rt = rnn_traj[-1]  # (batch, 2)
        diff = z_rnn_final_rt - z_target_final
        rt_dists = torch.sqrt(torch.sum(diff**2, dim=1)).numpy()

        # CT
        z_rnn_0, z_target_0, u_seq, z_target_final = env.sample_CT(n_trials)
        rnn_traj = run_model_trajectory(model, z_rnn_0, z_target_0, u_seq)
        z_rnn_final_ct = rnn_traj[-1]
        diff = z_rnn_final_ct - z_target_final
        ct_dists = torch.sqrt(torch.sum(diff**2, dim=1)).numpy()

    return rt_dists, ct_dists


def plot_distance_distributions(rt_dists, ct_dists, save_path=None):
    """
    Plot histogram of final distances — Fig 1C, 1D style.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, dists, label, color in zip(
        axes,
        [rt_dists, ct_dists],
        ['RT', 'CT'],
        ["#d0767e", "#657bda"]
    ):
        ax.hist(dists, bins=30, color=color, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Dist. to target (m)')
        ax.set_ylabel('% trajectories')
        ax.set_title(f'{label} — median: {np.median(dists):.3f}m')
        ax.axvline(np.median(dists), color='black', linestyle='--',
                   linewidth=1.5, label=f'Median')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Final Distance to Target', fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_summary(rt_dists, ct_dists):
    """Print evaluation summary."""
    print("=" * 40)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 40)
    print(f"RT trials (n={len(rt_dists)}):")
    print(f"  Median dist : {np.median(rt_dists):.4f} m")
    print(f"  Mean dist   : {np.mean(rt_dists):.4f} m")
    print(f"  75th pctile : {np.percentile(rt_dists, 75):.4f} m")
    print(f"\nCT trials (n={len(ct_dists)}):")
    print(f"  Median dist : {np.median(ct_dists):.4f} m")
    print(f"  Mean dist   : {np.mean(ct_dists):.4f} m")
    print(f"  75th pctile : {np.percentile(ct_dists, 75):.4f} m")
    print("=" * 40)

if __name__ == '__main__':
    env = PursuitEnvironment(L=1.0, T=50, dt=0.02, v_max=0.0225)
 
    torch.manual_seed(42)
    n = 4  # example number of trajectories to plot
 
    rt_data = env.sample_RT(n)
    ct_data = env.sample_CT(n)
 
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    fig.suptitle('RT ve CT Trajectories', fontsize=13, fontweight='bold')
 
    for row, (data, label, color) in enumerate([
        (rt_data, 'RT', 'steelblue'),
        (ct_data, 'CT', 'darkorange'),
    ]):
        z_rnn_0, z_target_0, u_seq, z_target_final = data
        traj = reconstruct_target_trajectory(z_target_0, u_seq, env.dt)
 
        for i in range(n):
            ax = axes[row, i]
            ax.set_xlim(-0.55, 0.55)
            ax.set_ylim(-0.55, 0.55)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
 
            # area
            rect = plt.Rectangle((-0.5, -0.5), 1.0, 1.0,
                                  fill=False, edgecolor='gray', linewidth=1)
            ax.add_patch(rect)
 
            # target trajectory
            tx = traj[:, i, 0].numpy()
            ty = traj[:, i, 1].numpy()
            ax.plot(tx, ty, color=color, linewidth=1.8)
 
            # start (triangle) and end (square)
            ax.plot(tx[0],  ty[0],  '^', color=color, markersize=7, label='Start')
            ax.plot(tx[-1], ty[-1], 's', color='black', markersize=7, label='End')
 
            # RNN agent start
            rx0 = z_rnn_0[i, 0].item()
            ry0 = z_rnn_0[i, 1].item()
            ax.plot(rx0, ry0, '^', color='red', markersize=7, label='RNN start')
 
            ax.set_title(f'{label} {i+1}', fontsize=9)
 
    # legend 
    handles = [
        plt.Line2D([0], [0], marker='^', color='steelblue',  linestyle='None', label='Target start'),
        plt.Line2D([0], [0], marker='s', color='black',      linestyle='None', label='Target end'),
        plt.Line2D([0], [0], marker='^', color='red',        linestyle='None', label='RNN start'),
    ]
    axes[0, 0].legend(handles=handles, fontsize=7, loc='upper left')
 
    plt.tight_layout()
    plt.savefig('trajectories_example.png', dpi=150, bbox_inches='tight')
    print("saved: trajectories_example.png")
    plt.show()
