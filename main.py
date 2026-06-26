# main script to start training the model

import torch
import torch.optim as optim
import numpy as np
import os
from model import PursuitRNN, pursuit_loss
from environment import PursuitEnvironment


# config 
CONFIG = {
    'N'               : 1024,      # dim
    'dt'              : 0.02,      # timestep
    'v_max'           : 1.125,     # max speed (m/s!!!!!)
    'T'               : 50,        # tot timesteps
    'L'               : 1.0,       # area

    # training part
    'epochs'          : 100,
    'batches_per_epoch': 1000,
    'batch_size'      : 400,
    'lr'              : 1e-4,    # learning rate
    'weight_decay'    : 1e-4,      # weight decay for optimizer (WHY/HOW????)
    'percent_CT'      : 0.25,      # %25 CT, %75 RT

    'save_dir'        : 'checkpoints',
    'save_every'      : 100,        # save every 10 epochs
    'log_every'       : 100,       # log every 100 batches
}

# training 
def train(config):
    # dir for checkpoints
    os.makedirs(config['save_dir'], exist_ok=True)

    # model
    model = PursuitRNN(N=config['N'],dt=config['dt'],v_max=config['v_max'])

    #environment
    env = PursuitEnvironment(L=config['L'],T=config['T'],dt=config['dt'],v_max=config['v_max'])

    # optimizer - Adam + weight decay 
    optimizer = optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])

    # losses
    epoch_losses = []

    print("training...")
    print(f"  N={config['N']}, T={config['T']}, dt={config['dt']}")
    print(f"  epochs={config['epochs']}, batch_size={config['batch_size']}")
    print(f"  lr={config['lr']}, weight_decay={config['weight_decay']}")
    print()

    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        for batch_idx in range(config['batches_per_epoch']):

            # 1 - data
            z_rnn_0, z_target_0, u_seq, z_target_final = env.sample_batch(
            batch_size=config['batch_size'],
            percent_CT=config['percent_CT']
        )

            # 2 - forward
            z_rnn_final, r_seq, v_seq = model(z_rnn_0, z_target_0, u_seq)

            # 3 - loss
            # L_end = ||z_RNN(T) - z_target(T)||^2 
            loss = pursuit_loss(z_rnn_final, z_target_final)

            # 4 - backpropagation
            optimizer.zero_grad()   # previous gradients kept in the stack make them zero before backward
            loss.backward()         # calculate gradients for all parameters in the model with respect to the loss

            # 5 - update
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()        # weight update: w = w - lr * dw (with Adam)

            epoch_loss += loss.item()

            # Log
            if (batch_idx + 1) % config['log_every'] == 0:
                avg = epoch_loss / (batch_idx + 1)
                dist = avg ** 0.5   # loss = dist^2, dist = sqrt(loss)
                print(f"  Epoch {epoch+1:3d} | Batch {batch_idx+1:4d} "
                      f"| Loss: {avg:.4f} | Dist: {dist:.4f} m")

        # mean per epoch
        avg_epoch_loss = epoch_loss / config['batches_per_epoch']
        avg_dist = avg_epoch_loss ** 0.5
        epoch_losses.append(avg_epoch_loss)

        print(f"Epoch {epoch+1:3d} done | "
              f"Mean Loss: {avg_epoch_loss:.4f} | "
              f"Mean Dist: {avg_dist:.4f} m")

        # checkpoint save
        if (epoch + 1) % config['save_every'] == 0:
            path = os.path.join(
                config['save_dir'],
                f"model_epoch{epoch+1}.pt"
            )
            torch.save({
                'epoch'      : epoch + 1,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'losses'     : epoch_losses,
                'config'     : config,
            }, path)
            print(f"  -> Saved: {path}")

    print("\ntraining completed")

    # final model save
    torch.save({
        'epoch'      : config['epochs'],
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'losses'     : epoch_losses,
        'config'     : config,
    }, os.path.join(config['save_dir'], 'model_final.pt'))

    return model, epoch_losses


# main
if __name__ == '__main__':
    model, losses = train(CONFIG)

    print(f"\nfirst epoch loss : {losses[0]:.4f}")
    print(f"last epoch loss : {losses[-1]:.4f}")