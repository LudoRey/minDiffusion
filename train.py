from typing import Dict, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM

import time
import os
import argparse

from data.axons_dataset import AxonsDataset

# idea of parameters for general train.py : dataset name, batch size
def train(dataset_name, n_epoch=100, batch_size=512, load_checkpoint=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date_time_string = time.strftime("%Y%m%d_%H%M", time.localtime()) 
    
    # Create Dataset and DataLoader
    dataset = globals()[dataset_name](root="./data")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    ddpm = DDPM(eps_model=NaiveUnet(1, 1, n_feat=128), betas=(1e-4, 0.02), n_T=1000)
    # Create checkpoint directory (or use previous one)
    if load_checkpoint is None:
        start_epoch = 0
        checkpoint_dir = "./checkpoints/"+dataset_name+"_"+date_time_string
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
    else:
        # Load model from checkpoint
        ddpm.load_state_dict(torch.load(load_checkpoint))
        # Keep the same checkpoint directory 
        last_slash_index = load_checkpoint.rfind('/')
        checkpoint_dir = load_checkpoint[:last_slash_index]
        # Retrieve epoch number from checkpoint path
        last_dot_index = load_checkpoint.rfind('.')
        last_underscore_index = load_checkpoint.rfind('_')
        start_epoch = int(load_checkpoint[last_underscore_index+1:last_dot_index]) + 1
        # Write info about training run in checkpoint directory
        with open(os.path.join(checkpoint_dir,"training_info.txt"), "a") as file:
            file.write("Training run resumed at "+ date_time_string + " from epoch " + str(start_epoch-1) + "\n")

    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    for i in range(start_epoch, n_epoch+1):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for batch in pbar:
            optim.zero_grad()
            x = batch['STED']
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        if i % 10 == 0 or i == n_epoch:
            ddpm.eval()
            with torch.no_grad():
                xh = ddpm.sample(4, (1, 32, 32), device)
                xset = torch.cat([xh, x[:4]], dim=0)
                grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
                # save grid image and model
                save_image(grid, checkpoint_dir + f"/sample_{i}.png")
                torch.save(ddpm.state_dict(), checkpoint_dir + f"/net_{i}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--n_epoch", type=int, default=100, help="The training run stops after the epoch count reaches n_epoch")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--load_checkpoint", default=None, help="The path to a .pth file which is used to initialize the model")
    args = parser.parse_args()

    train(dataset_name=args.dataset_name, n_epoch=args.n_epoch, batch_size=args.batch_size, load_checkpoint=args.load_checkpoint)