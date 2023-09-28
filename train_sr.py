from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm_cond import cDDPM

import time
import os
import argparse
import csv

from data.axons_dataset import AxonsDataset

def train_sr(opt):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date_time_string = time.strftime("%Y%m%d_%H%M", time.localtime()) 
    
    # Create Dataset and DataLoader
    dataset = globals()[opt.dataset_name](root="./data", crop=opt.crop)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    img_shape = dataset[0]["STED"].shape
    # Create test Dataset
    dataset = globals()[opt.dataset_name](root="./data", crop=opt.crop, phase="test")
    test = dataset[6]
    test_confocal = test["confocal"].unsqueeze(0).to(device)
    test_STED = test["STED"].unsqueeze(0).to(device)

    # Initialize model
    ddpm = cDDPM(eps_model=NaiveUnet(img_shape[0]*2, img_shape[0], n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    # Create checkpoint directory (or use previous one)
    if opt.load_checkpoint is None:
        start_epoch = 0
        checkpoint_dir = "./checkpoints/"+opt.dataset_name+"_"+date_time_string
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        # Create csv file to store loss
        with open(os.path.join(checkpoint_dir,"loss_log.csv"), mode="a", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Epoch', 'Loss'])
            writer.writeheader()
    else:
        # Keep the same checkpoint directory and start epoch count at load_epoch
        checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
        if opt.load_epoch is None:
            raise ValueError("Error: <load_epoch> must be set to an integer when <load_checkpoint> is not None.")
        start_epoch = opt.load_epoch + 1
        # Load model from checkpoint
        ddpm.load_state_dict(torch.load(os.path.join(checkpoint_dir, "net_"+str(opt.load_epoch)+".pth")))
        
    # Write info about training run in checkpoint directory
    with open(os.path.join(checkpoint_dir, "training_info.txt"), "a") as file:
        file.write("Run started at "+date_time_string+ " with training options:" + "\n")
        for key, value in vars(opt).items():
            file.write(f"{key}: {value} \n")
        file.write("\n")

    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    saved_loss_log = [] # resets every <opt.save_every> 
    for i in range(start_epoch, opt.n_epoch+1):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_epoch = 0
        for batch in pbar:
            optim.zero_grad()
            # Extract STED and confocal images from batch
            STED = batch['STED']
            confocal = batch['confocal']
            STED = STED.to(device)
            confocal = confocal.to(device)
            # Compute loss (here it is the output of the ddpm model)
            loss = ddpm(x=STED, x_cond=confocal)
            loss_epoch += loss.item()
            # Update parameters
            loss.backward()
            optim.step()

        # Store loss
        loss_epoch /= opt.batch_size
        saved_loss_log.append({'Epoch': i, 'Loss': loss_epoch})

        if i % opt.save_every == 0:
            ddpm.eval()
            with torch.no_grad():
                # Sample test
                samples = ddpm.sample(test_confocal, device)
                # Save grid
                xset = torch.cat((test_confocal, samples, test_STED), dim=0)
                grid = make_grid(xset, normalize=True, value_range=(-0.4, 1), nrow=4)
                save_image(grid, checkpoint_dir + f"/sample_{i}.png")
                # Save model
                torch.save(ddpm.state_dict(), checkpoint_dir + f"/net_{i}.pth")
                # Save loss in csv file
                with open(os.path.join(checkpoint_dir,"loss_log.csv"), mode="a", newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=['Epoch', 'Loss'])
                    writer.writerows(saved_loss_log)
                    saved_loss_log = []


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Options related to dataset
    parser.add_argument("--dataset_name", type=str, default="AxonsDataset", help="Name of Dataset class. Must be imported.")
    parser.add_argument("--crop", type=bool, default=False, help="Passed to Dataset argument.")
    # Options related to training
    parser.add_argument("--n_epoch", type=int, default=100, help="The training run stops after the epoch count reaches n_epoch")
    parser.add_argument("--batch_size", type=int, default=8)
    # Options related to loading and saving checkpoints (as well as saving other stuff)
    parser.add_argument("--load_checkpoint", type=str, default=None, help="The checkpoint folder to be loaded. Will continue to save data/log info in the same folder.")
    parser.add_argument("--load_epoch", type=int, default=None, help="The model ./checkpoint/<load_checkpoint>/net_<load_epoch>.pth will be loaded.")
    parser.add_argument("--save_every", type=int, default=10, help="Frequency of checkpoint/samples saves.")
    
    opt = parser.parse_args()

    train_sr(opt)
