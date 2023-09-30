from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision.utils import save_image, make_grid

from models.unet import NaiveUnet
from models.ddpm_cond import cDDPM

import os

from datasets.axons_dataset import AxonsDataset
from misc.options import parse_options
from misc.utils import *

def train_sr(opt):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If opt.load_checkpoint is None : create new directory. Otherwise use the same.
    checkpoint_dir = get_checkpoint_dir(opt)

    # Write options to text file
    write_training_info(opt, os.path.join(checkpoint_dir, "training_info.txt"))
    
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
    if opt.load_checkpoint is not None:
        # Load model from checkpoint
        ddpm.load_state_dict(torch.load(os.path.join(checkpoint_dir, "net_"+str(opt.load_epoch)+".pth")))
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True)

    for i in range(opt.load_epoch+1, opt.n_epoch+1):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_epoch = 0
        logged_loss = []
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
        logged_loss.append({'Epoch': i, 'Loss': loss_epoch})

        # Update LR 
        scheduler.step(loss_epoch)

        if i % opt.save_every == 0:
            ddpm.eval()
            with torch.no_grad():
                # Make sample grid
                samples = ddpm.sample(test_confocal, device)
                xset = torch.cat((test_confocal, samples, test_STED), dim=0)
                grid = make_grid(xset, normalize=True, value_range=(-0.4, 1), nrow=4)
            # Save sample grid, model, loss
            save_image(grid, checkpoint_dir + f"/sample_{i}.png")
            torch.save(ddpm.state_dict(), checkpoint_dir + f"/net_{i}.pth")
            save_loss(logged_loss, checkpoint_dir + "/loss_log.csv")
            logged_loss = []


if __name__ == "__main__":
    opt = parse_options()
    train_sr(opt)
