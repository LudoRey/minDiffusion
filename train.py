from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision.utils import save_image, make_grid

from models.unet import Unet
from models.ddpm_cond import cDDPM

import os

from datasets.axons_dataset import AxonsDataset
from datasets.dendrites_dataset import DendritesDataset
from misc.options import parse_options
from misc.utils import *

def train(opt):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If opt.load_checkpoint is None : create new directory. Otherwise use the same.
    checkpoint_dir = get_checkpoint_dir(opt)

    # Write options to text file
    write_training_info(opt, os.path.join(checkpoint_dir, "training_info.txt"))
    
    # Create training Dataset and DataLoader
    train_dataset = globals()[opt.dataset_name](root="./data", crop=opt.crop, phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    # Create validation Dataset and Dataloader
    valid_dataset = globals()[opt.dataset_name](root="./data", crop=opt.crop, phase="valid")
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size)
    # Get test image
    test = valid_dataset[0]
    test_confocal = test["confocal"].unsqueeze(0).to(device)
    test_STED = test["STED"].unsqueeze(0).to(device)

    img_shape = train_dataset[0]["STED"].shape
    # Initialize model
    ddpm = cDDPM(eps_model=Unet(img_shape[0]*2, img_shape[0], n_feat=128), betas=(1e-4, 0.02), n_T=1000)
    # Load model from checkpoint
    if opt.load_checkpoint is not None:
        ddpm.load_state_dict(torch.load(os.path.join(checkpoint_dir, "net_"+str(opt.load_epoch)+".pth")))
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

    logger = []
    for i in range(opt.load_epoch+1, opt.n_epoch+1):

        print(f"Epoch {i}/{opt.n_epoch} : ")

        ddpm.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc="Training..."):
            optim.zero_grad()
            # Extract STED and confocal images from batch
            STED = batch['STED'].to(device)
            confocal = batch['confocal'].to(device)
            # Compute loss (here it is the output of the ddpm model)
            loss = ddpm(x=STED, x_cond=confocal)
            train_loss += loss.item()
            # Update parameters
            loss.backward()
            optim.step()
        train_loss /= len(train_dataloader)
        
        ddpm.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Computing validation loss..."):
                STED = batch['STED'].to(device)
                confocal = batch['confocal'].to(device)
                loss = ddpm(x=STED, x_cond=confocal)
                valid_loss += loss.item()
        valid_loss /= len(valid_dataloader)

        # Store loss and lr
        lr = optim.param_groups[0]['lr']
        logger.append({'Epoch': i, 'Train loss': train_loss, 'Valid loss' : valid_loss, 'Learning rate' : lr})

        # Update LR
        scheduler.step(valid_loss)

        if i % opt.save_every == 0:
            ddpm.eval()
            with torch.no_grad():
                # Make sample grid
                samples = ddpm.sample(test_confocal, device)
                xset = torch.cat((test_confocal, samples, test_STED), dim=0)
                yset = apply_colormap(xset, vmin=-0.8, vmax=7)
                grid = make_grid(yset, nrow=4)
            # Save sample grid, model, loss
            save_image(grid, checkpoint_dir + f"/sample_{i}.png")
            torch.save(ddpm.state_dict(), checkpoint_dir + f"/net_{i}.pth")
            save_logger(logger, checkpoint_dir + "/logger.csv")
            logger = []


if __name__ == "__main__":
    opt = parse_options()
    train_sr(opt)
