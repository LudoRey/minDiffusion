from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision.utils import save_image, make_grid

from models.unet import Unet
from models.cddm import cDDPM

import os

from datasets.axons_dataset import AxonsDataset
from datasets.dendrites_dataset import DendritesDataset
from misc.options import parse_options
from misc.utils import *
from misc.logger import Logger

def train(opt):
    
    logger = Logger(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create training Dataset and DataLoader
    train_dataset = globals()[opt.dataset_name](root="./data", phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    # Create validation Dataset and Dataloader
    valid_dataset = globals()[opt.dataset_name](root="./data", phase="valid")
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    # Get test image
    test = valid_dataset[228]
    test_confocal = test["confocal"].unsqueeze(0).to(device) # unsqueeze needed for batch dimension
    test_STED = test["STED"].unsqueeze(0).to(device)

    img_shape = train_dataset[0]["STED"].shape

    # Initialize model
    model = cDDPM(denoising_net=Unet(img_shape[0]*2, img_shape[0], n_feat=128), denoising_target='y')
    # Load model from checkpoint
    if opt.load_checkpoint is not None:
        loaded_checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
        model.denoising_net.load_state_dict(torch.load(os.path.join(loaded_checkpoint_dir, f"denoising_net_{opt.load_epoch}.pth")))
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    if opt.patience is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=opt.patience)

    for i in range(opt.load_epoch+1, opt.n_epoch+1):

        print(f"Epoch {i}/{opt.n_epoch} : ")

        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc="Training..."):
            optim.zero_grad()
            # Extract STED and confocal images from batch
            STED = batch['STED'].to(device)
            confocal = batch['confocal'].to(device)
            # Compute loss (here it is the output of the model)
            loss = model(y=STED, x=confocal)
            train_loss += loss.item()
            # Update parameters
            loss.backward()
            optim.step()
        train_loss /= len(train_dataloader)
        
        model.eval()
        seed = 0
        valid_loss = 0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Computing validation loss..."):
                STED = batch['STED'].to(device)
                confocal = batch['confocal'].to(device)
                loss = model(y=STED, x=confocal, seed=seed)
                seed += 1
                valid_loss += loss.item()
        valid_loss /= len(valid_dataloader)

        # Store loss and lr
        lr = optim.param_groups[0]['lr']
        logger.log({'Epoch': i, 'Train loss': train_loss, 'Valid loss' : valid_loss, 'Learning rate' : lr})

        # Update LR
        if opt.patience is not None:
            scheduler.step(valid_loss)

        if i % opt.save_every == 0:
            model.eval()
            with torch.no_grad():
                # Make sample grid
                samples = model.sample(test_confocal)
                d_confocal = valid_dataset.denormalize(test_confocal, 'confocal')
                d_samples = valid_dataset.denormalize(samples, 'STED')
                d_STED = valid_dataset.denormalize(test_STED, 'STED')
                xset = torch.cat((d_confocal, d_samples, d_STED), dim=0)
                yset = apply_colormap(xset, vmin=0, vmax=1)
                grid = make_grid(yset, nrow=3)
            # Save sample grid, model, loss
            save_image(grid, logger.checkpoint_dir + f"/sample_{i}.png")
            torch.save(model.denoising_net.state_dict(), logger.checkpoint_dir + f"/denoising_net_{i}.pth")
            logger.save_csv_log()


if __name__ == "__main__":
    opt = parse_options()
    train(opt)
