from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision.utils import save_image, make_grid

from importlib import import_module
from models.diffusion import TADM, cDM
from models.segmentation.unet import Unet

import datasets
from misc.options import parse_options
from misc.utils import *
from misc.logger import Logger

def train(opt):
    logger = Logger(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create training Dataset and DataLoader
    Dataset = getattr(datasets, opt.dataset_name)
    train_dataset = Dataset(root="./data", phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    # Create validation Dataset and Dataloader
    valid_dataset = Dataset(root="./data", phase="valid")
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)

    # Initialize/load denoising network
    UnetInfusedTimestep = getattr(import_module(f"models.denoising.{opt.denoising_net_name}"), "Unet")
    denoising_net = UnetInfusedTimestep(in_channels=2, out_channels=2 if opt.denoising_target=="both" else 1)
    if opt.load_checkpoint is not None:
        denoising_net.load_state_dict(torch.load(f"./checkpoints/{opt.load_checkpoint}/denoising_net_{opt.load_epoch}.pth"))
    # Initialize/load segmentation network
    task_net = Unet(in_channels=1, out_channels=2).to(device)
    task_net.load_state_dict(torch.load("./checkpoints/STEDActinFCN_Dendrites/params.net"))

    model = TADM(denoising_net, task_net, opt.task_weight, opt.denoising_target, opt.loss_weighting)
    model.to(device)

    optim = torch.optim.Adam(model.denoising_net.parameters(), lr=opt.learning_rate)
    if opt.patience is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=opt.patience)
  
    for i in range(opt.load_epoch+1, opt.n_epoch+1):
        print(f"Epoch {i}/{opt.n_epoch} : ")

        model.train()
        train_denoising_loss, train_task_loss = 0, 0
        for batch in tqdm(train_dataloader, desc="Training..."):
            optim.zero_grad()
            # Extract STED and confocal images from batch
            y = batch['STED'].to(device)
            x = batch['confocal'].to(device)
            # Compute losses
            denoising_loss, task_loss = model(y, x)
            total_loss = denoising_loss + task_loss
            # Update parameters
            total_loss.backward()
            optim.step()
            # Keep track of losses
            train_denoising_loss += denoising_loss.item()*len(y)
            train_task_loss += task_loss.item()*len(y)
        train_denoising_loss /= len(train_dataset)
        train_task_loss /= len(train_dataset)
        print(f'Denoising loss : {train_denoising_loss}, Task loss : {train_task_loss}')
        
        model.eval()
        seed = 0 # To make reverse process deterministic
        valid_denoising_loss, valid_task_loss = 0, 0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Computing validation loss..."):
                # Load inputs
                y = batch['STED'].to(device)
                x = batch['confocal'].to(device)
                # Compute losses
                denoising_loss, task_loss = model(y, x, seed=seed)
                # Update seed
                seed += 1
                # Keep track of losses
                valid_denoising_loss += denoising_loss.item()*len(y)
                valid_task_loss += task_loss.item()*len(y)
        valid_denoising_loss /= len(valid_dataset)
        valid_task_loss /= len(valid_dataset)
        print(f'Denoising loss : {valid_denoising_loss}, Task loss : {valid_task_loss}')

        # Store loss and lr
        lr = optim.param_groups[0]['lr']
        logger.log({'Epoch': i,
                    'Train denoising loss': train_denoising_loss, 
                    'Train task loss': train_task_loss,
                    'Valid denoising loss' : valid_denoising_loss,
                    'Valid task loss' : valid_task_loss,
                    'Learning rate' : lr})

        # Update LR
        if opt.patience is not None:
            scheduler.step(valid_denoising_loss+valid_task_loss)

        if i % opt.save_every == 0:
            # Get test image
            data = valid_dataset[228]
            x = data["confocal"].unsqueeze(0).to(device) # unsqueeze needed for batch dimension
            y = data["STED"].unsqueeze(0).to(device)
            # Generate image
            model.eval()
            with torch.no_grad():
                y_t, estimated_y = model.sample(x, return_trajectory=True, seed=42)
            y_t = y_t[::200]
            estimated_y = estimated_y[::200]
            # Denormalize
            x = valid_dataset.denormalize(x, 'confocal')
            y_t = valid_dataset.denormalize(y_t, 'STED')
            estimated_y = valid_dataset.denormalize(estimated_y, 'STED')
            y = valid_dataset.denormalize(y, 'STED')
            # Need to scale inputs for the segmentation network
            factor = 5.684 # 0.8*2.25 (0.8*max(dataset)) divided by 0.3166458 (match means avg correction)
            with torch.no_grad():
                estimated_y_task = model.task_net(estimated_y/factor)
                y_task = model.task_net(y/factor)
            # Make grid
            images = torch.cat((x.repeat(7,1,1,1), y_t, y, estimated_y, y)).cpu()
            images = apply_colormap(images, vmin=0, vmax=1)
            seg = torch.cat((estimated_y_task, y_task)).cpu()
            seg = combine_purple_green(seg[:,1:2], seg[:,0:1])
            grid = make_grid(torch.cat((images, seg)), nrow=7)
            # Save sample grid, model, loss
            save_image(grid, logger.checkpoint_dir + f"/sample_{i}.png")
            torch.save(model.denoising_net.state_dict(), logger.checkpoint_dir + f"/denoising_net_{i}.pth")
            logger.save_csv_log()


if __name__ == "__main__":
    opt = parse_options()
    train(opt)
