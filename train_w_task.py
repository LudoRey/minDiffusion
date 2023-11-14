from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision.utils import save_image, make_grid

from importlib import import_module
from models.cddm import TADM
from models.segmentation.UNet import UNet

import os

import datasets
from misc.options import parse_options
from misc.utils import *
from misc.logger import Logger

def train(opt):
    
    logger = Logger(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create training Dataset and DataLoader
    Dataset = getattr(datasets, opt.dataset_name)
    train_dataset = Dataset(root="./data", phase="train", match_means=False)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    # Create validation Dataset and Dataloader
    valid_dataset = Dataset(root="./data", phase="valid", match_means=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    # Get test image
    test = valid_dataset[228]
    test_confocal = test["confocal"].unsqueeze(0).to(device) # unsqueeze needed for batch dimension
    test_STED = test["STED"].unsqueeze(0).to(device)

    # Initialize/load denoising network
    Unet = getattr(import_module(f"models.denoising.{opt.denoising_net_name}"), "Unet")
    denoising_net = Unet(in_channels=2, out_channels=1)
    if opt.load_checkpoint is not None:
        loaded_checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
        denoising_net.load_state_dict(torch.load(os.path.join(loaded_checkpoint_dir, f"denoising_net_{opt.load_epoch}.pth")))
    # Initialize/load segmentation network
    task_net = UNet(in_channels=1, out_channels=2).to(device)
    state_dict = torch.load("./checkpoints/STEDActinFCN_Dendrites/params.net", map_location=device)
    task_net.load_state_dict(state_dict)

    model = TADM(denoising_net, task_net, denoising_target=opt.denoising_target)
    model.to(device)

    optim = torch.optim.Adam(model.denoising_net.parameters(), lr=opt.learning_rate)
    if opt.patience is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=opt.patience)

    lambda_TA = torch.Tensor([opt.lambda_TA]).to(device)
    for i in range(opt.load_epoch+1, opt.n_epoch+1):

        print(f"Epoch {i}/{opt.n_epoch} : ")

        model.train()
        train_denoising_loss, train_task_loss = 0, 0
        for batch in tqdm(train_dataloader, desc="Training..."):
            optim.zero_grad()
            # Extract STED and confocal images from batch
            STED = batch['STED'].to(device)
            confocal = batch['confocal'].to(device)
            seg_GT = torch.cat([batch['seg_GTrings'], batch['seg_GTfibers']], dim=1).to(device)
            # Compute losses
            denoising_loss, task_loss = model(y=STED, x=confocal, task_target=seg_GT)
            total_loss = denoising_loss + lambda_TA*task_loss
            # Update parameters
            total_loss.backward()
            optim.step()
            # Keep track of losses
            train_denoising_loss += denoising_loss.item()
            train_task_loss += task_loss.item()
        train_denoising_loss /= len(train_dataloader)
        train_task_loss /= len(train_dataloader)
        print(f'Denoising loss : {train_denoising_loss}, Task loss : {train_task_loss}')
        
        model.eval()
        seed = 0
        valid_denoising_loss, valid_task_loss, valid_total_loss = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Computing validation loss..."):
                # Load inputs
                STED = batch['STED'].to(device)
                confocal = batch['confocal'].to(device)
                seg_GT = torch.cat([batch['seg_GTrings'], batch['seg_GTfibers']], dim=1).to(device)
                # Compute loss
                denoising_loss, task_loss = model(y=STED, x=confocal, task_target=seg_GT, seed=seed)
                total_loss = denoising_loss + lambda_TA*task_loss
                # Update seed
                seed += 1
                # Keep track of losses
                valid_denoising_loss += denoising_loss.item()
                valid_task_loss += task_loss.item()
                valid_total_loss += total_loss.item()
                
        valid_denoising_loss /= len(valid_dataloader)
        valid_task_loss /= len(valid_dataloader)
        valid_total_loss /= len(valid_dataloader)
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
            scheduler.step(valid_total_loss)

        if i % opt.save_every == 0:
            model.eval()
            with torch.no_grad():
                # Get test image
                test = valid_dataset[228]
                test_confocal = test["confocal"].unsqueeze(0).to(device) # unsqueeze needed for batch dimension
                test_STED = test["STED"].unsqueeze(0).to(device)
                test_segGT = torch.cat([test['seg_GTrings'], test['seg_GTfibers']], dim=0).unsqueeze(0).to(device)
                # Make sample grid
                test_sample = model.sample(test_confocal, seed=42)
                
                test_confocal = valid_dataset.denormalize(test_confocal, 'confocal')
                test_sample = valid_dataset.denormalize(test_sample, 'STED')
                test_STED = valid_dataset.denormalize(test_STED, 'STED')

                test_segGen = model.task_net(test_sample)*0.5 + 0.5
                test_segSTED = model.task_net((test_STED-0.5)/0.5)*0.5 + 0.5

                test_seg = torch.cat((test_segGT, test_segGen, test_segSTED)).cpu()
                test_seg = combine_purple_green(test_seg[:,1:2], test_seg[:,0:1])
                test_images = torch.cat((test_confocal, test_sample, test_STED)).cpu()
                test_images = apply_colormap(test_images, vmin=0, vmax=1)
                grid = make_grid(torch.cat((test_images, test_seg)), nrow=3)
            # Save sample grid, model, loss
            save_image(grid, logger.checkpoint_dir + f"/sample_{i}.png")
            torch.save(model.denoising_net.state_dict(), logger.checkpoint_dir + f"/denoising_net_{i}.pth")
            logger.save_csv_log()


if __name__ == "__main__":
    opt = parse_options()
    train(opt)
