import torch
import argparse
import os

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from models.unet import Unet
from models.cddm import *

from datasets.axons_dataset import AxonsDataset
from datasets.dendrites_dataset import DendritesDataset
from misc.options import parse_options
from misc.utils import apply_colormap

def sample(opt, n_samples=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test Dataset
    dataset = globals()[opt.dataset_name](root="./data", crop=opt.crop, phase="valid")
    test = dataset[0]
    test_confocal = test["confocal"].to(device).repeat(n_samples,1,1,1)
    test_STED = test["STED"].to(device).repeat(n_samples,1,1,1)

    img_shape = dataset[0]["STED"].shape
    # Load model
    ddpm = cDDPM(eps_model=Unet(img_shape[0]*2, img_shape[0], n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
    ddpm.load_state_dict(torch.load(os.path.join(checkpoint_dir, "net_"+str(opt.load_epoch)+".pth")))
    ddpm.to(device)

    ddpm.eval()
    with torch.no_grad():
        # Sample test
        trajectory, estimates = ddpm.sample(test_confocal, device, return_trajectory=True)

    # Concatenate along width axis
    frames = torch.cat([trajectory, estimates], dim=3)
    #frames = trajectory
    # Only save one of every 10 intermediate results
    frames = frames[::10]
    # Convert to color images
    frames = apply_colormap(frames, vmin=-0.8, vmax=7)
    # Convert back to uint8
    frames = (frames*255).type(torch.uint8)
    # Convert to TxHxWxC format
    frames = frames.permute((0,2,3,1))[:,:,:,:3]
    # Save frames as tensor
    torch.save(frames, "samples/frames_"+opt.load_checkpoint+"_"+str(opt.load_epoch)+".pt")
    # Save video (requires OpenCV package)
    #write_video("samples/generation_"+opt.load_checkpoint+"_"+str(opt.load_epoch)+".avi", frames, fps=20)

    
if __name__ == "__main__":
    opt = parse_options()
    sample(opt)