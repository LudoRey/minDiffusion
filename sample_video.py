import torch
import argparse
import os
import subprocess

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
    dataset = globals()[opt.dataset_name](root="./data", phase="valid")
    data = dataset[228]
    confocal = data["confocal"].to(device).repeat(1,1,1,1)
    STED = data["STED"].to(device).repeat(1,1,1,1)

    img_shape = dataset[0]["STED"].shape
    # Load model
    model = cDDPM(denoising_net=Unet(img_shape[0]*2, img_shape[0], n_feat=128), denoising_target='eps')

    checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
    model.denoising_net.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"denoising_net_{opt.load_epoch}.pth")))
    model.to(device)

    model.eval()
    with torch.no_grad():
        # Sample test
        trajectory, estimates = model.sample(confocal, return_trajectory=True)

    # Keep 1 every 10 frames, apply colormap
    estimates = estimates[::10]
    estimates = dataset.denormalize(estimates, 'STED')
    estimates = apply_colormap(estimates)[:,0:3]
    trajectory = trajectory[::10]
    trajectory = dataset.denormalize(trajectory, 'STED')
    trajectory = apply_colormap(trajectory)[:,0:3]

    # Save frames
    frames_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}/frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir, exist_ok=True)
    for i in range(estimates.shape[0]):
        save_image(estimates[i], f"{frames_dir}/estimates_{i}.png")
        save_image(trajectory[i], f"{frames_dir}/trajectory_{i}.png")

    # Make video from frames
    fps=20
    video_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}"
    subprocess.run(['ffmpeg', '-framerate', str(fps), '-i', f"{frames_dir}/estimates_%d.png", '-b:v', '50M', '-y', f"{video_dir}/estimates.mp4"],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    subprocess.run(['ffmpeg', '-framerate', str(fps), '-i', f"{frames_dir}/trajectory_%d.png", '-b:v', '50M', '-y', f"{video_dir}/trajectory.mp4"],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    
if __name__ == "__main__":
    opt = parse_options()
    sample(opt)