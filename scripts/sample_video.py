import torch
import os
import subprocess
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import datasets
from datasets.utils import DeterministicBatchSampler
from misc.options import parse_options
from misc.utils import *
from . import load_model

def sample_video(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(opt)
    model.to(device)

    # Load dataset and batch
    dataset = getattr(datasets, opt.dataset_name)(root="./data", phase="valid")
    indices = [228]
    batch = next(iter(DataLoader(dataset, batch_sampler=DeterministicBatchSampler(indices)))) # could do collat_fn(dataset[indices]) but too lazy to implement collat_fn 
    x, y = batch["confocal"].to(device), batch["STED"].to(device)

    model.eval()
    with torch.no_grad():
        y_t, estimated_y = model.sample(x, return_trajectory=True, seed=42)

    # Select subset of timesteps
    n = 100
    t = get_linspace_timesteps(n, model.T)
    y_t = y_t[t]
    estimated_y = estimated_y[t]

    # Denormalize
    y_t = dataset.denormalize(y_t, 'STED')
    estimated_y = dataset.denormalize(estimated_y, 'STED')

    # Apply colormap
    y_t = apply_colormap(y_t)
    estimated_y = apply_colormap(estimated_y)
    
    # Save frames
    frames_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}/frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir, exist_ok=True)
    for i in range(y_t.shape[0]):
        save_image(estimated_y[i], f"{frames_dir}/estimates_{i}.png")
        save_image(y_t[i], f"{frames_dir}/trajectory_{i}.png")

    # Make video from frames
    fps=20
    video_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}"
    subprocess.run(['ffmpeg', '-framerate', str(fps), '-i', f"{frames_dir}/estimates_%d.png", '-b:v', '5M', '-y', f"{video_dir}/estimates.mp4"],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    subprocess.run(['ffmpeg', '-framerate', str(fps), '-i', f"{frames_dir}/trajectory_%d.png", '-b:v', '5M', '-y', f"{video_dir}/trajectory.mp4"],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    
if __name__ == "__main__":
    opt = parse_options()
    sample_video(opt)