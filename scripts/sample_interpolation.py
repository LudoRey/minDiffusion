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

def sample_interpolation(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(opt)
    model.to(device)

    # Load dataset and batch
    dataset = getattr(datasets, opt.dataset_name)(root="./data", phase="valid")
    n_samples = 50
    indices = [228]
    batch = next(iter(DataLoader(dataset, batch_sampler=DeterministicBatchSampler(indices)))) # could do collat_fn(dataset[indices]) but too lazy to implement collat_fn 
    x, y = batch["confocal"].to(device), batch["STED"].to(device)

    model.eval()
    with torch.no_grad():
        generator = torch.Generator(device).manual_seed(0)
        y_T0 = torch.randn(x.shape, generator=generator, device=device)
        y_T1 = torch.randn(x.shape, generator=generator, device=device)
        w = torch.linspace(0, 1, steps=n_samples, device=device)
        y_T = ((1-w[:,None,None,None])*y_T0 + w[:,None,None,None]*y_T1)/(torch.sqrt((1-w)**2+w**2))[:,None,None,None]
        y_DM = model.sample(x.repeat(n_samples,1,1,1), n_steps=100, sampling_mode='DDIM', y_T=y_T)
    
    # Denormalize
    x = dataset.denormalize(x, 'confocal')
    y_DM = dataset.denormalize(y_DM, 'STED')
    y = dataset.denormalize(y, 'STED')

    # Make grid
    y_DM = apply_colormap(y_DM, vmin=0, vmax=1)

    # Save frames
    frames_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}/interpolation"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir, exist_ok=True)
    for i in range(y_DM.shape[0]):
        save_image(y_DM[i], f"{frames_dir}/interpolation_{i}.png")

    # Make video from frames
    fps=10
    video_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}"
    subprocess.run(['ffmpeg', '-framerate', str(fps), '-i', f"{frames_dir}/interpolation_%d.png", '-b:v', '5M', '-y', f"{video_dir}/estimates.mp4"],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    opt = parse_options()
    sample_interpolation(opt)