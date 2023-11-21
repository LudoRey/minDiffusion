import torch
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import datasets
from datasets.utils import DeterministicBatchSampler
from misc.options import parse_options
from misc.utils import *
from . import load_model

def sample_trajectory(opt):
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
    n = 9
    t = get_linspace_timesteps(n, model.T)
    x = x.repeat(n,1,1,1)
    y = y.repeat(n,1,1,1)
    y_t = y_t[t]
    estimated_y = estimated_y[t]

    # Denormalize
    x = dataset.denormalize(x, 'confocal')
    y_t = dataset.denormalize(y_t, 'STED')
    estimated_y = dataset.denormalize(estimated_y, 'STED')
    y = dataset.denormalize(y, 'STED')

    # Segmentation (Need to scale inputs for the segmentation network)
    factor = 0.8*2.25 # Anthony's numbers (0.8*max(dataset))
    factor *= 1/0.3166458 # Match means linear fit in training dataset
    with torch.no_grad():
        estimated_y_task = model.task_net(estimated_y/factor)
        y_task = model.task_net(y/factor)

    # Make grid
    images = torch.cat((x, y_t, estimated_y, y)).cpu()
    images = apply_colormap(images, vmin=0, vmax=1)
    seg = torch.cat((estimated_y_task, y_task)).cpu()
    seg = combine_purple_green(seg[:,1:2], seg[:,0:1])
    grid = make_grid(torch.cat((images, seg)), nrow=n)

    save_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_image(grid, f"{save_dir}/trajectory.png")
    
if __name__ == "__main__":
    opt = parse_options()
    sample_trajectory(opt)