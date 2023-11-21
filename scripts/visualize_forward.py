import torch
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import datasets
from datasets.utils import DeterministicBatchSampler
from misc.options import parse_options
from misc.utils import *
from . import load_model

def visualize_forward(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(opt)
    model.to(device)

    # Load dataset and batch
    dataset = getattr(datasets, opt.dataset_name)(root="./data", phase="valid")
    indices = [228]
    batch = next(iter(DataLoader(dataset, batch_sampler=DeterministicBatchSampler(indices)))) # could do collat_fn(dataset[indices]) but too lazy to implement collat_fn 
    x, y = batch["confocal"].to(device), batch["STED"].to(device)
    
    n = 9
    t = get_symmetric_logspace_timesteps(n, model.T, start_at_one=True)
    x = x.repeat(n,1,1,1)
    y = y.repeat(n,1,1,1)

    model.eval()
    with torch.no_grad():
        t, eps, y_t, estimated_target = model.forward_prediction(y, x, seed=42, t=t)
    estimated_y = model.get_y_from_target(estimated_target, y_t, t)
    estimated_eps = model.get_eps_from_target(estimated_target, y_t, t)

    # Denormalize
    x = dataset.denormalize(x, 'confocal')
    y_t = dataset.denormalize(y_t, 'STED')
    estimated_y = dataset.denormalize(estimated_y, 'STED')
    estimated_eps = dataset.denormalize(estimated_eps, 'STED')
    y = dataset.denormalize(y, 'STED')
    eps = dataset.denormalize(eps, 'STED')

    # Segmentation (Need to scale inputs for the segmentation network)
    factor = 0.8*2.25 # Anthony's numbers (0.8*max(dataset))
    factor *= 1/0.3166458 # Match means linear fit in training dataset
    with torch.no_grad():
        estimated_y_task = model.task_net(estimated_y/factor)
        y_task = model.task_net(y/factor)

    # Make grid
    images = torch.cat((x, y_t, estimated_y, estimated_eps)).cpu()
    images = apply_colormap(images, vmin=0, vmax=1)
    seg = torch.cat((estimated_y_task, y_task)).cpu()
    seg = combine_purple_green(seg[:,1:2], seg[:,0:1])
    grid = make_grid(torch.cat((images, seg)), nrow=n)

    save_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_image(grid, f"{save_dir}/forward_prediction.png")

if __name__ == "__main__":
    opt = parse_options()
    visualize_forward(opt)