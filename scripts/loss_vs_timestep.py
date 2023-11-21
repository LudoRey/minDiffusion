import torch
import os
import pandas as pd
from torch.utils.data import DataLoader

import datasets
from datasets.utils import DeterministicBatchSampler
from misc.options import parse_options
from misc.utils import *
from . import load_model

from tqdm import tqdm

def loss_vs_timestep(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(opt)
    model.to(device)

    # Load dataset and batch
    dataset = getattr(datasets, opt.dataset_name)(root="./data", phase="valid")

    generator = torch.Generator().manual_seed(0)
    indices = torch.randint(0, len(dataset), size=(32,), generator=generator).tolist()
    batch = next(iter(DataLoader(dataset, batch_sampler=DeterministicBatchSampler(indices)))) # could do collat_fn(dataset[indices]) but too lazy to implement collat_fn 
    x, y = batch["confocal"].to(device), batch["STED"].to(device)

    n = 51 # odd number
    tt = get_symmetric_logspace_timesteps(n, model.T, start_at_one=True)

    loss_per_t = torch.zeros(n)
    model.eval()
    with torch.no_grad():
        for i in range(n):
            t = tt[i].repeat(x.shape[0])
            t, eps, y_t, estimated_target = model.forward_prediction(y, x, seed=42, t=t)
            estimated_y = model.get_y_from_target(estimated_target, y_t, t)
            loss_per_pixel = torch.nn.MSELoss(reduction='none')(y, estimated_y) # BxCxHxW tensor
            loss_per_image = torch.mean(loss_per_pixel, dim=(1,2,3)) # B tensor
            loss_per_t[i] += torch.sum(loss_per_image).item()
    loss_per_t /= len(indices)

    save_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    loss_vs_t = pd.DataFrame({'Timestep' : tt, 'Loss image' : loss_per_t}) 
    loss_vs_t.to_csv(f"{save_dir}/loss_vs_t_{opt.denoising_target}.csv", header=True, mode='w', index=False) 

if __name__=="__main__":
    opt = parse_options()
    loss_vs_timestep(opt)