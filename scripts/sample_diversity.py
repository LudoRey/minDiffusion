import torch
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import datasets
from datasets.utils import DeterministicBatchSampler
from misc.options import parse_options
from misc.utils import *
from . import load_model

def sample_diversity(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(opt)
    model.to(device)

    # Load dataset and batch
    dataset = getattr(datasets, opt.dataset_name)(root="./data", phase="valid")
    indices = [228]*16
    batch = next(iter(DataLoader(dataset, batch_sampler=DeterministicBatchSampler(indices)))) # could do collat_fn(dataset[indices]) but too lazy to implement collat_fn 
    x, y = batch["confocal"].to(device), batch["STED"].to(device)

    model.eval()
    with torch.no_grad():
        y_DM = model.sample(x, seed=42)
    
    # Denormalize
    x = dataset.denormalize(x, 'confocal')
    y_DM = dataset.denormalize(y_DM, 'STED')
    y = dataset.denormalize(y, 'STED')

    # Segmentation (Need to scale inputs for the segmentation network)
    factor = 0.8*2.25 # Anthony's numbers (0.8*max(dataset))
    factor *= 1/0.3166458 # Match means linear fit in training dataset
    with torch.no_grad():
        y_DM_task = model.task_net(y_DM/factor)
        y_task = model.task_net(y/factor)

    # Make grid
    n_samples = len(indices)
    images = torch.cat([y_DM]).cpu()
    images = apply_colormap(images, vmin=0, vmax=1)
    seg = torch.cat([y_DM_task]).cpu()
    seg = combine_purple_green(seg[:,1:2], seg[:,0:1])
    grid = make_grid(torch.cat((images, seg)), nrow=4)

    save_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}"
    tifs_dir = os.path.join(save_dir, "tifs")
    if not os.path.exists(tifs_dir):
        os.makedirs(tifs_dir, exist_ok=True)
    save_image(grid, f"{save_dir}/sample_diversity.png")
    
    for i in range(n_samples):
        save_tensor_as_tif(x[i], f"{tifs_dir}/valid_{indices[i]}_confocal.tif")
        save_tensor_as_tif(y[i], f"{tifs_dir}/valid_{indices[i]}_STED.tif")
        save_tensor_as_tif(y_task[i], f"{tifs_dir}/valid_{indices[i]}_segSTED.tif")
        save_tensor_as_tif(y_DM[i], f"{tifs_dir}/valid_{indices[i]}_{i}_DM.tif")
        save_tensor_as_tif(y_DM_task[i], f"{tifs_dir}/valid_{indices[i]}_segDM.tif")

if __name__ == "__main__":
    opt = parse_options()
    sample_diversity(opt)