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
    dataset = globals()[opt.dataset_name](root="./data", phase="valid")
    data = dataset[228]
    confocal = data["confocal"].to(device).repeat(1,1,1,1)
    STED = data["STED"].to(device).repeat(1,1,1,1)

    img_shape = dataset[0]["STED"].shape
    # Load model
    model = cDDPM(eps_model=Unet(img_shape[0]*2, img_shape[0], n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "net_"+str(opt.load_epoch)+".pth")))
    model.to(device)

    model.eval()
    with torch.no_grad():
        # Sample test
        trajectory, estimates = model.sample(confocal, return_trajectory=True)

    frames = trajectory
    # (optional) Stack estimates
    frames = torch.stack([trajectory, estimates])
    # Only save one of every 100 intermediate results
    #frames = frames[::100]
    # Save frames as tensor
    save_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(frames, f"{save_dir}/frames.pt")

    
if __name__ == "__main__":
    opt = parse_options()
    sample(opt)