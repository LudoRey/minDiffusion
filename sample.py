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

def sample(opt, n_samples=4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test Dataset
    dataset = globals()[opt.dataset_name](root="./data", crop=opt.crop, phase="valid")
    #test = dataset[6]
    test = dataset[0]
    test_confocal = test["confocal"].to(device).repeat(n_samples,1,1,1)
    test_STED = test["STED"].to(device).repeat(n_samples,1,1,1)

    img_shape = dataset[0]["STED"].shape
    # Load model
    model = cDDIM(eps_model=Unet(img_shape[0]*2, img_shape[0], n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
    state_dict = torch.load(os.path.join(checkpoint_dir, "net_"+str(opt.load_epoch)+".pth"))
    state_dict = {key: value for key, value in state_dict.items() if key.startswith('eps_model')} # register_buffer used to be persistent 
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    tau = get_tau(dim=100)

    model.eval()
    with torch.no_grad():
        samples = model.sample(test_confocal, tau)
    
    # Make and save grid
    xset = torch.cat((test_confocal, samples, test_STED), dim=0)
    yset = apply_colormap(xset, vmin=-0.8, vmax=7)
    grid = make_grid(yset, nrow=n_samples)
    save_image(grid, "samples/sample_"+opt.load_checkpoint+"_"+str(opt.load_epoch)+".png")
    
if __name__ == "__main__":
    opt = parse_options()
    sample(opt)