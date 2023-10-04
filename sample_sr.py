import torch
import argparse
import os

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from models.unet import Unet
from models.ddpm_cond import cDDPM
from models.ddim_cond import cDDIM

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
    ddpm = cDDPM(eps_model=Unet(img_shape[0]*2, img_shape[0], n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
    ddpm.load_state_dict(torch.load(os.path.join(checkpoint_dir, "net_"+str(opt.load_epoch)+".pth")))
    ddpm.to(device)

    ddpm.eval()
    with torch.no_grad():
        # Sample test
        samples = ddpm.sample(test_confocal, device)
    
    # Make and save grid
    xset = torch.cat((test_confocal, samples, test_STED), dim=0)
    yset = apply_colormap(xset, vmin=-0.8, vmax=1)
    grid = make_grid(yset, nrow=4)
    save_image(grid, "samples/sample_"+opt.load_checkpoint+"_"+str(opt.load_epoch)+".png")
    
if __name__ == "__main__":
    opt = parse_options()
    sample(opt)