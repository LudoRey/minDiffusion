import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from torch.nn.functional import normalize
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM

from data.axons_dataset import AxonsDataset

def sample(dataset_name, load_checkpoint, n_samples=2):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Dataset and DataLoader
    dataset = globals()[dataset_name](root="./data", crop=True)
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=True)
    img_shape = dataset[0]["STED"].shape

    ddpm = DDPM(eps_model=NaiveUnet(img_shape[0], img_shape[0], n_feat=128), betas=(1e-4, 0.02), n_T=1000)
    ddpm.load_state_dict(torch.load(load_checkpoint))
    ddpm.to(device)

    ddpm.eval()
    with torch.no_grad():
        samples = ddpm.sample(n_samples, img_shape, device).cpu()
        images = next(iter(dataloader))['STED']
        samples_images = torch.cat([samples, images], dim=0)
        grid = make_grid(samples_images, normalize=True, value_range=(-0.4, 1), nrow=n_samples)

    save_image(grid, "samples/sample1.png")
    
    #samples = normalize_image(samples, -0.4, 1)
    #img = samples.squeeze()
    #img = img.cpu().numpy()
    #img = (img * 255).astype(np.uint8)

    #img = Image.fromarray(img)
    #img.save("./samples/sample.png")

def normalize_image(img, vmin=0, vmax=1):
    '''Clips values outside [vmin, vmax], then normalizes from [vmin, vmax] to [0,1]'''
    img = torch.clamp(img, vmin, vmax)
    img = (img-vmin) / (vmax-vmin)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--load_checkpoint", type=str, help="The path to a .pth file which is used to initialize the model")
    args = parser.parse_args()

    sample(dataset_name=args.dataset_name, load_checkpoint=args.load_checkpoint)

