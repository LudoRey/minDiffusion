import torch
import argparse
import os

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from models.unet import NaiveUnet
from models.ddpm_cond import cDDPM

from datasets.axons_dataset import AxonsDataset

def sample(opt, n_samples=4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Dataset and DataLoader
    dataset = globals()[opt.dataset_name](root="./data", crop=opt.crop)
    #dataset = globals()[opt.dataset_name](root="./data", crop=opt.crop, phase="test")
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
    img_shape = dataset[0]["STED"].shape
    # Create test Dataset
    test = next(iter(dataloader))
    test_confocal = test["confocal"].to(device)
    test_STED = test["STED"].to(device)
    print(test_confocal.shape)

    # Load model
    ddpm = cDDPM(eps_model=NaiveUnet(img_shape[0]*2, img_shape[0], n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
    ddpm.load_state_dict(torch.load(os.path.join(checkpoint_dir, "net_"+str(opt.load_epoch)+".pth")))
    ddpm.to(device)

    ddpm.eval()
    with torch.no_grad():
        # Sample test
        samples = ddpm.sample(test_confocal, device)
    
    min=torch.min(samples)
    max=torch.max(samples)
    print(min,max)
    # Save grid
    xset = torch.cat((test_confocal, samples, test_STED), dim=0)
    grid = make_grid(xset, normalize=True, value_range=(-0.4, 5), nrow=4)
    save_image(grid, "samples/sample_"+str(opt.load_epoch)+".png")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="AxonsDataset", help="Name of Dataset class. Must be imported.")
    parser.add_argument("--crop", type=bool, default=False, help="Passed to Dataset argument.")
    parser.add_argument("--load_checkpoint", type=str, help="The checkpoint folder to be loaded. Will continue to save data/log info in the same folder.")
    parser.add_argument("--load_epoch", type=int, help="The model ./checkpoint/<load_checkpoint>/net_<load_epoch>.pth will be loaded.")
    opt = parser.parse_args()

    sample(opt)

