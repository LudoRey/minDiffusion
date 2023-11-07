import torch
import argparse
import ntpath
import os

from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, BatchSampler

from models.unet import Unet
from models.cddm import *
from models.segmentation import networks

from datasets.axons_dataset import AxonsDataset
from datasets.dendrites_dataset import DendritesDataset
from misc.options import parse_options
from misc.utils import *

class DeterministicBatchSampler(BatchSampler):
    def __init__(self, indices):
        self.indices = indices
        super().__init__(self, len(indices), drop_last=False)
    def __iter__(self):
        yield self.indices

def sample(opt, n_samples=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Dataset
    dataset = globals()[opt.dataset_name](root="./data", phase="valid")

    #indices = torch.randint(0, len(dataset), size=(n_samples,)).tolist()
    #indices = [228, 138, 354, 264]
    indices = [228, 138]
    n_samples = len(indices)
    batch_sampler = DeterministicBatchSampler(indices)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    batch = next(iter(dataloader))
    batch_confocal = batch["confocal"].to(device)
    batch_STED = batch["STED"].to(device)

    img_shape = dataset[0]["STED"].shape
    # Load model
    model = cDDPM(denoising_net=Unet(img_shape[0]*2, img_shape[0], n_feat=128))

    loaded_checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
    legacy = True
    if legacy:  
        state_dict = torch.load(os.path.join(loaded_checkpoint_dir, "net_"+str(opt.load_epoch)+".pth"))
        state_dict = {".".join(key.split(".")[1:]): value for key, value in state_dict.items() if key.startswith('eps_model')}
    else:
        state_dict = torch.load(os.path.join(loaded_checkpoint_dir, f"denoising_net_{opt.load_epoch}.pth"))
    model.denoising_net.load_state_dict(state_dict)
    model.to(device)
    
    netS = networks.define_S(input_nc=1, output_nc=2, use_dropout=False, gpu_ids=[0])
    state_dict = torch.load("./checkpoints/TAGAN_Dendrites/500_net_S.pth", map_location=device)
    netS.module.load_state_dict(state_dict)

    num_params = get_number_params(netS)
    #print('[%s Network] Total number of parameters : %.3f M' % ('Segmentation', num_params / 1e6))
    num_params = get_number_params(model.denoising_net)
    #print('[%s Network] Total number of parameters : %.3f M' % ('Denoising', num_params / 1e6))

    
    model.eval()
    with torch.no_grad():
        batch_generated = model.sample(batch_confocal)
    
    # Make and save grid
    batch_confocal = dataset.denormalize(batch_confocal, 'confocal')
    batch_STED = dataset.denormalize(batch_STED, 'STED')
    batch_generated = dataset.denormalize(batch_generated, 'STED')

    batch_fakeSTED = transforms.Normalize(0.5,0.5)(batch_generated)
    with torch.no_grad():
        seg_shape = list(batch_STED.shape)
        seg_shape[1]*= 2
        batch_seg_fakeSTED = torch.zeros(seg_shape, device=device)
        # Need to loop because netS is not in eval mode : batch_size needs to be 1 for batchnorm
        for i in range(batch_fakeSTED.shape[0]):
            batch_seg_fakeSTED[i:i+1] = netS(batch_fakeSTED[i:i+1])
    batch_seg_fakeSTED = (batch_seg_fakeSTED + 1) / 2
    print(seg_shape)
    print(batch_seg_fakeSTED.shape)
    
    xset = torch.cat((batch_confocal, batch_STED, batch_generated), dim=0)
    xset = apply_colormap(xset, vmin=0, vmax=1)
    grid = make_grid(xset, nrow=n_samples, pad_value=1)

    save_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_image(grid, f"{save_dir}/sample.png")
    
    for i in range(n_samples):
        confocal = batch_confocal[i]
        STED = batch_STED[i]
        generated = batch_generated[i]
        seg_fakeSTED = batch_seg_fakeSTED[i]

        image = torch.cat((confocal, STED, generated, seg_fakeSTED), dim=0)
        save_tensor_as_tif(image, f"{save_dir}/sample_{i}.tif")

if __name__ == "__main__":
    opt = parse_options()
    sample(opt)