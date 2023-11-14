import torch
import argparse
import ntpath
import os

from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, BatchSampler

from importlib import import_module
from models.cddm import *
from models.segmentation.UNet import UNet

import datasets
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
    Dataset = getattr(datasets, opt.dataset_name)
    dataset = Dataset(root="./data", phase="valid")

    #indices = torch.randint(0, len(dataset), size=(n_samples,)).tolist()
    indices = [228, 138, 354, 264]
    #indices = [228, 138]
    n_samples = len(indices)
    batch_sampler = DeterministicBatchSampler(indices)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    batch = next(iter(dataloader))
    batch_confocal = batch["confocal"].to(device)
    batch_STED = batch["STED"].to(device)
    batch_segGT = torch.cat([batch['seg_GTrings'], batch['seg_GTfibers']], dim=1).to(device)

    # Load model
    denoising_net = getattr(import_module(f"models.denoising.{opt.denoising_net_name}"), "Unet")(in_channels=2, out_channels=1)
    loaded_checkpoint_dir = os.path.join("./checkpoints", opt.load_checkpoint)
    legacy = False
    if legacy:  
        state_dict = torch.load(os.path.join(loaded_checkpoint_dir, "net_"+str(opt.load_epoch)+".pth"))
        state_dict = {".".join(key.split(".")[1:]): value for key, value in state_dict.items() if key.startswith('eps_model')}
    else:
        state_dict = torch.load(os.path.join(loaded_checkpoint_dir, f"denoising_net_{opt.load_epoch}.pth"))
    denoising_net.load_state_dict(state_dict)

    model = cDDPM(denoising_net, denoising_target=opt.denoising_target)
    model.to(device)
    
    task_net = UNet(in_channels=1, out_channels=2).to(device)
    state_dict = torch.load("./checkpoints/STEDActinFCN_Dendrites/params.net", map_location=device)
    task_net.load_state_dict(state_dict)

    num_params = get_number_params(task_net)
    print('[%s Network] Total number of parameters : %.3f M' % ('Segmentation', num_params / 1e6))
    num_params = get_number_params(denoising_net)
    print('[%s Network] Total number of parameters : %.3f M' % ('Denoising', num_params / 1e6))

    
    model.eval()
    with torch.no_grad():
        batch_DM = model.sample(batch_confocal)
    
    batch_confocal = dataset.denormalize(batch_confocal, 'confocal')
    batch_STED = dataset.denormalize(batch_STED, 'STED')
    batch_DM = dataset.denormalize(batch_DM, 'STED')

    task_net.eval() # very different results because of batch norm 
    with torch.no_grad():
        # Need to scale inputs for the segmentation network
        factor = 0.8*2.25 # Anthony's numbers (0.8*max(dataset))
        factor *= 1/0.3166458 # Match means linear fit in training dataset
        batch_segDM = task_net(batch_DM/factor)
        batch_segSTED = task_net(batch_STED/factor)

    batch_seg = torch.cat((batch_segGT, batch_segSTED, batch_segDM))
    batch_seg = combine_red_purple(batch_seg[:,1:2], batch_seg[:,0:1])
    matrix_seg = torch.reshape(batch_seg, (n_samples,3,3,224,224))

    batch_images = torch.cat((batch_confocal, batch_STED, batch_DM)) 
    batch_images = apply_colormap(batch_images, vmin=0, vmax=1)
    matrix_images = torch.reshape(batch_images, (n_samples,3,3,224,224))
    
    matrix_output = torch.zeros((n_samples,6,3,224,224))
    matrix_output[:,::2] = matrix_images
    matrix_output[:,1::2] = matrix_seg
    #matrix_output = torch.swapaxes(matrix_output, 0, 1)
    batch_output = torch.reshape(matrix_output, (-1,3,224,224))

    grid = make_grid(batch_output, nrow=n_samples*2, pad_value=1)

    save_dir = f"samples/{opt.load_checkpoint}_{opt.load_epoch}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_image(grid, f"{save_dir}/sample.png")
    
    for i in range(n_samples):
        save_tensor_as_tif(batch_confocal[i], f"{save_dir}/valid_{indices[i]}_confocal.tif")
        save_tensor_as_tif(batch_segGT[i], f"{save_dir}/valid_{indices[i]}_segGT.tif")
        save_tensor_as_tif(batch_STED[i], f"{save_dir}/valid_{indices[i]}_STED.tif")
        save_tensor_as_tif(batch_segSTED[i], f"{save_dir}/valid_{indices[i]}_segSTED.tif")
        save_tensor_as_tif(batch_DM[i], f"{save_dir}/valid_{indices[i]}_DM.tif")
        save_tensor_as_tif(batch_segDM[i], f"{save_dir}/valid_{indices[i]}_segDM.tif")

if __name__ == "__main__":
    opt = parse_options()
    sample(opt)