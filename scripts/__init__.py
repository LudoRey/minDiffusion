import torch

from importlib import import_module
from models.diffusion import *
from models.segmentation.unet import Unet

def load_model(opt):
    UnetInfusedTimestep = getattr(import_module(f"models.denoising.{opt.denoising_net_name}"), "Unet") # Model is dynamically imported depending on denoising_net_name
    denoising_net = UnetInfusedTimestep(in_channels=2, out_channels=2 if opt.denoising_target=="both" else 1)
    denoising_net.load_state_dict(torch.load(f"./checkpoints/{opt.load_checkpoint}/denoising_net_{opt.load_epoch}.pth"))

    task_net = Unet(in_channels=1, out_channels=2)
    task_net.load_state_dict(torch.load("./checkpoints/STEDActinFCN_Dendrites/params.net"))

    model = TADM(denoising_net, task_net, opt.task_weight, opt.denoising_target, opt.loss_weighting)
    return model
