import tifffile as tif
import torch
from matplotlib import cm   

def apply_colormap(x, vmin=0, vmax=1, cmap=cm.get_cmap('hot')):
    '''Input: a Bx1xHxW tensor. Output: a Bx3xHxW tensor.'''
    x = torch.clamp(x, vmin, vmax)
    x = (x-vmin)/(vmax-vmin)
    y = cmap(x.squeeze(1).cpu().numpy())
    y = torch.Tensor(y).permute(0,3,1,2)[:,:3] # BxHxWx4 -> Bx3xHxW
    return y

def combine_purple_green(channel1, channel2):
    '''Input: two Bx1xHxW tensor. Output: a Bx3xHxW tensor.'''
    image = torch.cat([channel1, channel2, channel1], dim=1)
    return image

def save_tensor_as_tif(image, filepath):
    image = torch.clamp(image, min=0, max=1)
    image = (image*255).cpu().numpy().astype('uint16')
    tif.imsave(filepath, image)

def get_number_params(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params