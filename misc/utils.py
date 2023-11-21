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
    image = (image*255).cpu().numpy().astype('uint8')
    tif.imsave(filepath, image)

def get_number_params(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params

def get_linspace_timesteps(n, T, start_at_one=False):
    zero_one = torch.linspace(0,1,n)
    if start_at_one:
        t = 1+torch.ceil((T-1)*zero_one).type(torch.int64)
    else:
        t = torch.ceil(T*zero_one).type(torch.int64)
    return t

def get_symmetric_logspace_timesteps(n, T, base=100, start_at_one=False):
    '''Returns tensor of size n (or n-1 if n is even) containing points in [0,T] (or [1,T] if start_at_one=True).
    The spacing in [0, T/2] is logarithmic, and ]T/2, T] is the same but flipped.'''
    logspace = torch.logspace(start=0, end=1, steps=(n+1)//2, base=base)
    zero_mid = 0.5*normalize_zero_one(logspace)
    mid_one = (1-zero_mid[:-1]).flip(0)
    zero_one = torch.cat([zero_mid, mid_one])
    if start_at_one:
        t = 1+torch.ceil((T-1)*zero_one).type(torch.int64)
    else:
        t = torch.ceil(T*zero_one).type(torch.int64)
    return t

def normalize_zero_one(x):
    x = (x-x.min())/(x.max()-x.min())
    return x