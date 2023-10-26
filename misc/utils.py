import os
import csv
import time
import torch
#import cv2
from matplotlib import cm   

def apply_colormap(x, vmin=0, vmax=1, cmap=cm.get_cmap('hot')):
    '''Input: a Bx1xHxW tensor. Output: a Bx3xHxW tensor.'''
    x = torch.clamp(x, vmin, vmax)
    x = (x-vmin)/(vmax-vmin)
    y = cmap(x.squeeze(1).cpu().numpy())
    y = torch.Tensor(y).permute(0,3,1,2) # BxHxWxC -> BxCxHxW
    return y