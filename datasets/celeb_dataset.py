import os.path
from torch.utils.data import Dataset
from datasets.image_folder import make_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import tifffile

class CelebDataset(Dataset):
    """A dataset class for paired image dataset with segmentation mask fo one structure.

    """

    def __init__(self, root="./data", phase="train", normalize=True, crop=False):
        """Initialize this dataset class.

        """
        self.dir_images = os.path.join(root, "celeb")  # get the image directory
        self.image_paths = sorted(make_dataset(self.dir_images))  # get image paths
        #self.coeffs = (0.582, 0.196)
        self.normalize = normalize
        self.coeffs_confocal = (0.5, 0.5) # (mean, std) for normalization
        self.coeffs_sted = (0.5, 0.5) # ! computed after matching means
        self.crop = crop


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            confocal(tensor) - - a confocal image 
            STED (tensor) - - its corresponding STED image
            image_paths (str) - - image paths
        """
        # read a image given a random integer index
        image_path = self.image_paths[index]
        image = tifffile.imread(image_path).astype(float) / 255.0
        image = torch.Tensor(image)

        # preprocessing
        if self.normalize:
            # Normalize to mean=0, std=1
            image[0:1] = transforms.Normalize(*self.coeffs_confocal)(image[0:1])
            image[1:2] = transforms.Normalize(*self.coeffs_sted)(image[1:2])
        if self.crop:
            tf = transforms.RandomCrop(64)
            image = tf(image)

        # split image into confocal and STED
        confocal = image[0].unsqueeze(0)
        STED = image[1].unsqueeze(0)

        return {'confocal': confocal, 'STED': STED}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
    
    def denormalize(self, images, type):
        if type=='confocal':
            images = images*self.coeffs_confocal[1] + self.coeffs_confocal[0]
        if type=='STED': # STED images might be out out [0,1] range because of matching means
            images = images*self.coeffs_sted[1] + self.coeffs_sted[0]
        return images