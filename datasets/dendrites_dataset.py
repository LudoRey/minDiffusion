import os.path
from torch.utils.data import Dataset
from datasets.image_folder import make_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import tifffile

class DendritesDataset(Dataset):
    """A dataset class for paired image dataset with segmentation mask fo one structure.

    """

    def __init__(self, root="./data", phase="train", match_means=True, normalize=True, normalize_mode="custom", crop_size=None):
        """
        Parameters:
            root (string) - - dataset directory should be <root>/DendriticFActinDataset
            phase (string) - - either train, test or valid
            match_means (bool) - - if True, the mean of the STED image is matched to the mean of the confocal image
            normalize (bool) - - if True, dataset is normalized according to normalize mode (see below)
            normalize_mode (string) - - either custom or default. Custom uses mean/std values computed over the dataset, default uses 0.5/0.5
            crop_size (int) - - default is None : no cropping
        """
        self.dir_images = os.path.join(root, "DendriticFActinDataset", phase)  # get the image directory
        self.image_paths = sorted(make_dataset(self.dir_images))  # get image paths
        self.match_means = match_means
        self.normalize = normalize
        self.crop_size = crop_size

        if normalize_mode == "custom":
            self.coeffs_confocal = (0.0850, 0.0934) # (mean, std) for normalization
            if self.match_means:
                self.coeffs_sted = (0.0850, 0.1107) # ! computed after matching means
            else:
                self.coeffs_sted = (0.0287, 0.0375)
        elif normalize_mode == "default":
            self.coeffs_confocal = (0.5, 0.5)
            self.coeffs_sted     = (0.5, 0.5)
        

    def __getitem__(self, index):
        """
        Returns a dictionary that contains
            confocal (tensor) - - a confocal image 
            STED (tensor) - - its corresponding STED image
            seg_GTrings (tensor) - - rings ground truth seg. mask
            seg_GTfibers (tensor) - - fibers ground truth seg. mask
            image_paths (str) - - image path
        """
        # read a image given a random integer index
        image_path = self.image_paths[index]
        image = tifffile.imread(image_path).astype(float) / 255.0
        image = torch.Tensor(image)
        image = image.unsqueeze(1) #4xHxW -> 4x1xHxW

        ### Preprocessing
        if self.match_means: # Match the mean of the STED image with the confocal image   
            ratio = torch.mean(image[0])/torch.mean(image[1])
            image[1] = ratio*image[1]
        if self.normalize: # Normalize
            image[0] = transforms.Normalize(*self.coeffs_confocal)(image[0])
            image[1] = transforms.Normalize(*self.coeffs_sted)(image[1])
        if self.crop_size is not None:
            #shape = [224 * (i // 224) for i in image.shape[1:3]] # for test images of various sizes
            tf = transforms.CenterCrop(self.crop_size)
            image = tf(image)

        # split image into confocal, STED, etc.
        confocal = image[0]
        STED = image[1]
        seg_GTrings = image[2]
        seg_GTfibers = image[3]
        #return {'confocal': confocal, 'STED': STED}
        return {'confocal': confocal, 'STED': STED, 'seg_GTrings': seg_GTrings, 'seg_GTfibers': seg_GTfibers, 'image_path': image_path}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
    
    def denormalize(self, images, type):
        if type=='confocal':
            images = images*self.coeffs_confocal[1] + self.coeffs_confocal[0]
        if type=='STED': # STED images might be out out [0,1] range because of matching means
            images = images*self.coeffs_sted[1] + self.coeffs_sted[0]
        return images