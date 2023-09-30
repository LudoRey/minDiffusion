import os.path
from torch.utils.data import Dataset
from datasets.image_folder import make_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import tifffile

class AxonsDataset(Dataset):
    """A dataset class for paired image dataset with segmentation mask fo one structure.

    """

    def __init__(self, root="./data", phase="train", normalize=True, crop=True):
        """Initialize this dataset class.

        """
        self.dir_images = os.path.join(root, "AxonalRingsDataset", phase)  # get the image directory
        self.image_paths = sorted(make_dataset(self.dir_images))  # get image paths
        self.normalize = normalize
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
        image = tifffile.imread(image_path)[0:2].astype(float) / 255.0
        image = torch.Tensor(image)

        # preprocessing
        if self.normalize:
            tf = transforms.Normalize(0.04, 0.1)
            image = tf(image)
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
