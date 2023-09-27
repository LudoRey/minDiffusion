import os.path
from torch.utils.data import Dataset
from data.image_folder import make_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import tifffile

class AxonsDataset(Dataset):
    """A dataset class for paired image dataset with segmentation mask fo one structure.

    """

    def __init__(self, root="./data", phase="train", transform=None):
        """Initialize this dataset class.

        """
        self.dir_images = os.path.join(root, "AxonalRingsDataset", phase)  # get the image directory
        self.image_paths = sorted(make_dataset(self.dir_images))  # get image paths
        self.transform = transform # tf to be applied 

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
        # split image into confocal and STED
        confocal = torch.Tensor(image[0]).unsqueeze(0)
        STED = torch.Tensor(image[1]).unsqueeze(0)
        if self.transform is None:
            tf = transforms.Compose(
                [transforms.Normalize(0.5, 0.5), transforms.RandomCrop(32)]
            )
            confocal = tf(confocal)
            STED = tf(STED)
        return {'confocal': confocal, 'STED': STED}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
