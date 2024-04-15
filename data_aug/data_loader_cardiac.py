
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from torch.utils.data import DataLoader, Dataset
import numpy as np
class cardiac_data (Dataset):
    """Cardiac Data Dataset"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            base_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # self.base_dir = base_dir
        self.root_dir=root_dir
        self.list_images = glob.glob(os.path.join(root_dir, "*.*")) 
        self.transform = transform

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        img_name = self.list_images[idx]
        image = Image.open(img_name).convert("RGB")
        
        # (image*255)/image.max 
        


        if self.transform is not None:
            img = self.transform(image)
    
        return img,-1


