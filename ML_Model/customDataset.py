import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
from PIL import Image
import os
import random

class TrainDataset(Dataset):
    def __init__(self, sample_dir, mask_dir, transform=None):

        self.sample_dir = sample_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.samples = os.listdir(sample_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = os.path.join(self.sample_dir, self.samples[index])
        mask_path = os.path.join(self.mask_dir, self.samples[index]) # Indexing list of samples because they have identical names

        sample = np.array(Image.open(sample_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        # After this point, masks are 0.0 or 255.0
        mask[mask == 255.0] = 1 # Now mask is binary

        if self.transform:
            # Ensure that the flips and rotations are consistent between the sample and the mask
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            sample = self.transform(sample)

            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        return sample, mask
        

class TestDataset(Dataset):
    def __init__(self, sample_dir, transform=None):
        self.sample_dir = sample_dir
        self.transform = transform
        self.samples = os.listdir(sample_dir)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample_path = os.path.join(self.sample_dir, self.samples[index])
        sample = np.array(Image.open(sample_path).convert('L'))

        if self.transform:
            sample = self.transform(sample)

        return sample