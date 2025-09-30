from torch.utils.data import Dataset
import rasterio as rio
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

from utils import standardize
from utils import RandomCrop
from utils import RandomFlipPair
from utils import RandomRotationPair


class BeforeData(Dataset):
    def __init__(self,
                before_dir,
                after_dir,
                label_dir, 
                split: str = 'train',
                num_augmentations: int | None = 1
                ):
        
        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'.")

        self.before_files = sorted(Path(before_dir).glob("*.tif"))
        self.after_files = sorted(Path(after_dir).glob("*.tif"))
        self.label_files = sorted(Path(label_dir).glob("*.tif"))
        self.split = split
        self.num_augmentations = num_augmentations

        assert len(self.before_files) == len(self.after_files), "Mismatch in number of before and after images"
        assert len(self.before_files) == len(self.label_files), "Mismatch in number of before images and labels"
        
        
        if self.split == "train":
            self.transform = transforms.Compose([
                RandomCrop(224),
                RandomFlipPair(),
                RandomRotationPair()
            ])
        else:
            self.transform = RandomCrop(224) # NEED TO EDIT THIS PROBABLY FOR TEST & figure out way to account for all data


    def __len__(self):
        return len(self.before_files) * self.num_augmentations

    def __getitem__(self, index):

        if index != 0:
            index = index // self.num_augmentations ######keep working from here...

        
        with rio.open(self.before_files[index]) as src_x_before,\
            rio.open(self.after_files[index]) as src_x_after,\
            rio.open(self.label_files[index]) as src_y :
            x_before = torch.from_numpy(src_x_before.read()).float()
            x_after = torch.from_numpy(src_x_after.read()).float()
            y = torch.from_numpy(src_y.read()).squeeze()   #switching size from [1,457, 447] to [457,447]

            sample = {'x_before': x_before, 'x_after': x_after, 'y': y}

            if self.transform:
                sample = self.transform(sample)

            sample['x_before'] = standardize(sample['x_before'], dim =1)
            sample['x_after'] = standardize(sample['x_after'], dim =1)
 
            
            x_before = sample['x_before'].float()
            x_after = sample['x_after'].float()
            y = sample['y'].long()
            
            return (x_before, x_after), y