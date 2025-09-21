from torch.utils.data import Dataset
import rasterio as rio
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.transforms import functional as F


class BeforeData(Dataset):
    def __init__(self,
                before_dir,
                after_dir,
                label_dir, 
                # input_size = (224, 224), # what size tifs would you like TerraMind to process? #Channels and batch size are handled separately
                transform = None # drop input_size and num_grabs 
                ):

        self.before_files = sorted(Path(before_dir).glob("*.tif"))
        self.after_files = sorted(Path(after_dir).glob("*.tif"))
        self.label_files = sorted(Path(label_dir).glob("*.tif"))
        
        self.transform = transform

        assert len(self.before_files) == len(self.after_files), "Mismatch in number of before and after images"
        assert len(self.before_files) == len(self.label_files), "Mismatch in number of before images and labels"
        


    def __len__(self):
        return len(self.before_files)

    def __getitem__(self, index):
        with rio.open(self.before_files[index]) as src_x_before,\
            rio.open(self.after_files[index]) as src_x_after,\
            rio.open(self.label_files[index]) as src_y :
            x_before = torch.from_numpy(src_x_before.read())
            x_after = torch.from_numpy(src_x_after.read())
            y = torch.from_numpy(src_y.read())

            sample = {"x_before": x_before,
                      "x_after": x_after,
                      "y": y}

            if self.transform:
                sample = self.transform(sample)
            
            x_before = sample['x_before'].float()
            x_after = sample['x_after'].float()
            y = sample['y'].int()
            
            return (x_before, x_after), y