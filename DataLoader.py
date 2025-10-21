#patches + location tracking multimodal
from torch.utils.data import Dataset
import rasterio as rio
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
import random, math
from functools import lru_cache

from utils import standardize, RandomFlipPair, RandomRotationPair


class MultiModalBeforeAfterDataset(Dataset):
    def __init__(
        self,
        modalities: dict,
        label_dir: str,
        split: str = 'train',
        num_augmentations: int = 0,
        patch_size: int = 224,
        stride: int = 224,
        add_random_offset: bool = True,
        preload: bool = True
    ):
        """
        Multi-modal before/after dataset supporting multiple data sources (S2, S1, etc.)

        Args:
            modalities (dict): {"S2": ("path/to/S2_before", "path/to/S2_after"), 
                                "S1": ("path/to/S1_before", "path/to/S1_after")}
            label_dir (str): Path to label directory.
            split (str): "train", "test", or "validation"
            num_augmentations (int): Repeated samples per patch with new augmentations.
            patch_size (int): Size of extracted patch.
            stride (int): Overlap stride between patches.
            add_random_offset (bool): Add random offset to grid.
            preload (bool): If True, load all rasters into memory once.
        """
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'validation', or 'test'.") #should I drop this?

        self.modalities = modalities
        self.label_files = sorted(Path(label_dir).glob("*.tif"))
        self.split = split
        self.num_augmentations = num_augmentations
        self.patch_size = patch_size
        self.stride = stride
        self.add_random_offset = add_random_offset
        self.preload = preload
        self.size_helper = 1 if self.num_augmentations == 0 else self.num_augmentations

        # Verify all modalities have same number of images
        label_len = len(self.label_files)
        for name, (before_dir, after_dir) in modalities.items():
            before_files = sorted(Path(before_dir).glob("*.tif"))
            after_files = sorted(Path(after_dir).glob("*.tif"))
            if len(before_files) != len(after_files):
                raise ValueError(f"Modality {name} before/after count mismatch")
            if len(before_files) != label_len:
                raise ValueError(f"Modality {name} count differs from labels")
        self.num_images = label_len

        # Augmentation policy
        self.augment = None
        if (self.split == "train") & (self.num_augmentations>0):
            self.augment = transforms.Compose([
                RandomFlipPair(),
                RandomRotationPair(),
            ])
        

        # Optionally preload images into RAM (saves time if dataset fits memory)
        self.cache = {}
        if preload:
            self._preload_images()

        # Precompute patch coordinates for all images
        self.index_map = self._build_patch_index_map()

    # ---------------------------------------------------
    # 1. Utility functions
    # ---------------------------------------------------
    def _pad_image(self, img):
        if not img.is_floating_point():
            img = img.float()

        _, H, W = img.shape
        pad_h = (math.ceil((H - self.patch_size) / self.stride) * self.stride + self.patch_size) - H
        pad_w = (math.ceil((W - self.patch_size) / self.stride) * self.stride + self.patch_size) - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Use positional "reflect" for compatibility
        padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), padding_mode =  "reflect")

        return padded, pad_top, pad_left

    def _get_random_offset(self):
        if not self.add_random_offset:
            return 0, 0
        return random.randint(0, self.stride - 1), random.randint(0, self.stride - 1)

    def _extract_patch_coords(self, img):
        _, H, W = img.shape
        dx, dy = self._get_random_offset()
        coords = []
        for y in range(dy, H - self.patch_size + 1, self.stride):
            for x in range(dx, W - self.patch_size + 1, self.stride):
                coords.append((y, x))
        return coords

    # ---------------------------------------------------
    # 2. Image loading & caching
    # ---------------------------------------------------
    @lru_cache(maxsize=None) #should we change maxsize to num_aug? But then this would likely mean that we'd redownload each epoch
    def _load_tif(self, path):
        with rio.open(path) as src:
            return torch.from_numpy(src.read()).float()

    def _preload_images(self):
        for idx in range(self.num_images):
            image_dict = {}
            for name, (before_dir, after_dir) in self.modalities.items():
                before_file = sorted(Path(before_dir).glob("*.tif"))[idx]
                after_file = sorted(Path(after_dir).glob("*.tif"))[idx]
                image_dict[name] = {
                    "before": self._load_tif(before_file),
                    "after": self._load_tif(after_file)
                }
            label_file = self.label_files[idx]
            image_dict["label"] = self._load_tif(label_file).squeeze()
            self.cache[idx] = image_dict

    # ---------------------------------------------------
    # 3. Patch indexing
    # ---------------------------------------------------
    def _build_patch_index_map(self):
        index_map = []
        # Use the first modality as shape reference
        first_mod = next(iter(self.modalities.keys()))
        first_before_dir, _ = self.modalities[first_mod]
        for i in range(self.num_images):
            ref_path = sorted(Path(first_before_dir).glob("*.tif"))[i]
            with rio.open(ref_path) as src:
                # dummy = torch.zeros((1, src.height, src.width))
                dummy = torch.zeros((1, src.height, src.width), dtype=torch.float32)
                dummy, _, _ = self._pad_image(dummy)
                coords = self._extract_patch_coords(dummy)
                for c in coords:
                    index_map.append((i, c))
        return index_map

    # ---------------------------------------------------
    # 4. Dataset interface
    # ---------------------------------------------------
    def __len__(self):
        return len(self.index_map) * self.size_helper #counting num patches per tif through index_map. Then multiplied by number of augmentations per patch

    def __getitem__(self, index):
        img_index = index // self.size_helper
        i, (y, x) = self.index_map[img_index]

        # Load data (either cached or on demand)
        if self.preload:
            data = self.cache[i]
        else:
            data = {}
            for name, (before_dir, after_dir) in self.modalities.items():
                before_file = sorted(Path(before_dir).glob("*.tif"))[i]
                after_file = sorted(Path(after_dir).glob("*.tif"))[i]
                data[name] = {
                    "before": self._load_tif(before_file),
                    "after": self._load_tif(after_file)
                }
            label_file = self.label_files[i]
            data["label"] = self._load_tif(label_file).squeeze()

        before_dict = {}
        after_dict = {}

        for name in self.modalities:
            x_before, pad_top, pad_left = self._pad_image(data[name]["before"])
            x_after, _, _ = self._pad_image(data[name]["after"])
            patch_before = x_before[:, y:y+self.patch_size, x:x+self.patch_size]
            patch_after  = x_after[:, y:y+self.patch_size, x:x+self.patch_size]

            # Standardize per modality
            patch_before = standardize(patch_before, dim=1)
            patch_after  = standardize(patch_after, dim=1)

            before_dict[name] = patch_before.float()
            after_dict[name]  = patch_after.float()

        # Label patch
        y_full, _, _ = self._pad_image(data["label"].unsqueeze(0))
        y_patch = y_full[0, y:y+self.patch_size, x:x+self.patch_size]

        sample = {
            "before": before_dict,
            "after": after_dict,
            "y": y_patch
        }

        # Augmentation (training only): augment expects sample['y'] to exist
        if self.augment and self.split == "train":
            sample = self.augment(sample)

        # After augmentation, separate inputs and label for return
        y = sample.pop("y")  # extract label tensor (now possibly transformed)
        input_sample = {"before": sample["before"], "after": sample["after"]}

        # Final types/standardization already applied per modality; ensure label type
        return input_sample, y.long()


########################## Most recent GPT improvement above, only changing how output is gathered. just below is old working copy

# #patches + location tracking multimodal
# from torch.utils.data import Dataset
# import rasterio as rio
# from pathlib import Path
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# import random, math
# from functools import lru_cache

# from utils import standardize, RandomFlipPair, RandomRotationPair


# class MultiModalBeforeAfterDataset(Dataset):
#     def __init__(
#         self,
#         modalities: dict,
#         label_dir: str,
#         split: str = 'train',
#         num_augmentations: int = 1,
#         patch_size: int = 224,
#         stride: int = 224,
#         add_random_offset: bool = True,
#         preload: bool = True
#     ):
#         """
#         Multi-modal before/after dataset supporting multiple data sources (S2, S1, etc.)

#         Args:
#             modalities (dict): {"S2": ("path/to/S2_before", "path/to/S2_after"), 
#                                 "S1": ("path/to/S1_before", "path/to/S1_after")}
#             label_dir (str): Path to label directory.
#             split (str): "train", "test", or "validation"
#             num_augmentations (int): Repeated samples per patch with new augmentations.
#             patch_size (int): Size of extracted patch.
#             stride (int): Overlap stride between patches.
#             add_random_offset (bool): Add random offset to grid.
#             preload (bool): If True, load all rasters into memory once.
#         """
#         if split not in ["train", "validation", "test"]:
#             raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'.") #should I drop this?

#         self.modalities = modalities
#         self.label_files = sorted(Path(label_dir).glob("*.tif"))
#         self.split = split
#         self.num_augmentations = num_augmentations
#         self.patch_size = patch_size
#         self.stride = stride
#         self.add_random_offset = add_random_offset
#         self.preload = preload

#         # Verify all modalities have same number of images
#         label_len = len(self.label_files)
#         for name, (before_dir, after_dir) in modalities.items():
#             before_files = sorted(Path(before_dir).glob("*.tif"))
#             after_files = sorted(Path(after_dir).glob("*.tif"))
#             if len(before_files) != len(after_files):
#                 raise ValueError(f"Modality {name} before/after count mismatch")
#             if len(before_files) != label_len:
#                 raise ValueError(f"Modality {name} count differs from labels")
#         self.num_images = label_len

#         # Augmentation policy
#         self.augment = None
#         if split == "train":
#             self.augment = transforms.Compose([
#                 RandomFlipPair(),
#                 RandomRotationPair(),
#             ])

#         # Optionally preload images into RAM (saves time if dataset fits memory)
#         self.cache = {}
#         if preload:
#             self._preload_images()

#         # Precompute patch coordinates for all images
#         self.index_map = self._build_patch_index_map()

#     # ---------------------------------------------------
#     # 1. Utility functions
#     # ---------------------------------------------------
#     def _pad_image(self, img):
#         if not img.is_floating_point():
#             img = img.float()

#         _, H, W = img.shape
#         pad_h = (math.ceil((H - self.patch_size) / self.stride) * self.stride + self.patch_size) - H
#         pad_w = (math.ceil((W - self.patch_size) / self.stride) * self.stride + self.patch_size) - W
#         pad_top = pad_h // 2
#         pad_bottom = pad_h - pad_top
#         pad_left = pad_w // 2
#         pad_right = pad_w - pad_left

#         # Use positional "reflect" for compatibility
#         padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), padding_mode =  "reflect")

#         return padded, pad_top, pad_left

#     def _get_random_offset(self):
#         if not self.add_random_offset:
#             return 0, 0
#         return random.randint(0, self.stride - 1), random.randint(0, self.stride - 1)

#     def _extract_patch_coords(self, img):
#         _, H, W = img.shape
#         dx, dy = self._get_random_offset()
#         coords = []
#         for y in range(dy, H - self.patch_size + 1, self.stride):
#             for x in range(dx, W - self.patch_size + 1, self.stride):
#                 coords.append((y, x))
#         return coords

#     # ---------------------------------------------------
#     # 2. Image loading & caching
#     # ---------------------------------------------------
#     @lru_cache(maxsize=None) #should we change maxsize to num_aug? But then this would likely mean that we'd redownload each epoch
#     def _load_tif(self, path):
#         with rio.open(path) as src:
#             return torch.from_numpy(src.read()).float()

#     def _preload_images(self):
#         for idx in range(self.num_images):
#             image_dict = {}
#             for name, (before_dir, after_dir) in self.modalities.items():
#                 before_file = sorted(Path(before_dir).glob("*.tif"))[idx]
#                 after_file = sorted(Path(after_dir).glob("*.tif"))[idx]
#                 image_dict[name] = {
#                     "before": self._load_tif(before_file),
#                     "after": self._load_tif(after_file)
#                 }
#             label_file = self.label_files[idx]
#             image_dict["label"] = self._load_tif(label_file).squeeze()
#             self.cache[idx] = image_dict

#     # ---------------------------------------------------
#     # 3. Patch indexing
#     # ---------------------------------------------------
#     def _build_patch_index_map(self):
#         index_map = []
#         # Use the first modality as shape reference
#         first_mod = next(iter(self.modalities.keys()))
#         first_before_dir, _ = self.modalities[first_mod]
#         for i in range(self.num_images):
#             ref_path = sorted(Path(first_before_dir).glob("*.tif"))[i]
#             with rio.open(ref_path) as src:
#                 # dummy = torch.zeros((1, src.height, src.width))
#                 dummy = torch.zeros((1, src.height, src.width), dtype=torch.float32)
#                 dummy, _, _ = self._pad_image(dummy)
#                 coords = self._extract_patch_coords(dummy)
#                 for c in coords:
#                     index_map.append((i, c))
#         return index_map

#     # ---------------------------------------------------
#     # 4. Dataset interface
#     # ---------------------------------------------------
#     def __len__(self):
#         return len(self.index_map) * self.num_augmentations #counting num patches per tif through index_map. Then multiplied by number of augmentations per patch

#     def __getitem__(self, index):
#         img_index = index // self.num_augmentations
#         i, (y, x) = self.index_map[img_index]

#         # Load data (either cached or on demand)
#         if self.preload:
#             data = self.cache[i]
#         else:
#             data = {}
#             for name, (before_dir, after_dir) in self.modalities.items():
#                 before_file = sorted(Path(before_dir).glob("*.tif"))[i]
#                 after_file = sorted(Path(after_dir).glob("*.tif"))[i]
#                 data[name] = {
#                     "before": self._load_tif(before_file),
#                     "after": self._load_tif(after_file)
#                 }
#             label_file = self.label_files[i]
#             data["label"] = self._load_tif(label_file).squeeze()

#         # Pad and extract patches per modality
#         before_modalities, after_modalities = [], []
#         for name in self.modalities:
#             x_before, pad_top, pad_left = self._pad_image(data[name]["before"])
#             x_after, _, _ = self._pad_image(data[name]["after"])
#             patch_before = x_before[:, y:y+self.patch_size, x:x+self.patch_size]
#             patch_after  = x_after[:, y:y+self.patch_size, x:x+self.patch_size]
#             before_modalities.append(patch_before)
#             after_modalities.append(patch_after)

#         # Stack all modalities channel-wise
#         x_before = torch.cat(before_modalities, dim=0)
#         x_after  = torch.cat(after_modalities, dim=0)

#         # Label patch
#         y_full, _, _ = self._pad_image(data["label"].unsqueeze(0))
#         y_patch = y_full[0, y:y+self.patch_size, x:x+self.patch_size]

#         sample = {'x_before': x_before, 'x_after': x_after, 'y': y_patch}

#         # Augmentation (training only)
#         if self.augment and self.split == "train":
#             sample = self.augment(sample)

#         # Standardize
#         sample['x_before'] = standardize(sample['x_before'], dim=1)
#         sample['x_after']  = standardize(sample['x_after'], dim=1)

#         return (sample['x_before'].float(), sample['x_after'].float()), sample['y'].long()
    

    ################ START HERE: I want to change the output style from just these to a dictionary which names input type and keeps all the patches in order between the datasets


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