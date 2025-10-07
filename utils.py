import numpy as np
import torch
from torchvision import transforms
import random


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        x_before, x_after, y = sample['x_before'], sample['x_after'], sample['y']


        h, w = x_before.shape[1:] #grabbing h and w assuming [C, H, W]
        new_h, new_w = self.output_size

        assert h >= new_h , "Output height is larger than original height"
        assert w >= new_w, "Output width is larger than original width"

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        x_before = x_before[:, top:top+new_h, left:left+new_w]
        x_after = x_after[:, top:top+new_h, left:left+new_w]
        y = y[top:top+new_h, left:left+new_w]

        return {'x_before': x_before, 'x_after': x_after, 'y': y}
    

class RandomFlipPair(object):
    """Random horizontal and vertical flips for paired images and mask."""
    def __call__(self, sample):
        x_before, x_after, y = sample['x_before'], sample['x_after'], sample['y']

        if random.random() > 0.5:
            x_before = torch.flip(x_before, dims=[2])
            x_after = torch.flip(x_after, dims=[2])
            y = torch.flip(y, dims=[1])

        if random.random() > 0.5:
            x_before = torch.flip(x_before, dims=[1])
            x_after = torch.flip(x_after, dims=[1])
            y = torch.flip(y, dims=[0])

        return {'x_before': x_before, 'x_after': x_after, 'y': y}


class RandomRotationPair(object):
    """Random rotation by 90, 180, or 270 degrees for paired images and mask."""
    def __call__(self, sample):
        x_before, x_after, y = sample['x_before'], sample['x_after'], sample['y']
        k = random.randint(1, 3)  # 1 -> 90°, 2 -> 180°, 3 -> 270°

        x_before = torch.rot90(x_before, k, dims=[1,2])
        x_after = torch.rot90(x_after, k, dims=[1,2])
        y = torch.rot90(y, k, dims=[0,1])

        return {'x_before': x_before, 'x_after': x_after, 'y': y}


def standardize(data: torch.Tensor, dim: int = 1, eps: float = 1e-8):
    
    means = data.mean(dim=dim, keepdim=True)
    stds = data.std(dim=dim, keepdim=True)
    normalized = (data - means) / (stds + eps)
    return normalized


def weights(dataloader, num_classes=3, ignore_index=0, device="cpu"):
    num_pixels = torch.zeros(num_classes, dtype=torch.float, device=device)

    for _, y in dataloader:   # y has shape [B, H, W]
        y = y.to(device)
        for c in range(num_classes):
            if c != ignore_index:
                num_pixels[c] += (y == c).sum()

    return num_pixels



# def calc_batch_metrics(logits, y, ignore_index=None):
#     predictions = torch.argmax(logits, dim=1)
#     eps = .1e-6

#     if ignore_index is not None:
#         mask = (y != ignore_index)
#     else:
#         mask = torch.ones_like(y, dtype=torch.bool)

#     TP = ((predictions == y) & mask & (y != 0)).sum().item() 
#     FP = ((predictions != y) & mask & (predictions != 0)).sum().item()
#     FN = ((predictions != y) & mask & (y != 0)).sum().item()
#     TN = ((predictions == y) & mask & (y == 0)).sum().item()

#     return TP, FP, FN, TN

def calc_batch_metrics(logits, y, ignore_index=None, positive_class=2, negative_class=1):
    predictions = torch.argmax(logits, dim=1)

    if ignore_index is not None:
        mask = (y != ignore_index)
    else:
        mask = torch.ones_like(y, dtype=torch.bool)

    TP = ((predictions == positive_class) & (y == positive_class) & mask).sum().item()
    TN = ((predictions == negative_class) & (y == negative_class) & mask).sum().item()
    FP = ((predictions == positive_class) & (y == negative_class) & mask).sum().item()
    FN = ((predictions == negative_class) & (y == positive_class) & mask).sum().item()

    return TP, FP, FN, TN


def calc_epoch_metrics(TP, FP, FN, TN):
    eps = .1e-6

    accuracy = TP /(TP + FP + TN + FN + eps)
    precision = TP /(TP + FP + eps)
    recall = TP /(TP + FN + eps)
    f1 = (2*precision*recall) /(precision + recall  + eps)
    iou = TP/ (TP + FP + FN + eps)
    
    results = {"accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "IoU": iou}

    return results