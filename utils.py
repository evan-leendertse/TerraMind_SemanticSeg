import numpy as np
import torch
from torchvision import transforms
import random
from datetime import datetime
import os
from  torch.utils.tensorboard.writer import SummaryWriter
import yaml
from omegaconf import OmegaConf
import glob

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
    


class RandomFlipPair:
    def __call__(self, sample):
        before, after, y = sample['before'], sample['after'], sample['y']

        if random.random() > 0.5:
            for m in before.keys():
                before[m] = torch.flip(before[m], dims=[2])
                after[m]  = torch.flip(after[m], dims=[2])
            y = torch.flip(y, dims=[1])

        if random.random() > 0.5:
            for m in before.keys():
                before[m] = torch.flip(before[m], dims=[1])
                after[m]  = torch.flip(after[m], dims=[1])
            y = torch.flip(y, dims=[0])

        return {'before': before, 'after': after, 'y': y}



# class RandomFlipPair(object):
#     """Random horizontal and vertical flips for paired images and mask."""
#     def __call__(self, sample):
#         x_before, x_after, y = sample['x_before'], sample['x_after'], sample['y']

#         if random.random() > 0.5:
#             x_before = torch.flip(x_before, dims=[2])
#             x_after = torch.flip(x_after, dims=[2])
#             y = torch.flip(y, dims=[1])

#         if random.random() > 0.5:
#             x_before = torch.flip(x_before, dims=[1])
#             x_after = torch.flip(x_after, dims=[1])
#             y = torch.flip(y, dims=[0])

#         return {'x_before': x_before, 'x_after': x_after, 'y': y}




class RandomRotationPair:
    def __call__(self, sample):
        before, after, y = sample['before'], sample['after'], sample['y']
        k = random.randint(0, 3)

        for m in before.keys():
            before[m] = torch.rot90(before[m], k, dims=[1, 2])
            after[m]  = torch.rot90(after[m], k, dims=[1, 2])
        y = torch.rot90(y, k, dims=[0, 1])

        return {'before': before, 'after': after, 'y': y}



# class RandomRotationPair(object):
#     """Random rotation by 90, 180, or 270 degrees for paired images and mask."""
#     def __call__(self, sample):
#         x_before, x_after, y = sample['x_before'], sample['x_after'], sample['y']
#         k = random.randint(1, 3)  # 1 -> 90°, 2 -> 180°, 3 -> 270°

#         x_before = torch.rot90(x_before, k, dims=[1,2])
#         x_after = torch.rot90(x_after, k, dims=[1,2])
#         y = torch.rot90(y, k, dims=[0,1])

#         return {'x_before': x_before, 'x_after': x_after, 'y': y}


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

    accuracy = (TP+TN) /(TP + FP + TN + FN + eps)
    precision = TP /(TP + FP + eps)
    recall = TP /(TP + FN + eps)
    f1 = (2*precision*recall) /(precision + recall  + eps)
    iou = TP/ (TP + FP + FN + eps)
    
    results = {"Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou}

    return results


def move_to_device(batch, device):
    """
    Recursively moves a nested dict or list of tensors to the given device.
    """
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch
    


# Set seeds
def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

    


def create_writer(experiment_name, experiment_number):
    log_dir = os.path.join("runs", experiment_name, str(experiment_number))
    writer = SummaryWriter(log_dir=log_dir)
    return writer, log_dir



def save_checkpoint(encoder, decoder, optimizer, epoch, val_loss, cfg, save_dir="checkpoints"):
    """
    Save model checkpoint and config, deleting previous best files.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---------------- Delete previous best checkpoint ---------------- #
    old_checkpoints = glob.glob(os.path.join(save_dir, "best_model_*.pt"))
    for f in old_checkpoints:
        os.remove(f)

    # ---------------- Delete previous best config ---------------- #
    old_configs = glob.glob(os.path.join(save_dir, "best_config_*.yaml"))
    for f in old_configs:
        os.remove(f)

    # ---------------- Save new best checkpoint ---------------- #
    checkpoint_path = os.path.join(save_dir, f"best_model_epoch{epoch}_valloss{val_loss:.4f}.pt")
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, checkpoint_path)

    # ---------------- Save config used for this checkpoint ---------------- #
    cfg_path = os.path.join(save_dir, f"best_config_epoch{epoch}_valloss{val_loss:.4f}.yaml")
    OmegaConf.save(cfg, cfg_path)

    print(f"Saved new best checkpoint: {checkpoint_path}")
    print(f"Saved corresponding config: {cfg_path}")



# def save_checkpoint(encoder, decoder, optimizer, epoch, val_loss, cfg, save_dir="checkpoints"):
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Save the model checkpoint
#     save_path = os.path.join(save_dir, f"model_epoch{epoch}_valloss{val_loss:.4f}.pt")
#     torch.save({
#         'epoch': epoch,
#         'encoder_state_dict': encoder.state_dict(),
#         'decoder_state_dict': decoder.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'val_loss': val_loss,
#     }, save_path)
    
#     # Save the full Hydra config as YAML
#     cfg_path = os.path.join(save_dir, f"config_epoch{epoch}.yaml")
#     cfg_dict = OmegaConf.to_container(cfg, resolve=True)
#     with open(cfg_path, 'w') as f:
#         yaml.dump(cfg_dict, f)

# def save_checkpoint(encoder, decoder, optimizer, epoch, val_loss, config, save_dir="checkpoints"):
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"model_epoch{epoch}_valloss{val_loss:.4f}.pt")
    
#     torch.save({
#         'epoch': epoch,
#         'encoder_state_dict': encoder.state_dict(),
#         'decoder_state_dict': decoder.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'val_loss': val_loss,  # this is your hyperparameter dictionary
#     }, save_path)




    # print(f"✅ Saved checkpoint: {save_path}")