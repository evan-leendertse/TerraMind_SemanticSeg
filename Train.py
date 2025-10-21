# Download from corresponding files in folder
from Decoder_UNet2D import UNet2D
from Encoder_TerraMind import TerraMindEncoder
from DataLoader import BeforeData
from utils import RandomFlipPair, RandomRotationPair, weights, calc_batch_metrics, calc_epoch_metrics, move_to_device, set_seeds, create_writer, save_checkpoint
from DataLoader import MultiModalBeforeAfterDataset

#General Requirements
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from monai.losses.dice import DiceLoss # replaced the above loss
import hydra
from terratorch.models import necks
from terratorch.registry import BACKBONE_REGISTRY
import albumentations
import numpy as np
import random
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------Set up train function using config.yaml including writer to save model details + metrics + config file --------------------------------------- #
@hydra.main(version_base = None, config_path = "configs" , config_name = 'config') #this should be right before main function
def main(cfg):
    set_seeds(cfg.model.seed)
    train_modalities = {
        name: (paths.before, paths.after)
        for name, paths in cfg.train_loader.modalities.items()}

    val_modalities = {
        name: (paths.before, paths.after)
        for name, paths in cfg.validation_loader.modalities.items()}


# ------------------------------Loading in data & setting up model  --------------------------------------- #    
    train_data = MultiModalBeforeAfterDataset(modalities = train_modalities,
        label_dir = cfg.train_loader.label_dir,
        split = 'train', #probably does not need to be a param in config file
        num_augmentations = cfg.train_loader.num_augmentations,
        patch_size = cfg.train_loader.patch_size,
        stride = cfg.train_loader.stride,
        add_random_offset = cfg.train_loader.add_random_offset,
        preload = cfg.train_loader.preload)
    train_dataloader = DataLoader(train_data,
                                  batch_size = cfg.train_loader.batch_size,
                                  shuffle = cfg.train_loader.shuffle,
                                  num_workers = cfg.train_loader.num_workers)

    val_data = MultiModalBeforeAfterDataset(
        modalities = val_modalities,
        label_dir = cfg.validation_loader.label_dir,
        split = 'validation', #probably not necessary in configs 
        patch_size =  cfg.validation_loader.patch_size,
        stride =  cfg.validation_loader.stride, #probably could recycle from train
        add_random_offset = cfg.validation_loader.add_random_offset, #probably could recycle from train
        preload = cfg.validation_loader.preload) #probably could recycle from train
    val_dataloader = DataLoader(val_data, 
                                batch_size = cfg.validation_loader.batch_size, 
                                shuffle = cfg.validation_loader.shuffle,
                                num_workers = cfg.validation_loader.num_workers)


    if cfg.model.apply_weight_loss:
        weights_ = weights(train_dataloader, num_classes=cfg.model.num_classes, ignore_index=cfg.model.ignore_index, device = device)
        criterion = nn.CrossEntropyLoss(weight = weights_, ignore_index= cfg.model.ignore_index)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=cfg.model.ignore_index)


    encoder = TerraMindEncoder(version = cfg.model.TM_version, 
                               pretrained =  cfg.model.pretrained, 
                               modalities =  list(cfg.model.modalities))
                                # modalities =  ["S2L2A", "S1GRD"])
    decoder = UNet2D(num_classes= cfg.model.num_classes)
    encoder.to(device)
    decoder.to(device)

    if cfg.model.TM_finetune:
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=cfg.model.learning_rate)
    else:
        optimizer = optim.Adam(decoder.parameters(),
                               lr=cfg.model.learning_rate)



# ---------------------------------------- Training Loop ---------------------------------------------- #

    writer, log_dir = create_writer(cfg.writer.experiment_name, cfg.writer.experiment_number)
    encoder.eval()

    best_val_loss = float("inf")
    for epoch in range(cfg.model.num_epochs):
        decoder.train() 
        running_train_loss = 0.0
        running_val_loss = 0.0
        TP = FP = FN = TN = 0.0
        
        for x, y in train_dataloader:
            x = move_to_device(x, device)
            y = y.to(device)

            z_before, z_after = encoder(x["before"]), encoder(x["after"]) #
            z_differenced = [after - before for before, after in zip(z_before, z_after)]
            logits = decoder(z_differenced)

            train_loss = criterion(logits, y)
            sz_batch = next(iter(x["before"].values())).size(0)
            running_train_loss += train_loss.item() * sz_batch

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
                

        epoch_loss = running_train_loss / len(train_data)
        print(f"Epoch {epoch+1}/{cfg.model.num_epochs} - Train Loss: {epoch_loss:.4f}")

        decoder.eval()
        with torch.no_grad():
            for x, y in val_dataloader:
                x = move_to_device(x, device)
                y = y.to(device)

                z_before, z_after = encoder(x["before"]), encoder(x["after"])
                z_differenced = [after - before for before, after in zip(z_before, z_after)]
                logits = decoder(z_differenced)
                
                batch_val_loss = criterion(logits, y)
                sz_batch = next(iter(x["before"].values())).size(0)
                running_val_loss += batch_val_loss.item() * sz_batch

                batch_metrics = calc_batch_metrics(logits, y, ignore_index = cfg.model.ignore_index, positive_class = cfg.model.positive_class, negative_class = cfg.model.negative_class)
                TP, FP, FN, TN = [x + y for x, y in zip((TP, FP, FN, TN), batch_metrics)]

            
            val_loss = running_val_loss/len(val_data)
            print(f"--Validation Loss: {val_loss}")


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # save_checkpoint(encoder, decoder, optimizer, epoch, val_loss, 
                #                 {"learning_rate": cfg.model.learning_rate, 
                #                  "Epochs": cfg.model.num_epochs, 
                #                  "log_dir": log_dir},
                #                 save_dir=os.path.join(log_dir, "checkpoints"))

                save_checkpoint(
                    encoder, decoder, optimizer, epoch, val_loss,
                    cfg,
                    save_dir=os.path.join(log_dir, "checkpoints")
)


            epoch_metrics = calc_epoch_metrics(TP, FP, FN, TN)
            print(f'--IoU: {epoch_metrics["IoU"]:.4f}\
            - Accuracy: {epoch_metrics["Accuracy"]:.4f}\
            - Precision: {epoch_metrics["Precision"]:.4f}\
            - Recall: {epoch_metrics["Recall"]:.4f}\
            - F1: {epoch_metrics["F1"]:.4f}\n')


        writer.add_scalars("Loss",
                        tag_scalar_dict={"train": epoch_loss,
                                            "Validation": val_loss},
                            global_step= epoch)
        writer.add_scalars("Metrics",
                        tag_scalar_dict={"IoU": epoch_metrics['IoU']},
                        global_step= epoch)
        writer.add_scalars("Metrics",
                        tag_scalar_dict={"Accuracy": epoch_metrics['Accuracy']},
                        global_step= epoch)
        writer.add_scalars("Metrics",
                        tag_scalar_dict={"Precision": epoch_metrics['Precision']},
                        global_step= epoch)
        writer.add_scalars("Metrics",
                        tag_scalar_dict={"Recall": epoch_metrics['Recall']},
                        global_step= epoch)
        writer.add_scalars("Metrics",
                        tag_scalar_dict={"F1": epoch_metrics['F1']},
                        global_step= epoch)
    writer.close()

if __name__ == "__main__":
    main()