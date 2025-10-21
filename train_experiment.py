import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from itertools import product


from Decoder_UNet2D import UNet2D
from Encoder_TerraMind import TerraMindEncoder
from DataLoader import MultiModalBeforeAfterDataset
from utils import (move_to_device, set_seeds, create_writer,
                   save_checkpoint, weights, calc_batch_metrics, calc_epoch_metrics)


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(cfg, overrides=None):
    # ---------------- Apply overrides ---------------- #
    if overrides:
        for k, v in overrides.items():
            if k == "learning_rate":
                cfg.model.learning_rate = v
            if k == "apply_weight_loss":
                cfg.model.apply_weight_loss = v
            if k == "TM_finetune":
                cfg.model.TM_finetune = v
            if k == "num_epochs":
                cfg.model.num_epochs = v
            elif k == "train_batch_size":
                cfg.train_loader.batch_size = v
            elif k == "num_augmentations":
                cfg.train_loader.num_augmentations = v
            elif k == "stride": ########### make sure that these two strides are the same
                cfg.train_loader.stride = v
                cfg.validation_loader.stride = v
            elif k == "add_random_offset":
                cfg.train_loader.add_random_offset = v
                cfg.validation_loader.add_random_offset = v
            elif k == "experiment_number":
                cfg.writer.experiment_number = v
            # Add more overrides if needed

    set_seeds(cfg.model.seed)

    # ---------------- Prepare data ---------------- #
    train_data = MultiModalBeforeAfterDataset(
        modalities={name: (paths.before, paths.after) 
                    for name, paths in cfg.train_loader.modalities.items()},
        label_dir=cfg.train_loader.label_dir,
        split='train',
        num_augmentations=cfg.train_loader.num_augmentations,
        patch_size=cfg.train_loader.patch_size,
        stride=cfg.train_loader.stride,
        add_random_offset=cfg.train_loader.add_random_offset,
        preload=cfg.train_loader.preload
    )
    train_dataloader = DataLoader(train_data,
                                  batch_size=cfg.train_loader.batch_size,
                                  shuffle=cfg.train_loader.shuffle,
                                  num_workers=cfg.train_loader.num_workers)

    val_data = MultiModalBeforeAfterDataset(
        modalities={name: (paths.before, paths.after)
                    for name, paths in cfg.validation_loader.modalities.items()},
        label_dir=cfg.validation_loader.label_dir,
        split='validation',
        patch_size=cfg.validation_loader.patch_size,
        stride=cfg.validation_loader.stride,
        add_random_offset=cfg.validation_loader.add_random_offset,
        preload=cfg.validation_loader.preload
    )
    val_dataloader = DataLoader(val_data,
                                batch_size=cfg.validation_loader.batch_size,
                                shuffle=cfg.validation_loader.shuffle,
                                num_workers=cfg.validation_loader.num_workers)

    # ---------------- Loss & model ---------------- #
    if cfg.model.apply_weight_loss:
        weights_ = weights(train_dataloader, num_classes=cfg.model.num_classes,
                           ignore_index=cfg.model.ignore_index, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights_, ignore_index=cfg.model.ignore_index)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=cfg.model.ignore_index)

    encoder = TerraMindEncoder(version=cfg.model.TM_version,
                               pretrained=cfg.model.pretrained,
                               modalities=list(cfg.model.modalities))
    decoder = UNet2D(num_classes=cfg.model.num_classes)

    encoder.to(device)
    decoder.to(device)

    if cfg.model.TM_finetune:
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=cfg.model.learning_rate)
    else:
        optimizer = optim.Adam(decoder.parameters(), lr=cfg.model.learning_rate)

    # ---------------- Writer ---------------- #
    writer, log_dir = create_writer(cfg.writer.experiment_name, cfg.writer.experiment_number)

    # ---------------- Training Loop ---------------- #
    best_val_loss = float("inf")
    encoder.eval()  # freeze encoder

    for epoch in range(cfg.model.num_epochs):
        decoder.train()
        running_train_loss = 0.0
        running_val_loss = 0.0
        TP = FP = FN = TN = 0.0

        for x, y in train_dataloader:
            x = move_to_device(x, device)
            y = y.to(device)

            z_before, z_after = encoder(x["before"]), encoder(x["after"])
            z_diff = [after - before for before, after in zip(z_before, z_after)]
            logits = decoder(z_diff)

            train_loss = criterion(logits, y)
            batch_size = next(iter(x["before"].values())).size(0)
            running_train_loss += train_loss.item() * batch_size

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        epoch_loss = running_train_loss / len(train_data)
        print(f"Epoch {epoch+1}/{cfg.model.num_epochs} - Train Loss: {epoch_loss:.4f}")

        # ---------------- Validation ---------------- #
        decoder.eval()
        with torch.no_grad():
            for x, y in val_dataloader:
                x = move_to_device(x, device)
                y = y.to(device)

                z_before, z_after = encoder(x["before"]), encoder(x["after"])
                z_diff = [after - before for before, after in zip(z_before, z_after)]
                logits = decoder(z_diff)

                val_loss = criterion(logits, y)
                batch_size = next(iter(x["before"].values())).size(0)
                running_val_loss += val_loss.item() * batch_size

                batch_metrics = calc_batch_metrics(logits, y,
                                                   ignore_index=cfg.model.ignore_index,
                                                   positive_class=cfg.model.positive_class,
                                                   negative_class=cfg.model.negative_class)
                TP, FP, FN, TN = [x + y for x, y in zip((TP, FP, FN, TN), batch_metrics)]

            val_loss = running_val_loss / len(val_data)
            print(f"--Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(encoder, decoder, optimizer, epoch, val_loss, cfg,
                                save_dir=os.path.join(log_dir, "checkpoints"))

            epoch_metrics = calc_epoch_metrics(TP, FP, FN, TN)
            print(f"--Metrics: IoU {epoch_metrics['IoU']:.4f}, "
                  f"Accuracy {epoch_metrics['Accuracy']:.4f}, "
                  f"Precision {epoch_metrics['Precision']:.4f}, "
                  f"Recall {epoch_metrics['Recall']:.4f}, "
                  f"F1 {epoch_metrics['F1']:.4f}\n")

        # ---------------- TensorBoard ---------------- #
        writer.add_scalars("Loss", {"train": epoch_loss, "val": val_loss}, global_step=epoch)
        for metric in ["IoU", "Accuracy", "Precision", "Recall", "F1"]:
            writer.add_scalar(f"Metrics/{metric}", epoch_metrics[metric], epoch)

    writer.close()


if __name__ == "__main__":
    # ---------------- Load config ---------------- #
    cfg = OmegaConf.load("configs/config.yaml")

    # ---------------- Define parameter options ---------------- #
    param_options = {
        "learning_rate": [.0005, .00005, .000005],
        # "apply_weight_loss": [True, False],
        # "TM_finetune": [True],
        "num_epochs": [50],
        "train_batch_size": [8, 16, 32],
        "num_augmentations":  [0, 1, 5, 20],
        # "stride": [224, 200],
        "add_random_offset": [0, 20]
        }

    # ---------------- Create cartesian-product param grid ---------------- #
    keys = list(param_options.keys())
    values = list(param_options.values())
    param_grid = [dict(zip(keys, v)) for v in product(*values)]

    # ---------------- Run all experiments ---------------- #
    for i, params in enumerate(param_grid):
        params["experiment_number"] = i + 1
        print(f"\n=== Running experiment {i+1}/{len(param_grid)} ===")
        cfg_copy = copy.deepcopy(cfg)
        main(cfg_copy, overrides=params)