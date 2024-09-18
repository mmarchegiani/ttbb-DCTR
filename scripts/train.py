import os
import argparse
import numpy as np
import awkward as ak

from omegaconf import OmegaConf

import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ttbb_dctr.lib.data_preprocessing import get_dataloaders
from ttbb_dctr.models.binary_classifier import BinaryClassifier
from ttbb_dctr.utils.utils import get_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help="Config file with parameters for data preprocessing and training", required=True)
    parser.add_argument('-l', '--log_dir', type=str, help="Output folder", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    cfg_model = cfg["model"]
    cfg_training = cfg["training"]

    os.makedirs(args.log_dir, exist_ok=True)
    train_dataloader, val_dataloader, input_size = get_dataloaders(cfg, input_size=True)

    # Instantiate the model
    model = BinaryClassifier(input_size, **cfg_model, learning_rate=cfg_training["learning_rate"], weight_decay=cfg_training["weight_decay"])

    # Move the model to device and set training mode
    device = get_device()
    model = model.to(device)
    print("Initialized model:")
    print(model)

    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=-1, filename="{epoch:03d}-{step:03d}-{val_loss:.2f}", save_last=True, verbose=True),
        EarlyStopping(monitor="val_loss", min_delta=0.00, patience=cfg_training["patience"], verbose=False, mode="min"),
        LearningRateMonitor(logging_interval='epoch', log_weight_decay=True)
    ]
    trainer = Trainer(max_epochs=cfg_training["epochs"], default_root_dir=args.log_dir, callbacks=callbacks)
    print("Training model...")
    trainer.fit(model, train_dataloader, val_dataloader)
