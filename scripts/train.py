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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help="Config file with parameters for data preprocessing and training", required=True)
    parser.add_argument('-l', '--log_dir', type=str, help="Output folder", required=True)
    parser.add_argument('--save', action='store_true', help="Save the train and test datasets")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    cfg_model = cfg["model"]

    os.makedirs(args.log_dir, exist_ok=True)
    train_dataloader, val_dataloader = get_dataloaders(cfg)

    # Instantiate the model
    input_size = len(input_features)
    model = BinaryClassifier(input_size, **cfg_model, learning_rate=cfg_training["learning_rate"])

    # Move the model to device and set training mode
    model = model.to(device)
    print("Initialized model:")
    print(model)

    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=-1, filename="{epoch:03d}-{step:03d}-{val_loss:.2f}", save_last=True, verbose=True),
        EarlyStopping(monitor="val_loss", min_delta=0.00, patience=cfg_training["patience"], verbose=False, mode="min"),
        LearningRateMonitor(logging_interval='epoch')
    ]
    trainer = Trainer(max_epochs=cfg_training["epochs"], default_root_dir=args.log_dir, callbacks=callbacks)
    print("Training model...")
    trainer.fit(model, train_dataloader, val_dataloader)
