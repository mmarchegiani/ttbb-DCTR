import os
import argparse
import numpy as np
import awkward as ak

from omegaconf import OmegaConf

import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ttbb_dctr.lib.data_preprocessing import get_tensors, get_dataloader
from ttbb_dctr.models.binary_classifier import BinaryClassifier
from ttbb_dctr.utils.utils import get_device
from tthbb_spanet import DCTRDataset

def save_config(cfg, filename):
    with open(filename, "w") as f:
        OmegaConf.save(cfg, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help="Config file with parameters for data preprocessing and training", required=True)
    parser.add_argument('-l', '--log_dir', type=str, help="Output folder", required=True)
    parser.add_argument('--threshold', action='store_true', help="Apply threshold to the DCTR output in the loss function")
    parser.add_argument('--dry', action='store_true', help="Dry run, do not train the model")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    cfg_model = cfg["model"]
    cfg_training = cfg["training"]

    events_train = ak.from_parquet(cfg_training["training_file"])
    events_test = ak.from_parquet(cfg_training["test_file"])
    X_train, Y_train, W_train = get_tensors(events_train, normalize_inputs=True, normalize_weights=True)
    X_test, Y_test, W_test = get_tensors(events_test, normalize_inputs=True, normalize_weights=True)
    train_dataloader = get_dataloader(X_train, Y_train, W_train, batch_size=cfg_training["batch_size"], shuffle=True)
    val_dataloader = get_dataloader(X_test, Y_test, W_test, batch_size=cfg_training["batch_size"], shuffle=False) # Do not shuffle validation dataset
    # Instantiate the model
    input_size = X_train.shape[1]
    if args.threshold:
        model = BinaryClassifierWithThreshold(input_size, **cfg_model, learning_rate=cfg_training["learning_rate"], weight_decay=cfg_training.get("weight_decay", 0), score_threshold=cfg_training.get("score_threshold", 0.1))
    else:
        model = BinaryClassifier(input_size, **cfg_model, learning_rate=cfg_training["learning_rate"], weight_decay=cfg_training.get("weight_decay", 0))

    # Move the model to device and set training mode
    device = get_device()
    model = model.to(device)
    print("Initialized model:")
    print(model)

    if not args.dry:
        os.makedirs(args.log_dir, exist_ok=True)
        callbacks = [
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=-1, filename="{epoch:03d}-{step:03d}-{val_loss:.2f}", save_last=True, verbose=True),
            EarlyStopping(monitor="val_loss", min_delta=0.00, patience=cfg_training["patience"], verbose=False, mode="min"),
            LearningRateMonitor(logging_interval='epoch', log_weight_decay=True)
        ]
        trainer = Trainer(max_epochs=cfg_training["epochs"], default_root_dir=args.log_dir, callbacks=callbacks)
        print("Training model...")
        trainer.fit(model, train_dataloader, val_dataloader)
        save_config(cfg_training, os.path.join(trainer.logger.log_dir, "cfg_training.yaml"))
