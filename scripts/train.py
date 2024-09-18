import os
import argparse
import numpy as np
import awkward as ak

from omegaconf import OmegaConf

import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split

from tthbb_spanet.lib.dataset.h5 import Dataset
from ttbb_dctr.lib.data_preprocessing import _stack_arrays, get_tensors, get_dataloader, get_datasets_list, get_events, get_cr_mask, get_njet_reweighting, get_input_features
from ttbb_dctr.models.binary_classifier import BinaryClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help="Config file with parameters for training", required=True)
    parser.add_argument('-l', '--log_dir', type=str, help="Output folder", required=True)
    parser.add_argument('--save', action='store_true', help="Save the train and test datasets")
    parser.add_argument('--test', action='store_true', help="Run in test mode")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    cfg_input = cfg["input"]
    cfg_cuts = cfg["cuts"]
    cfg_preprocessing = cfg["preprocessing"]
    cfg_model = cfg["model"]
    cfg_training = cfg["training"]

    os.makedirs(args.log_dir, exist_ok=True)
    datasets = get_datasets_list(cfg_input)
    dataset_training = Dataset(datasets, cfg_preprocessing, frac_train=1.0, shuffle=False, reweigh=True, has_data=True)
    #dataset_training.save(os.path.join(args.log_dir, "datasets", "dataset_full_Run2.parquet"))
    events = get_events(dataset_training, shuffle=cfg_preprocessing["shuffle"], seed=cfg_preprocessing["seed"])
    mask_cr = get_cr_mask(events, cfg_cuts["cr1"])

    # Select events in control region
    events = events[mask_cr]

    # Run only over 10 batches for testing
    if args.test:
        events = events[:cfg_training["batch_size"]*10]

    # Define event classes
    mask_data = (events.data == 1)
    mask_data_minus_minor_bkg = (events.data == 1) | (events.ttcc == 1) | (events.ttlf == 1) | (events.tt2l2nu == 1) | (events.wjets == 1) | (events.singletop == 1) | (events.tthbb == 1)
    mask_data_minus_minor_bkg_no_tthbb = (events.data == 1) | (events.ttcc == 1) | (events.ttlf == 1) | (events.tt2l2nu == 1) | (events.wjets == 1) | (events.singletop == 1)
    mask_ttbb = (events.ttbb == 1)
    w = events.event.weight
    w_nj = get_njet_reweighting(events, mask_data_minus_minor_bkg, mask_ttbb)
    input_features = get_input_features(events)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    X, Y, W = get_tensors(input_features, events.dctr, events.event.weight * w_nj, device=device, normalize_inputs=True, normalize_weights=True)

    # Since the shuffling is already performed by permutation of the events, we don't shuffle the tensors here in order to maintain the same order in events and X_train, X_test
    X_train, X_test = train_test_split(X, test_size=cfg_preprocessing["test_size"], shuffle=False)
    Y_train, Y_test = train_test_split(Y, test_size=cfg_preprocessing["test_size"], shuffle=False)
    W_train, W_test = train_test_split(W, test_size=cfg_preprocessing["test_size"], shuffle=False)

    if cfg_preprocessing["num_workers"]:
        if cfg_preprocessing["num_workers"] > 0:
            raise NotImplementedError("num_workers > 0: multiprocessing is not supported yet in the data preprocessing")

    # By setting shuffle=True, the DataLoader will shuffle the data at the beginning of each epoch
    train_dataloader = get_dataloader(X_train, Y_train, W_train, batch_size=cfg_training["batch_size"], num_workers=cfg_preprocessing["num_workers"], shuffle=True)
    val_dataloader = get_dataloader(X_test, Y_test, W_test, batch_size=cfg_training["batch_size"], num_workers=cfg_preprocessing["num_workers"], shuffle=True)

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
