import os
import json
import multiprocessing
from multiprocessing import Pool
from functools import partial
import argparse
import numpy as np
import pandas as pd
import awkward as ak
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")

from tthbb_spanet import DCTRDataset
from ttbb_dctr.lib.data_preprocessing import get_input_features, get_tensors
from ttbb_dctr.lib.plotting import plot_shapes

def load_training_config(log_directory):
    filename = "cfg_training.yaml"
    assert filename in os.listdir(log_directory), f"No {filename} found in log directory"
    cfg_training = OmegaConf.load(os.path.join(log_directory, filename))
    return cfg_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_directory', type=str, help="Pytorch Lightning Log directory containing the checkpoint and hparams.yaml file.")
    parser.add_argument('--cfg', type=str, default=None, help="Config file with parameters for data preprocessing and training. If passed as argument, it overrides the configuration stored in the Pytorch log directory.", required=False)
    parser.add_argument('--plot_dir', type=str, default="plots", help="Output folder for plots", required=False)
    parser.add_argument('--density', action="store_true", default=False, help="Plot densities instead of counts", required=False)
    parser.add_argument('-j', '--workers', type=int, default=8, help="Number of workers for parallel processing")
    args = parser.parse_args()

    if args.workers > multiprocessing.cpu_count():
        print(f"Number of workers ({args.workers}) is greater than number of CPUs ({multiprocessing.cpu_count()}). Setting number of workers to {multiprocessing.cpu_count()}")
        args.workers = multiprocessing.cpu_count()
    plot_dir = os.path.join(args.log_directory, args.plot_dir)

    if args.cfg is None:
        cfg_training = load_training_config(args.log_directory)
    else:
        cfg = OmegaConf.load(args.cfg)
        cfg_training = cfg["training"]

    # Load model and compute predictions to extract the DCTR weight
    assert cfg_training.get("training_file", None) is not None or cfg_training.get("test_file", None) is not None, "No training or test file provided"
    if cfg_training.get("training_file", None) is not None:
        dataset_train = DCTRDataset(cfg_training["training_file"], shuffle=False, reweigh=False, label=False)
        events_train = dataset_train.df
        input_features_train = get_input_features(events_train, only=cfg_training.get("input_features", None))
        X_train, Y_train, W_train = get_tensors(input_features_train, events_train.dctr, events_train.event.weight, normalize_inputs=True, normalize_weights=True)
    if cfg_training.get("test_file", None) is not None:
        dataset_test = DCTRDataset(cfg_training["test_file"], shuffle=False, reweigh=False, label=False)
        events_test = dataset_test.df
        input_features_test = get_input_features(events_test, only=cfg_training.get("input_features", None))
        X_test, Y_test, W_test = get_tensors(input_features_test, events_test.dctr, events_test.event.weight, normalize_inputs=True, normalize_weights=True)
    if cfg_training.get("training_file", None) is not None and cfg_training.get("test_file", None) is not None:
        events = ak.concatenate((events_train, events_test))
    if cfg_training.get("training_file", None) is None:
        events_train = None
        events = events_test
        X_full = X_test
        Y_full = Y_test
        W_full = W_test
    elif cfg_training.get("test_file", None) is None:
        events_test = None
        events = events_train
        X_full = X_train
        Y_full = Y_train
        W_full = W_train
    else:
        events = ak.concatenate((events_train, events_test))
        X_full = torch.concatenate((X_train, X_test))
        Y_full = torch.concatenate((Y_train, Y_test))
        W_full = torch.concatenate((W_train, W_test))

    input_features = get_input_features(events, only=cfg_training.get("input_features", None))

    if events_train is not None:
        mask_train = ak.local_index(ak.num(events.JetGood)) < X_train.shape[0]
    else:
        mask_train = ak.zeros_like(ak.num(events.JetGood), dtype=bool)
    mask_test = ~mask_train

    shapes_dir = os.path.join(plot_dir, "shapes")
    for folder in [plot_dir, shapes_dir]:
        os.makedirs(folder, exist_ok=True)
    
    # Plot shapes
    for dataset_type, _events in zip(["train", "test"], [events_train, events_test]):
        if _events is None: continue
        elif len(_events) == 0: continue
        # Define partial function for multiprocessing
        def f(varname_x):
            return plot_shapes(_events, shapes_dir, only_var=varname_x, density=args.density)
        if args.workers == 1:
            plot_shapes(_events, shapes_dir, density=args.density)
        else:
            with Pool(processes=args.workers) as pool:
                pool.map(f, list(input_features.keys()))
                pool.close()
                pool.join()
