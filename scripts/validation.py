import os
import multiprocessing
from multiprocessing import Pool
from functools import partial
import argparse
import numpy as np
import pandas as pd
import awkward as ak
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

from tthbb_spanet.lib.dataset.h5 import DCTRDataset
from ttbb_dctr.models.binary_classifier import BinaryClassifier
from ttbb_dctr.lib.data_preprocessing import get_tensors, get_device, get_datasets_list, get_input_features
from ttbb_dctr.lib.plotting import plot_correlation, plot_correlation_matrix

def load_model(log_directory):
    assert "checkpoints" in os.listdir(os.path.join(log_directory)), "No checkpoints found in log directory"
    assert "hparams.yaml" in os.listdir(log_directory), "No hparams.yaml found in log directory"
    hparams = OmegaConf.load(os.path.join(log_directory, "hparams.yaml"))
    checkpoint = os.path.join(log_directory, "checkpoints", "last.ckpt")
    model = BinaryClassifier.load_from_checkpoint(checkpoint, **hparams)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_directory', type=str, help="Pytorch Lightning Log directory containing the checkpoint and hparams.yaml file.")
    parser.add_argument('--cfg', type=str, help="Config file with parameters for data preprocessing and training", required=True)
    parser.add_argument('-j', '--workers', type=int, default=8, help="Number of workers for parallel processing")
    args = parser.parse_args()

    if args.workers > multiprocessing.cpu_count():
        print(f"Number of workers ({args.workers}) is greater than number of CPUs ({multiprocessing.cpu_count()}). Setting number of workers to {multiprocessing.cpu_count()}")
        args.workers = multiprocessing.cpu_count()
    plot_dir = os.path.join(args.log_directory, "plots")

    device = get_device()

    cfg = OmegaConf.load(args.cfg)
    cfg_training = cfg["training"]

    # Load model and compute predictions to extract the DCTR weight
    events_train = ak.from_parquet(cfg_training["training_file"])
    events_test = ak.from_parquet(cfg_training["test_file"])
    events = ak.concatenate((events_train, events_test))
    X_train, Y_train, W_train = get_tensors(events_train, normalize_inputs=True, normalize_weights=True)
    X_test, Y_test, W_test = get_tensors(events_test, normalize_inputs=True, normalize_weights=True)
    X_full = torch.concatenate((X_train, X_test))
    Y_full = torch.concatenate((Y_train, Y_test))
    W_full = torch.concatenate((W_train, W_test))
    model = load_model(args.log_directory)
    model.eval()
    with torch.no_grad():
        predictions = model(X_full)
    predictions = F.sigmoid(predictions)
    score = ak.Array(predictions[:,0].to("cpu").detach().numpy())
    weight_ttbb = score / (1-score)

    # Select events in control region, without cut on tthbb
    mask_data = (events.data == 1)
    mask_mc = (events.data == 0)
    mask_ttbb = mask_mc & (events.ttbb == 1)
    mask_ttcc = mask_mc & (events.ttcc == 1)
    mask_ttlf = mask_mc & (events.ttlf == 1)

    os.makedirs(plot_dir, exist_ok=True)
    for subdir in ["correlations_spanet", "correlation_matrix"]:
        os.makedirs(os.path.join(plot_dir, subdir), exist_ok=True)
    for mask, title in zip([mask_data, mask_mc, mask_ttbb, mask_ttcc, mask_ttlf],
                        ["Data", "MC", "ttbb", "ttcc", "ttlf"]):
        input_features = get_input_features(events[mask])
        for score in ["tthbb_transformed", "ttlf"]:
            y = events[mask].spanet_output[score]
            w = events[mask].event.weight
            def f(varname_x):
                return plot_correlation(input_features[varname_x], y, w, varname_x, title, score, os.path.join(plot_dir, "correlations_spanet"))
            if args.workers == 1:
                for varname_x in input_features.keys():
                    f(varname_x)
            else:
                with Pool(processes=args.workers) as pool:
                    pool.map(f, list(input_features.keys()))
                    pool.close()
                    pool.join()

        df = pd.DataFrame(input_features)
        df["w_dctr"] = weight_ttbb[mask]
        plot_correlation_matrix(df, title, os.path.join(plot_dir, "correlation_matrix"))
