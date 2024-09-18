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

from tthbb_spanet.lib.dataset.h5 import Dataset
from ttbb_dctr.models.binary_classifier import BinaryClassifier
from ttbb_dctr.lib.data_preprocessing import get_tensors, get_device, get_datasets_list, get_events, get_input_features
from ttbb_dctr.lib.plotting import plot_correlation

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
    cfg = OmegaConf.load(args.cfg)

    # Load model and compute predictions to extract the DCTR weight
    X_train, X_test, Y_train, Y_test, W_train, W_test = get_tensors(cfg)
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
    MASK_ttbb = np.array(Y_full.to("cpu")) == 0
    MASK_data = np.array(Y_full.to("cpu")) == 1

    device = get_device()
    cfg_input = cfg["input"]
    cfg_preprocessing = cfg["preprocessing"]
    cfg_cuts = cfg["cuts"]
    cfg_training = cfg["training"]
    datasets = get_datasets_list(cfg_input)
    dataset_training = Dataset(datasets, cfg_preprocessing, frac_train=1.0, shuffle=False, reweigh=True, has_data=True)
    events = get_events(dataset_training, shuffle=cfg_preprocessing["shuffle"], seed=cfg_preprocessing["seed"])

    # Select events in control region, without cut on tthbb
    mask_cr_ttlf = events.ttlf < 0.6
    events = events[mask_cr_ttlf]
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
            with Pool(processes=args.workers) as pool:
                pool.map(f, list(input_features.keys()))
                pool.close()
                pool.join()

        df = pd.DataFrame(input_features)
        df["w_dctr"] = weight_ttbb[mask]
        corr = df.corr()
        plot_correlation_matrix(corr, title, os.path.join(plot_dir, "correlation_matrix"))
