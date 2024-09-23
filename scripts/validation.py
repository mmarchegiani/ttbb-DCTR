import os
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

import torch
import torch.nn.functional as F

from tthbb_spanet import DCTRDataset
from ttbb_dctr.models.binary_classifier import BinaryClassifier
from ttbb_dctr.lib.data_preprocessing import get_tensors, get_device, get_datasets_list, get_input_features
from ttbb_dctr.lib.plotting import plot_correlation, plot_correlation_matrix, plot_classifier_score, plot_dctr_weight

def load_model(log_directory):
    assert "checkpoints" in os.listdir(os.path.join(log_directory)), "No checkpoints found in log directory"
    assert "hparams.yaml" in os.listdir(log_directory), "No hparams.yaml found in log directory"
    hparams = OmegaConf.load(os.path.join(log_directory, "hparams.yaml"))
    checkpoint = os.path.join(log_directory, "checkpoints", "last.ckpt")
    print("Loading model from checkpoint:", checkpoint)
    model = BinaryClassifier.load_from_checkpoint(checkpoint, **hparams)
    return model

def load_training_config(log_directory):
    filename = "cfg_training.yaml"
    assert filename in os.listdir(log_directory), f"No {filename} found in log directory"
    cfg_training = OmegaConf.load(os.path.join(log_directory, filename))
    return cfg_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_directory', type=str, help="Pytorch Lightning Log directory containing the checkpoint and hparams.yaml file.")
    parser.add_argument('--cfg', type=str, default=None, help="Config file with parameters for data preprocessing and training. If passed as argument, it overrides the configuration stored in the Pytorch log directory.", required=False)
    parser.add_argument('-j', '--workers', type=int, default=8, help="Number of workers for parallel processing")
    args = parser.parse_args()

    if args.workers > multiprocessing.cpu_count():
        print(f"Number of workers ({args.workers}) is greater than number of CPUs ({multiprocessing.cpu_count()}). Setting number of workers to {multiprocessing.cpu_count()}")
        args.workers = multiprocessing.cpu_count()
    plot_dir = os.path.join(args.log_directory, "plots")

    device = get_device()

    if args.cfg is None:
        cfg_training = load_training_config(args.log_directory)
    else:
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
    events = ak.with_field(events, score, "dctr_score")
    events = ak.with_field(events, weight_ttbb, "dctr_weight")

    # Select events in control region, without cut on tthbb
    mask_data = (events.data == 1)
    mask_mc = (events.data == 0)
    mask_ttbb = mask_mc & (events.ttbb == 1)
    mask_ttcc = mask_mc & (events.ttcc == 1)
    mask_ttlf = mask_mc & (events.ttlf == 1)

    os.makedirs(plot_dir, exist_ok=True)

    for subdir in ["correlations_spanet", "correlation_matrix", "classifier"]:
        os.makedirs(os.path.join(plot_dir, subdir), exist_ok=True)
    """for mask, title in zip([mask_data, mask_mc, mask_ttbb, mask_ttcc, mask_ttlf],
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

        # Plot correlation matrix of input features and DCTR weight
        df = pd.DataFrame(input_features)
        df["w_dctr"] = weight_ttbb[mask]
        plot_correlation_matrix(df, title, os.path.join(plot_dir, "correlation_matrix"), suffix="w_dctr")

        # Plot correlation matrix of input features and SPANet output
        df = pd.DataFrame(input_features)
        df["tthbb_transformed"] = events[mask].spanet_output.tthbb_transformed
        df["ttlf"] = events[mask].spanet_output.ttlf
        df["w_dctr"] = weight_ttbb[mask]
        plot_correlation_matrix(df, title, os.path.join(plot_dir, "correlation_matrix"), suffix="spanet_output")"""

    # Plot DCTR classifier score for ttbb and data on the same plot
    mask_train = ak.local_index(ak.num(events.JetGood)) < X_train.shape[0]
    plot_classifier_score(events, mask_data, mask_ttbb, mask_train, os.path.join(plot_dir, "classifier"))
    plot_dctr_weight(events, mask_train, os.path.join(plot_dir, "classifier"))
