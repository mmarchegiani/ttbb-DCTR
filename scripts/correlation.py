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
from ttbb_dctr.lib.data_preprocessing import get_device, get_datasets_list, get_tensors, get_input_features
from ttbb_dctr.lib.plotting import plot_correlation, plot_correlation_matrix, plot_classifier_score, plot_dctr_weight, plot_closure_test, plot_closure_test_split_by_weight, get_central_interval

def get_epoch(s):
    return int(s.split("-")[0].split("=")[-1])

def load_model(log_directory, epoch=None):
    assert "checkpoints" in os.listdir(os.path.join(log_directory)), "No checkpoints found in log directory"
    assert "hparams.yaml" in os.listdir(log_directory), "No hparams.yaml found in log directory"
    hparams = OmegaConf.load(os.path.join(log_directory, "hparams.yaml"))
    if epoch is None:
        checkpoint = os.path.join(log_directory, "checkpoints", "last.ckpt")
    else:
        checkpoints_by_epoch = [c for c in os.listdir(os.path.join(log_directory, "checkpoints")) if not "last.ckpt" in c]
        checkpoints = list(filter(lambda x : get_epoch(x) == epoch, checkpoints_by_epoch))
        assert len(checkpoints) == 1, f"Found {len(checkpoints)} checkpoints for epoch {epoch}"
        checkpoint = os.path.join(log_directory, "checkpoints", checkpoints[0])
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
    parser.add_argument('--plot_dir', type=str, default="plots", help="Output folder for plots", required=False)
    parser.add_argument('--epoch', type=int, default=None, help="Select the epoch to load the model from", required=False)
    parser.add_argument('-j', '--workers', type=int, default=8, help="Number of workers for parallel processing")
    args = parser.parse_args()

    if args.workers > multiprocessing.cpu_count():
        print(f"Number of workers ({args.workers}) is greater than number of CPUs ({multiprocessing.cpu_count()}). Setting number of workers to {multiprocessing.cpu_count()}")
        args.workers = multiprocessing.cpu_count()
    plot_dir = os.path.join(args.log_directory, args.plot_dir)

    device = get_device()

    if args.cfg is None:
        cfg_training = load_training_config(args.log_directory)
    else:
        cfg = OmegaConf.load(args.cfg)
        cfg_training = cfg["training"]

    # Load model and compute predictions to extract the DCTR weight
    dataset_train = DCTRDataset(cfg_training["training_file"], shuffle=False, reweigh=False, label=False)
    dataset_test = DCTRDataset(cfg_training["test_file"], shuffle=False, reweigh=False, label=False)
    events_train = dataset_train.df
    events_test = dataset_test.df
    if len(events_train) == 0:
        events = events_test
    elif len(events_test) == 0:
        events = events_train
    else:
        events = ak.concatenate((events_train, events_test))
    input_features = get_input_features(events, only=cfg_training.get("input_features", None))
    X_full, Y_full, W_full = get_tensors(input_features, events.dctr, events.event.weight, normalize_inputs=True, normalize_weights=True)

    # Compute DCTR score and weight
    model = load_model(args.log_directory, epoch=args.epoch)
    model.eval()
    with torch.no_grad():
        predictions = model(X_full)
    predictions = F.sigmoid(predictions)
    score = ak.Array(predictions[:,0].to("cpu").detach().numpy())
    weight_ttbb = score / (1-score)

    # Add DCTR score and weight branches to the events
    events = ak.with_field(events, score, "dctr_score")
    events = ak.with_field(events, weight_ttbb, "dctr_weight")

    # Select events in control region, without cut on tthbb
    mask_data = (events.data == 1)
    mask_mc = (events.data == 0)
    mask_ttbb = mask_mc & (events.ttbb == 1)
    mask_ttcc = mask_mc & (events.ttcc == 1)
    mask_ttlf = mask_mc & (events.ttlf == 1)
    mask_data_minus_minor_bkg = mask_data | (mask_mc & ~mask_ttbb)

    os.makedirs(plot_dir, exist_ok=True)

    correlations_spanet_dir = os.path.join(plot_dir, "correlations_spanet")
    correlation_matrix_dir = os.path.join(plot_dir, "correlation_matrix")
    for subdir in [correlations_spanet_dir, correlation_matrix_dir]:
        os.makedirs(subdir, exist_ok=True)
    print("Plotting correlations")
    for mask, title in zip([mask_data, mask_mc, mask_ttbb, mask_ttcc, mask_ttlf],
                        ["Data", "MC", "ttbb", "ttcc", "ttlf"]):
        input_features_masked = get_input_features(events, mask)
        for score in ["tthbb_transformed", "ttlf"]:
            y = events[mask].spanet_output[score]
            w = events[mask].event.weight
            def f(varname_x):
                return plot_correlation(input_features_masked[varname_x], y, w, varname_x, title, score, os.path.join(plot_dir, "correlations_spanet"))
            if args.workers == 1:
                for varname_x in input_features_masked.keys():
                    f(varname_x)
            else:
                with Pool(processes=args.workers) as pool:
                    pool.map(f, list(input_features_masked.keys()))
                    pool.close()
                    pool.join()

        # Plot correlation matrix of input features and DCTR weight
        df = pd.DataFrame(input_features_masked)
        df["w_dctr"] = weight_ttbb[mask]
        plot_correlation_matrix(df, title, os.path.join(plot_dir, "correlation_matrix"), suffix="w_dctr")

        # Plot correlation matrix of input features and SPANet output
        df = pd.DataFrame(input_features_masked)
        df["tthbb_transformed"] = events[mask].spanet_output.tthbb_transformed
        df["ttlf"] = events[mask].spanet_output.ttlf
        df["w_dctr"] = weight_ttbb[mask]
        plot_correlation_matrix(df, title, os.path.join(plot_dir, "correlation_matrix"), suffix="spanet_output")
