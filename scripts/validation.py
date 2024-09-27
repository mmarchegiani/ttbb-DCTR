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
    events = ak.concatenate((events_train, events_test))
    input_features_train = get_input_features(events_train)
    input_features_test = get_input_features(events_test)
    input_features = get_input_features(events)
    X_train, Y_train, W_train = get_tensors(input_features_train, events_train.dctr, events_train.event.weight, normalize_inputs=True, normalize_weights=True)
    X_test, Y_test, W_test = get_tensors(input_features_test, events_test.dctr, events_test.event.weight, normalize_inputs=True, normalize_weights=True)
    X_full = torch.concatenate((X_train, X_test))
    Y_full = torch.concatenate((Y_train, Y_test))
    W_full = torch.concatenate((W_train, W_test))

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
    mask_train = ak.local_index(ak.num(events.JetGood)) < X_train.shape[0]
    mask_test = ~mask_train
    events_train = ak.with_field(events[mask_train], score[mask_train], "dctr_score")
    events_train = ak.with_field(events_train, weight_ttbb[mask_train], "dctr_weight")
    events_test = ak.with_field(events[mask_test], score[mask_test], "dctr_score")
    events_test = ak.with_field(events_test, weight_ttbb[mask_test], "dctr_weight")

    # Select events in control region, without cut on tthbb
    mask_data = (events.data == 1)
    mask_mc = (events.data == 0)
    mask_ttbb = mask_mc & (events.ttbb == 1)
    mask_ttcc = mask_mc & (events.ttcc == 1)
    mask_ttlf = mask_mc & (events.ttlf == 1)
    mask_data_minus_minor_bkg = mask_data | (mask_mc & ~mask_ttbb)

    os.makedirs(plot_dir, exist_ok=True)

    classifier_dir = os.path.join(plot_dir, "classifier")
    closure_test_dir = os.path.join(plot_dir, "closure_test")
    for subdir in [classifier_dir, closure_test_dir]:
        os.makedirs(subdir, exist_ok=True)

    # Plot DCTR classifier score for ttbb and data on the same plot
    plot_classifier_score(events, mask_data_minus_minor_bkg, mask_ttbb, mask_train, os.path.join(plot_dir, "classifier"))
    plot_dctr_weight(events, mask_train, os.path.join(plot_dir, "classifier"))
    print("Plotting closure test")
    weight_cuts_default = [(0, 0.8), (0.8, 1.2), (1.2, 3)]
    perc = 0.40
    w_lo, w_hi = get_central_interval(weight_ttbb[mask_ttbb], perc=perc)
    weight_cuts_symmetric = [(0, w_lo), (w_lo, w_hi), (w_hi, 3)]
    weight_cuts_tails = [(0, 0.1), (0.1, 1.9), (1.9, 3)]
    weight_dict = {
        "0p8To1p2": weight_cuts_default,
        "symmetric": weight_cuts_symmetric,
        "0p1To1p9": weight_cuts_tails
    }
    d = {"weight_cuts": weight_dict, "central_percentile": perc}
    filename_json = os.path.join(plot_dir, "closure_test", "weight_cuts.json")
    print(f"Writing weight cuts to {filename_json}")
    with open(filename_json, "w") as f:
        json.dump(d, f)
    for weight_name, weight_cuts in weight_dict.items():
        for dataset_type, mask_dataset, _events in zip(["train", "test"], [mask_train, mask_test], [events_train, events_test]):
            plot_dir_dataset = os.path.join(plot_dir, "closure_test", dataset_type)
            plot_dir_dataset_inclusive = os.path.join(plot_dir_dataset, "inclusive")
            plot_dir_dataset_split_by_weight = os.path.join(plot_dir_dataset, "split_by_weight")
            for subdir in [plot_dir_dataset, plot_dir_dataset_inclusive, plot_dir_dataset_split_by_weight]:
                os.makedirs(subdir, exist_ok=True)
            def f(varname_x):
                return plot_closure_test(_events, mask_data_minus_minor_bkg[mask_dataset], mask_ttbb[mask_dataset], plot_dir_dataset_inclusive, only_var=varname_x)
            def g(varname_x):
                return plot_closure_test_split_by_weight(_events, mask_data_minus_minor_bkg[mask_dataset], mask_ttbb[mask_dataset], weight_cuts, plot_dir_dataset_split_by_weight, only_var=varname_x, suffix=weight_name)
            if args.workers == 1:
                plot_closure_test(_events, mask_data_minus_minor_bkg[mask_dataset], mask_ttbb[mask_dataset], plot_dir_dataset_inclusive)
                plot_closure_test_split_by_weight(_events, mask_data_minus_minor_bkg[mask_dataset], mask_ttbb[mask_dataset], weight_cuts, plot_dir_dataset_split_by_weight, suffix=weight_name)
            else:
                with Pool(processes=args.workers) as pool:
                    pool.map(f, list(input_features.keys()))
                    pool.map(g, list(input_features.keys()))
                    pool.close()
                    pool.join()
