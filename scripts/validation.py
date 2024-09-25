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
from ttbb_dctr.lib.plotting import plot_correlation, plot_correlation_matrix, plot_classifier_score, plot_dctr_weight, plot_closure_test, plot_closure_test_split_by_weight

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
    parser.add_argument('--plot_dir', type=str, default="plots", help="Output folder for plots", required=False)
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
    model = load_model(args.log_directory)
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

    correlations_spanet_dir = os.path.join(plot_dir, "correlations_spanet")
    correlation_matrix_dir = os.path.join(plot_dir, "correlation_matrix")
    classifier_dir = os.path.join(plot_dir, "classifier")
    closure_test_dir = os.path.join(plot_dir, "closure_test")
    for subdir in [correlations_spanet_dir, correlation_matrix_dir, classifier_dir, closure_test_dir]:
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

    # Plot DCTR classifier score for ttbb and data on the same plot
    plot_classifier_score(events, mask_data_minus_minor_bkg, mask_ttbb, mask_train, os.path.join(plot_dir, "classifier"))
    plot_dctr_weight(events, mask_train, os.path.join(plot_dir, "classifier"))
    print("Plotting closure test")
    for dataset_type, mask_dataset, _events in zip(["train", "test"], [mask_train, mask_test], [events_train, events_test]):
        plot_dir_dataset = os.path.join(plot_dir, "closure_test", dataset_type)
        plot_dir_dataset_inclusive = os.path.join(plot_dir_dataset, "inclusive")
        plot_dir_dataset_split_by_weight = os.path.join(plot_dir_dataset, "split_by_weight")
        for subdir in [plot_dir_dataset, plot_dir_dataset_inclusive, plot_dir_dataset_split_by_weight]:
            os.makedirs(subdir, exist_ok=True)
        weight_cuts = [(0, 0.8), (0.8, 1.2), (1.2, 3)]
        def f(varname_x):
            return plot_closure_test(_events, mask_data_minus_minor_bkg[mask_dataset], mask_ttbb[mask_dataset], plot_dir_dataset_inclusive, only_var=varname_x)
        def g(varname_x):
            return plot_closure_test_split_by_weight(_events, mask_data_minus_minor_bkg[mask_dataset], mask_ttbb[mask_dataset], weight_cuts, plot_dir_dataset_split_by_weight, only_var=varname_x)
        if args.workers == 1:
            plot_closure_test(_events, mask_data_minus_minor_bkg[mask_dataset], mask_ttbb[mask_dataset], plot_dir_dataset_inclusive)
            plot_closure_test_split_by_weight(_events, mask_data_minus_minor_bkg[mask_dataset], mask_ttbb[mask_dataset], weight_cuts, plot_dir_dataset_split_by_weight)
        else:
            with Pool(processes=args.workers) as pool:
                pool.map(f, list(input_features.keys()))
                pool.map(g, list(input_features.keys()))
                pool.close()
                pool.join()
