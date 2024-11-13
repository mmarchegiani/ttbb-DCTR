import os
import yaml
import multiprocessing
from multiprocessing import Pool
from functools import partial
import argparse
import numpy as np
import pandas as pd
import awkward as ak
from omegaconf import OmegaConf

from tthbb_spanet import DCTRDataset
from ttbb_dctr.lib.data_preprocessing import get_datasets_list, get_cr_mask, get_njet_reweighting, get_ttlf_reweighting

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help="Config file with parameters for data preprocessing and training", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output file path", required=True)
    parser.add_argument('--compute-njet-reweighting', action='store_true', help="Apply 1D reweighting based on the number of jets")
    parser.add_argument('--dry', action='store_true', help="Dry run, do not save the dataset")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    cfg_input = cfg["input"]
    cfg_preprocessing = cfg["preprocessing"]
    cfg_cuts = cfg["cuts"]
    print("Cuts configuration:")
    print(cfg_cuts)
    datasets = get_datasets_list(cfg_input)
    dataset = DCTRDataset(datasets, cfg_preprocessing, shuffle=True, reweigh=True, has_data=True)
    dataset.store_masks({k : get_cr_mask(dataset.df, v) for k, v in cfg_cuts.items()})
    # Apply a pre-computed njet based reweighting if specified in the configuration file
    # This is needed to apply the same reweighting as applied to the training dataset, to the test dataset
    if cfg["preprocessing"].get("weights", None) is not None:
        cfg_weights = cfg["preprocessing"]["weights"]
    else:
        cfg_weights = {}
    # Store the njet reweighting factor
    if cfg_weights.get("njet_reweighting", None) is not None:
        cfg_njet_reweighting = cfg_weights["njet_reweighting"]
        with open(cfg_njet_reweighting["file"]) as f:
            njet_reweighting_map = yaml.load(f, Loader=yaml.FullLoader)[cfg_njet_reweighting["key"]]
        dataset.apply_weight(dataset.df, get_njet_reweighting(dataset.df, njet_reweighting_map))
    if not args.dry:
        if args.compute_njet_reweighting:
            if cfg_weights.get("njet_reweighting", None) is not None:
                raise ValueError("njet reweighting already computed and saved in the configuration file. Please remove the --compute-njet-reweighting flag.")
            # Apply the tt+LF reweighting before computing the njet reweighting
            if cfg_weights.get("ttlf_reweighting", None) is not None:
                cfg_ttlf_reweighting = cfg_weights["ttlf_reweighting"]
                dataset.apply_weight(dataset.df, get_ttlf_reweighting(dataset.df, cfg_ttlf_reweighting))
            dataset.compute_njet_weights()
            dataset.save_reweighting_map(args.output.replace(".parquet", "_reweighting_map.yaml"))
        else:
            dataset.save_all(args.output)
