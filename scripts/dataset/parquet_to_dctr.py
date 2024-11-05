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
from ttbb_dctr.lib.data_preprocessing import get_datasets_list, get_cr_mask, get_njet_reweighting

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help="Config file with parameters for data preprocessing and training", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output file path", required=True)
    parser.add_argument('--reweigh-njet', action='store_true', help="Apply 1D reweighting based on the number of jets")
    parser.add_argument('--dry', action='store_true', help="Dry run, do not save the dataset")
    parser.add_argument('--cr2', action='store_true', help="Apply reweighting for CR2")
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
    if args.cr2:
        assert len(list(cfg_cuts.keys())) == 1
        key = list(cfg_cuts.keys())[0]
        with open("/eos/user/m/mmarcheg/ttHbb/dctr/training_datasets/full_Run2_26features_v3/dataset_dctr_26features_reweighting_map.yaml") as f:
            njet_reweighting_map = yaml.load(f, Loader=yaml.FullLoader)["tthbb0p40_ttlf0p60"]
        dataset.store_weight(key, get_njet_reweighting(dataset.df, njet_reweighting_map, dataset.masks[key]))
    if args.reweigh_njet:
        dataset.compute_njet_weights()
    if not args.dry:
        dataset.save_all(args.output)
        if args.reweigh_njet:
            dataset.save_reweighting_map(args.output.replace(".parquet", "_reweighting_map.yaml"))
