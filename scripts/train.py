import os
import argparse
import numpy as np
import awkward as ak

from omegaconf import OmegaConf

from tthbb_spanet.lib.dataset.h5 import Dataset
from ttbb_dctr.lib.quantile_transformer import WeightedQuantileTransformer

def get_datasets_list(cfg):
    folders = cfg["input"]["folders"]
    datasets = []
    for folder in folders:
        print(f"Processing folder {folder}")
        exclude_samples = ["ttHTobb_ttToSemiLep", "TTbbSemiLeptonic_4f_tt+LF", "TTbbSemiLeptonic_4f_tt+C", "TTToSemiLeptonic_tt+B"]
        datasets_in_folder = list(filter(lambda x : x.endswith(".parquet"), os.listdir(folder)))
        datasets_in_folder = [dataset for dataset in datasets_in_folder if not any(s in dataset for s in cfg["input"]["exclude_samples"])]
        datasets_in_folder = [f"{folder}/{dataset}" for dataset in datasets_in_folder]
        datasets += datasets_in_folder
    return datasets

def get_events(dataset, shuffle=True, seed=None):
    events = dataset.train
    i_permutation = np.random.RandomState(seed=seed).permutation(len(events))
    events = events[i_permutation]
    mask_btag = ak.values_astype(events.JetGood.btag_M, bool)
    events = ak.with_field(events, events.JetGood[mask_btag], "BJetGood")
    transformer = WeightedQuantileTransformer(n_quantiles=100000, output_distribution='uniform')
    mask_tthbb = events.tthbb == 1
    X = events.spanet_output.tthbb[mask_tthbb]
    transformer.fit(X, sample_weight=-events.event.weight[mask_tthbb]) # Fit quantile transformer on ttHbb sample only (- in front of weights due to negative weights in DCTR samples)
    transformed_score = transformer.transform(events.spanet_output.tthbb)
    events["spanet_output"] = ak.with_field(events.spanet_output, transformed_score, "tthbb_transformed")
    return events

def get_cr_mask(events, params):
    mask = (
        (events.spanet_output.tthbb_transformed < params["tthbb_transformed_max"]) &
        (events.spanet_output.ttlf < params["ttlf_max"])
    )
    return mask

def get_njet_reweighting(events, mask_num, mask_den):
    reweighting_map_njet = {}
    njet = ak.num(events.JetGood)
    w = events.event.weight
    w_nj = np.ones(len(events_den))
    for nj in range(4,7):
        mask_nj = (njet == nj)
        reweighting_map_njet[nj] = sum(w[mask_num & mask_nj]) / sum(w[mask_den & mask_nj])
        w_nj = np.where(mask_nj & mask_ttbb, reweighting_map_njet[nj], w_nj)
    reweighting_map_njet[7] = sum(w[mask_num & (njet >= 7)]) / sum(w[mask_den & (njet >= 7)])
    w_nj = np.where((njet >= 7) & mask_ttbb, reweighting_map_njet[7], w_nj)
    return w_nj

def get_input_features(events):
    input_features = {
        "njet" : ak.num(events.JetGood),
        "nbjet" : ak.num(events.BJetGood),
        "ht" : events.events.ht,
        "ht_b" : events.events.bjets_ht,
        "ht_light" : events.events.lightjets_ht,
        "drbb_avg" : events.events.drbb_avg,
        "mbb_max" : events.events.mbb_max,
        "mbb_min" : events.events.mbb_min,
        "mbb_closest" : events.events.mbb_closest,
        "drbb_min" : events.events.drbb_min,
        "detabb_min" : events.events.detabb_min,
        "dphibb_min" : events.events.dphibb_min,
        "jet_pt_1" : events.JetGood.pt[:,0],
        "jet_pt_2" : events.JetGood.pt[:,1],
        "jet_pt_3" : events.JetGood.pt[:,2],
        "jet_pt_4" : events.JetGood.pt[:,3],
        "bjet_pt_1" : events.BJetGood.pt[:,0],
        "bjet_pt_2" : events.BJetGood.pt[:,1],
        "bjet_pt_3" : events.BJetGood.pt[:,2],
        "jet_eta_1" : events.JetGood.eta[:,0],
        "jet_eta_2" : events.JetGood.eta[:,1],
        "jet_eta_3" : events.JetGood.eta[:,2],
        "jet_eta_4" : events.JetGood.eta[:,3],
        "bjet_eta_1" : events.BJetGood.eta[:,0],
        "bjet_eta_2" : events.BJetGood.eta[:,1],
        "bjet_eta_3" : events.BJetGood.eta[:,2],
    }

    return input_features

def get_tensors(input_features, labels, weights, dtype=np.float32, device="cuda", normalize=True, normalize_weights=True):
    if type(input_features) is dict:
        input_features = list(input_features.values())
    elif type(input_features) is list:
        pass
    else:
        raise ValueError("input_features must be either a dictionary or a list")
    X_train = _stack_arrays(input_features, dtype=dtype, normalize=normalize)
    Y_train = torch.tensor(labels, dtype=torch.long)
    W = torch.Tensor(weights)

    # Move inputs, weights and labels to GPU, if available
    if (device.type == "cuda") | (device == "cuda"):
        X_train, Y_train, W = X_train.to(device), Y_train.to(device), W.to(device)

    if normalize_weights:
        W = W / W.mean()

    return X_train, Y_train, W

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help="Config file with parameters for training", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output folder", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    os.makedirs(args.output, exist_ok=True)
    datasets = get_datasets_list(cfg)
    dataset_training = Dataset(datasets, "test.h5", cfg["preprocessing"], frac_train=1.0, shuffle=False, reweigh=True, has_data=True)
    print(dataset_training)
    events = get_events(dataset_training, shuffle=cfg["preprocessing"]["shuffle"], seed=cfg["preprocessing"]["seed"])
    mask_cr = get_cr_mask(events, cfg["cuts"]["cr1"])

    # Select events in control region
    events = events[mask_cr]

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
    X, Y, W = get_tensors(input_features, events.dctr, events.event.weight * w_nj, device=device)

    # Since the shuffling is already performed by permutation of the events, we don't shuffle the tensors here in order to maintain the same order in events and X_train, X_test
    X_train, X_test = train_test_split(X, test_size=cfg["preprocessing"]["test_size"], shuffle=False)
    Y_train, Y_test = train_test_split(Y, test_size=cfg["preprocessing"]["test_size"], shuffle=False)
    W_train, W_test = train_test_split(W, test_size=cfg["preprocessing"]["test_size"], shuffle=False)

    train_dataloader = get_dataloader(X_train, Y_train, W_train, batch_size=4096)
    val_dataloader = get_dataloader(X_test, Y_test, W_test, batch_size=4096)
