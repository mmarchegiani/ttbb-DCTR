import os
import numpy as np
import awkward as ak
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import correctionlib

from ttbb_dctr.utils.utils import get_device

def get_datasets_list(cfg):
    folders = cfg["folders"]
    datasets = []
    for folder in folders:
        print(f"Processing folder {folder}")
        exclude_samples = ["ttHTobb_ttToSemiLep", "TTbbSemiLeptonic_4f_tt+LF", "TTbbSemiLeptonic_4f_tt+C", "TTToSemiLeptonic_tt+B"]
        datasets_in_folder = list(filter(lambda x : x.endswith(".parquet"), os.listdir(folder)))
        datasets_in_folder = [dataset for dataset in datasets_in_folder if not any(s in dataset for s in cfg["exclude_samples"])]
        datasets_in_folder = [f"{folder}/{dataset}" for dataset in datasets_in_folder]
        datasets += datasets_in_folder
    return datasets

def get_cr_mask(events, params):
    tthbb_transformed_min = params.get("tthbb_transformed_min", 0.0)
    tthbb_transformed_max = params.get("tthbb_transformed_max", 1.1)
    ttlf_max = params.get("ttlf_max", 1.1)
    mask = (
        (events.spanet_output.tthbb_transformed >= tthbb_transformed_min) &
        (events.spanet_output.tthbb_transformed < tthbb_transformed_max) &
        (events.spanet_output.ttlf < ttlf_max)
    )
    return mask

def get_njet_reweighting_map(events, mask_num, mask_den):
    reweighting_map_njet = {}
    njet = ak.num(events.JetGood)
    w = events.event.weight
    for nj in range(4,7):
        mask_nj = (njet == nj)
        reweighting_map_njet[nj] = sum(w[mask_num & mask_nj]) / sum(w[mask_den & mask_nj])
    for nj in range(7,21):
        reweighting_map_njet[nj] = sum(w[mask_num & (njet >= 7)]) / sum(w[mask_den & (njet >= 7)])
    return reweighting_map_njet

def get_njet_reweighting(events, reweighting_map_njet, mask=None):
    njet = ak.num(events.JetGood)
    w = events.event.weight
    w_nj = np.ones(len(events))
    if mask is None:
        mask = np.ones(len(events), dtype=bool)
    for nj in range(4,7):
        mask_nj = (njet == nj)
        w_nj = np.where(mask & mask_nj, reweighting_map_njet[nj], w_nj)
    for nj in range(7,21):
        w_nj = np.where(mask & (njet >= 7), reweighting_map_njet[nj], w_nj)
    print("1D reweighting map based on the number of jets:")
    print(reweighting_map_njet)
    return w_nj

def get_ttlf_reweighting(events, cfg, mask=None):
    '''Function to compute the ttlf reweighting based on the number of jets and the HT of the event.
    Only the tt+LF events are reweighted, the other events are left unchanged.'''
    njet = ak.num(events.JetGood)
    jetsHt = events.events.ht
    year = events.metadata.year
    mask_ttlf = events.ttlf == 1
    w = np.ones(len(events))
    if mask is None:
        mask = np.ones(len(events), dtype=bool)
    # Apply a different correction based on the year of the tt+LF sample
    for _year in cfg.keys():
        cset = correctionlib.CorrectionSet.from_file(
            cfg[_year]["file"]
        )
        corr = cset[cfg[_year]["key"]]
        w = np.where(year == _year, corr.evaluate(ak.to_numpy(njet), ak.to_numpy(jetsHt)), w)
    return ak.where(mask & mask_ttlf, w, 1.0)

def get_input_features(events, mask=None, only=None):
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
    if only is not None:
        input_features = {k : v for k, v in input_features.items() if k in only}
    if mask is not None:
        for key in input_features.keys():
            input_features[key] = input_features[key][mask]

    return input_features

def _stack_arrays(input_features: list, dtype=np.float32, normalize=True):
    assert len(input_features) > 1
    np_arrays = [np.asarray(ak.unflatten(x, 1), dtype=dtype) for x in input_features]
    X_np = np.hstack(np_arrays)
    if normalize:
        scaler = StandardScaler()
        scaler.fit(X_np)
        X_np = scaler.transform(X_np)
    return torch.from_numpy(X_np)

def get_tensors(input_features, labels, weights, dtype=np.float32, normalize_inputs=True, normalize_weights=True):
    device = get_device()

    if type(input_features) is dict:
        input_features = list(input_features.values())
    elif type(input_features) is list:
        pass
    else:
        raise ValueError("input_features must be either a dictionary or a list")
    X = _stack_arrays(input_features, dtype=dtype, normalize=normalize_inputs)
    Y = torch.tensor(labels, dtype=torch.long)
    W = torch.Tensor(weights)

    # Move inputs, weights and labels to GPU, if available
    if (device.type == "cuda") | (device == "cuda"):
        X, Y, W = X.to(device), Y.to(device), W.to(device)

    if normalize_weights:
        W = W / W.mean()

    return X, Y, W

def get_dataloader(X, Y, W, batch_size=2048, shuffle=True, num_workers=0):
    dataloader = DataLoader(
        TensorDataset(X, Y, W),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
