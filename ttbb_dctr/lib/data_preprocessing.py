import os
import pickle
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

def fit_standard_scaler(input_features, dtype=np.float32):
    '''Fit standard scaler on the input features. The input features are expected to be a dictionary of awkward arrays.
    The function returns the fitted StandardScaler object that can be saved and used to transform the input features.
    '''
    if type(input_features) is dict:
        input_features = list(input_features.values())
    elif type(input_features) is list:
        pass
    else:
        raise ValueError("input_features must be either a dictionary or a list")
    assert len(input_features) > 1
    np_arrays = [np.asarray(ak.unflatten(x, 1), dtype=dtype) for x in input_features]
    X_np = np.hstack(np_arrays)
    standard_scaler = StandardScaler()
    standard_scaler.fit(X_np)

    return standard_scaler

def save_standard_scaler(scaler, filename):
    if os.path.exists(filename):
        filename = filename.replace(".pkl", "_v0.pkl")
        i = 0
        while os.path.exists(filename):
            i += 1
            filename = filename.replace(f"_v{i-1}.pkl", f"_v{i}.pkl")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        print(f"Saving standard scaler to {filename}")
        pickle.dump(scaler, f)

def load_standard_scaler(folder):
    # Take the first file in the folder as standard scaler:
    filename = os.path.join(folder, os.listdir(folder)[0])
    with open(filename, "rb") as f:
        print(f"Loading standard scaler from {filename}")
        scaler = pickle.load(f)
    return scaler

def _stack_arrays(input_features, dtype=np.float32, standard_scaler=None):
    '''Stack the input features into a single torch tensor. The input features are expected to be either a list or a dictionary of awkward arrays.
    The `standard_scaler` argument is an optional StandardScaler object that can be used to transform the input features.
    If `standard_scaler` is None, the input features are not transformed.
    '''
    if type(input_features) is dict:
        input_features = list(input_features.values())
    elif type(input_features) is list:
        pass
    else:
        raise ValueError("input_features must be either a dictionary or a list")
    assert len(input_features) > 1
    np_arrays = [np.asarray(ak.unflatten(x, 1), dtype=dtype) for x in input_features]
    X_np = np.hstack(np_arrays)
    if standard_scaler is not None:
        X_np = standard_scaler.transform(X_np)
    return torch.from_numpy(X_np)

def get_tensors(input_features, labels, weights, dtype=np.float32, standard_scaler=None, normalize_weights=True):
    '''Convert the input features, labels and weights to torch tensors. The input features are expected to be a dictionary of awkward arrays.
    The function returns the input features, labels and weights as torch tensors. The input features are stacked into a single tensor.
    The `standard_scaler` argument is an optional StandardScaler object that can be used to transform the input features.
    If `standard_scaler` is None, the input features are not transformed.
    If `normalize_weights` is True, the weights are normalized to have a mean of 1.
    '''
    device = get_device()

    X = _stack_arrays(input_features, dtype=dtype, standard_scaler=standard_scaler)
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
