import numpy as np
import awkward as ak
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from ttbb_dctr.lib.quantile_transformer import WeightedQuantileTransformer

def get_datasets_list(cfg):
    folders = cfg["folders"]
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
    w_nj = np.ones(len(events))
    for nj in range(4,7):
        mask_nj = (njet == nj)
        reweighting_map_njet[nj] = sum(w[mask_num & mask_nj]) / sum(w[mask_den & mask_nj])
        w_nj = np.where(mask_den & mask_nj, reweighting_map_njet[nj], w_nj)
    reweighting_map_njet[7] = sum(w[mask_num & (njet >= 7)]) / sum(w[mask_den & (njet >= 7)])
    w_nj = np.where(mask_den & (njet >= 7), reweighting_map_njet[7], w_nj)
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

def _stack_arrays(input_features: list, dtype=np.float32, normalize=True):
    assert len(input_features) > 1
    np_arrays = [np.asarray(ak.unflatten(x, 1), dtype=dtype) for x in input_features]
    X_np = np.hstack(np_arrays)
    if normalize:
        scaler = StandardScaler()
        scaler.fit(X_np)
        X_np = scaler.transform(X_np)
    return torch.from_numpy(X_np)

def get_tensors(input_features, labels, weights, dtype=np.float32, device="cuda", normalize_inputs=True, normalize_weights=True):
    if type(input_features) is dict:
        input_features = list(input_features.values())
    elif type(input_features) is list:
        pass
    else:
        raise ValueError("input_features must be either a dictionary or a list")
    X_train = _stack_arrays(input_features, dtype=dtype, normalize=normalize_inputs)
    Y_train = torch.tensor(labels, dtype=torch.long)
    W = torch.Tensor(weights)

    # Move inputs, weights and labels to GPU, if available
    if (device.type == "cuda") | (device == "cuda"):
        X_train, Y_train, W = X_train.to(device), Y_train.to(device), W.to(device)

    if normalize_weights:
        W = W / W.mean()

    return X_train, Y_train, W

def get_dataloader(X_train, Y_train, W, batch_size=2048, shuffle=True, num_workers=4):
    dataloader = DataLoader(
        TensorDataset(X_train, Y_train, W),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
