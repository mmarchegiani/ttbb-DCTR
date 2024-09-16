import numpy as np
import awkward as ak
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

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

def get_dataloader(X_train, Y_train, W, batch_size=2048, shuffle=False, num_workers=4):
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train, W),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
