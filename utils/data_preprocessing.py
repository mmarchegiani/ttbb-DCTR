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

def get_tensors(input_features : list, labels, weights, dtype=np.float32, device="cuda", normalize=True):
    X_train = _stack_arrays(input_features, dtype=dtype, normalize=normalize)
    Y_train = torch.tensor(labels, dtype=torch.long)
    W = torch.Tensor(weights)

    # Move inputs, weights and labels to GPU, if available
    if (device.type == "cuda") | (device == "cuda"):
        X_train, Y_train, W = X_train.to(device), Y_train.to(device), W.to(device)
        
    return X_train, Y_train, W

def get_dataloader(X_train: torch.Tensor, Y_train: torch.Tensor, W: torch.Tensor, batch_size=2048, shuffle=True):
    dataloader = DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train, W),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataloader
