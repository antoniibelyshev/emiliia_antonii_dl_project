import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from abc import ABC, abstractmethod


class TorchDataset(Dataset):
    def __init__(self, X, y):
        super(TorchDataset, self).__init__()

        self.X = X
        self.y = y

        self.N = len(y)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AbstractDataset(ABC):
    def __init__(self, device, workdir, filename, *read_args, trunc_dim=35, valid_size=0.2, batch_size=20, **read_kwargs):
        self.device = device

        self.dataset = self.get_dataset(workdir, filename, *read_args, **read_kwargs)

        self.X = self.dataset.iloc[:, :5]
        self.Y = self.dataset.iloc[:, 5:]

        self.preprocess(trunc_dim)
        self.update_train_valid_split(valid_size)

        self.batch_size = batch_size

    def get_dataset(self, workdir, filename, *read_args, **read_kwargs):
        dataset = self.read(workdir, filename, *read_args, **read_kwargs)
        dataset = self.transform_to_canonical(dataset)
        return dataset

    @abstractmethod
    def read(self, workdor, filename, *read_args, **read_kwargs):
        return

    @abstractmethod
    def transform_to_canonical(self, dataset):
        return

    def get_dataloader(self, valid=False):
        batch_size = len(self.valid_idx) if valid else self.batch_size
        return DataLoader(TorchDataset(*self.get_tensor_data(self.device, valid)), batch_size=batch_size, shuffle=True)

    def get_tensor_data(self, device, valid=True):
        idx = self.valid_idx if valid else self.train_idx
        return torch.tensor(self.encoded_X.iloc[idx].values).to(device), torch.tensor(self.reduced_Y[idx], dtype=torch.float32).to(device)

    def update_train_valid_split(self, new_valid_size=None):
        if new_valid_size is not None:
            self.valid_size = new_valid_size
        self.create_train_valid_idx()
        
    def create_train_valid_idx(self):
        idx = np.arange(len(self.X))
        self.train_idx, self.valid_idx = train_test_split(idx, test_size=self.valid_size)

    def preprocess(self, new_trunc_dim=None):
        if new_trunc_dim is not None:
            self.trunc_dim = new_trunc_dim

        self.reducer = TruncatedSVD(n_components=self.trunc_dim, n_iter=10)
        self.reduced_Y = self.reducer.fit_transform(self.Y)

        encoder = LabelEncoder()
        columns = self.X.columns[[0, 1, 4]]
        self.encoded_X = self.X[columns].copy()
        self.nu = []

        for col in columns:
            self.encoded_X[col] = encoder.fit_transform(self.encoded_X[col])
            self.nu.append(len(self.encoded_X[col].unique()))

    def final_loss(self, valid_pred):
        return ((self.Y.iloc[self.valid_idx, :].values - self.reducer.inverse_transform(valid_pred)) ** 2).mean()
