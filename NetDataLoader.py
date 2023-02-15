import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


class NetDataset(Dataset):

    def __init__(self, model_name, isTrain=True):
        self.model_name = model_name
        self.isTrain = isTrain

        prefix = 'train' if isTrain else 'test'
        data_path = f"{prefix}_{model_name}_data.npy"
        label_path = f"{prefix}_{model_name}_labels.npy"

        self.data = np.load(data_path)
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.labels)

    def shape(self):
        return np.prod(self.data[1].shape)

    def __getitem__(self, index) -> T_co:
        return self.data[index], self.labels[index]
