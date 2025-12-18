from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """X: (N,B,H,W), y: (N,)"""
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
