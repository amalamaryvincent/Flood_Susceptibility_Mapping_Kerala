#!/usr/bin/env python
from __future__ import annotations
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fsm_autml.datasets.patch_dataset import PatchDataset
from fsm_autml.utils.repro import set_seed

class SimpleCNN(nn.Module):
    """A small baseline CNN for (B,H,W) inputs."""
    def __init__(self, in_ch: int = 12, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.fc(z)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/notebook_patches_3d.npz")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    d = np.load(args.data, allow_pickle=True)
    Xtr, ytr = d["X_train"], d["y_train"]
    Xte, yte = d["X_test"], d["y_test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN(in_ch=Xtr.shape[1]).to(device)

    train_dl = DataLoader(PatchDataset(Xtr, ytr), batch_size=args.batch, shuffle=True, num_workers=0)
    test_dl = DataLoader(PatchDataset(Xte, yte), batch_size=args.batch, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in tqdm(train_dl, desc=f"Epoch {ep}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)

        model.eval()
        correct = 0
        n = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += int((pred == yb).sum().item())
                n += int(yb.numel())
        print(f"Epoch {ep}: train_loss={total/len(train_dl.dataset):.4f} test_acc={correct/max(n,1):.4f}")

if __name__ == "__main__":
    main()
