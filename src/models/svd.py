"""PyTorch matrix factorization (SVD-style) with biases."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src import config


class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = df["user_idx"].to_numpy(dtype=np.int64)
        self.items = df["item_idx"].to_numpy(dtype=np.int64)
        self.ratings = df["rating"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class SVDModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 50, global_mean: float = 3.5):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.global_mean = float(global_mean)

        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        dot = (u * i).sum(dim=-1)
        bu = self.user_bias(user_idx).squeeze(-1)
        bi = self.item_bias(item_idx).squeeze(-1)
        return self.global_mean + bu + bi + dot

    @torch.no_grad()
    def predict(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        u = torch.tensor([user_idx], device=device)
        i = torch.tensor(item_indices, dtype=torch.long, device=device)
        u_emb = self.user_emb(u)
        i_emb = self.item_emb(i)
        scores = (u_emb * i_emb).sum(dim=-1)
        scores = scores + self.user_bias(u).squeeze(-1) + self.item_bias(i).squeeze(-1) + self.global_mean
        return scores.cpu().numpy()

    @torch.no_grad()
    def score_all_items(self, user_idx: int) -> np.ndarray:
        device = next(self.parameters()).device
        u = torch.tensor([user_idx], device=device)
        u_emb = self.user_emb(u)
        all_items = self.item_emb.weight
        scores = (u_emb * all_items).sum(dim=-1)
        scores = scores + self.user_bias(u).squeeze(-1) + self.item_bias.weight.squeeze(-1) + self.global_mean
        return scores.cpu().numpy()

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "n_users": self.n_users,
                "n_items": self.n_items,
                "embedding_dim": self.embedding_dim,
                "global_mean": self.global_mean,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        model = cls(
            n_users=ckpt["n_users"],
            n_items=ckpt["n_items"],
            embedding_dim=ckpt["embedding_dim"],
            global_mean=ckpt["global_mean"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


def train_svd(train_df, val_df, n_users, n_items, device="cpu", wandb_run=None):
    global_mean = float(train_df["rating"].mean())
    model = SVDModel(n_users, n_items, embedding_dim=config.SVD_EMBEDDING_DIM, global_mean=global_mean).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.SVD_LR, weight_decay=config.SVD_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        RatingsDataset(train_df),
        batch_size=config.SVD_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        RatingsDataset(val_df),
        batch_size=config.SVD_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    for epoch in range(config.SVD_EPOCHS):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for u, i, r in train_loader:
            u = u.to(device); i = i.to(device); r = r.to(device)
            optim.zero_grad()
            pred = model(u, i)
            loss = loss_fn(pred, r)
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(r)
            n_seen += len(r)
        train_rmse = (total_loss / n_seen) ** 0.5

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for u, i, r in val_loader:
                u = u.to(device); i = i.to(device); r = r.to(device)
                pred = model(u, i)
                val_loss += ((pred - r) ** 2).sum().item()
                n_val += len(r)
        val_rmse = (val_loss / n_val) ** 0.5
        print(f"[SVD] epoch {epoch+1}/{config.SVD_EPOCHS} train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f}")
        if wandb_run is not None:
            wandb_run.log({"svd/epoch": epoch + 1, "svd/train_rmse": train_rmse, "svd/val_rmse": val_rmse})

    return model
