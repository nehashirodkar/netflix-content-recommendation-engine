"""Neural Collaborative Filtering — embeddings + MLP, trained with negative sampling (BCE)."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src import config


class NCFTrainDataset(Dataset):
    """Each positive interaction is paired with K negatives sampled fresh per epoch."""
    def __init__(self, train_df, n_items: int, neg_samples: int, user_pos_items: dict[int, set[int]]):
        self.users = train_df["user_idx"].to_numpy(dtype=np.int64)
        self.pos_items = train_df["item_idx"].to_numpy(dtype=np.int64)
        self.n_items = n_items
        self.neg_samples = neg_samples
        self.user_pos_items = user_pos_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos = self.pos_items[idx]
        negs = []
        seen = self.user_pos_items.get(int(u), set())
        while len(negs) < self.neg_samples:
            n = np.random.randint(0, self.n_items)
            if n not in seen:
                negs.append(n)
        items = np.array([pos] + negs, dtype=np.int64)
        labels = np.array([1.0] + [0.0] * self.neg_samples, dtype=np.float32)
        users = np.array([u] * (1 + self.neg_samples), dtype=np.int64)
        return users, items, labels


def collate_ncf(batch):
    users = np.concatenate([b[0] for b in batch])
    items = np.concatenate([b[1] for b in batch])
    labels = np.concatenate([b[2] for b in batch])
    return (
        torch.from_numpy(users),
        torch.from_numpy(items),
        torch.from_numpy(labels),
    )


class NCFModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 32, mlp_layers=(64, 32, 16), dropout: float = 0.2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.mlp_layers = list(mlp_layers)
        self.dropout = dropout

        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        layers = []
        prev = 2 * embedding_dim
        for h in mlp_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        x = torch.cat([u, i], dim=-1)
        return self.mlp(x).squeeze(-1)  # logits

    @torch.no_grad()
    def score_all_items(self, user_idx: int, batch_size: int = 4096) -> np.ndarray:
        device = next(self.parameters()).device
        u = torch.tensor([user_idx], device=device)
        u_emb = self.user_emb(u).expand(self.n_items, -1)
        all_items = self.item_emb.weight
        scores = []
        for start in range(0, self.n_items, batch_size):
            end = min(start + batch_size, self.n_items)
            x = torch.cat([u_emb[start:end], all_items[start:end]], dim=-1)
            scores.append(self.mlp(x).squeeze(-1))
        scores = torch.cat(scores)
        return torch.sigmoid(scores).cpu().numpy()

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "n_users": self.n_users,
                "n_items": self.n_items,
                "embedding_dim": self.embedding_dim,
                "mlp_layers": self.mlp_layers,
                "dropout": self.dropout,
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
            mlp_layers=ckpt["mlp_layers"],
            dropout=ckpt["dropout"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


def build_user_pos_items(train_df) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for u, i in zip(train_df["user_idx"].to_numpy(), train_df["item_idx"].to_numpy()):
        out.setdefault(int(u), set()).add(int(i))
    return out


def train_ncf(train_df, val_df, n_users, n_items, device="cpu", wandb_run=None):
    user_pos_items = build_user_pos_items(train_df)
    dataset = NCFTrainDataset(train_df, n_items, config.NCF_NEG_SAMPLES, user_pos_items)
    loader = DataLoader(
        dataset,
        batch_size=config.NCF_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_ncf,
        pin_memory=(device != "cpu"),
    )

    model = NCFModel(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=config.NCF_EMBEDDING_DIM,
        mlp_layers=config.NCF_MLP_LAYERS,
        dropout=config.NCF_DROPOUT,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.NCF_LR, weight_decay=config.NCF_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    val_users = val_df["user_idx"].to_numpy(dtype=np.int64)
    val_items = val_df["item_idx"].to_numpy(dtype=np.int64)

    for epoch in range(config.NCF_EPOCHS):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for u, i, y in loader:
            u = u.to(device); i = i.to(device); y = y.to(device)
            optim.zero_grad()
            logits = model(u, i)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(y)
            n_seen += len(y)
        train_loss = total_loss / n_seen

        model.eval()
        with torch.no_grad():
            u = torch.from_numpy(val_users).to(device)
            i = torch.from_numpy(val_items).to(device)
            val_scores = torch.sigmoid(model(u, i)).cpu().numpy()
            val_mean_score = float(val_scores.mean())
        print(f"[NCF] epoch {epoch+1}/{config.NCF_EPOCHS} train_bce={train_loss:.4f} val_pos_score={val_mean_score:.4f}")
        if wandb_run is not None:
            wandb_run.log({"ncf/epoch": epoch + 1, "ncf/train_bce": train_loss, "ncf/val_pos_score": val_mean_score})

    return model
