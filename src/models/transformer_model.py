from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Callable, Dict, List, Optional

import numpy as np

from ..metrics import evaluate_predictions

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(self, input_dim: int, horizon: int, d_model: int = 96, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos = _PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, horizon))

    def forward(self, x):
        z = self.pos(self.in_proj(x))
        enc = self.encoder(z)
        pooled = enc.mean(dim=1)
        return self.head(pooled)


class TransformerPVModel:
    """Transformer Encoder model for PV day-ahead forecasting."""

    name = "Transformer"

    def __init__(
        self,
        input_dim: int,
        memory_length: int = 96,
        horizon: int = 96,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 10,
        patience: int = 3,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        if torch is None:
            raise ImportError("请先安装 PyTorch：pip install torch")
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = _TransformerNet(input_dim, horizon, d_model, nhead, num_layers, dropout, max_len=max(256, memory_length + 8)).to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = max(1, int(patience))
        self.history: List[Dict[str, float]] = []

    def _loader(self, X, y=None, shuffle=False):
        X_t = torch.tensor(X, dtype=torch.float32)
        if y is None:
            ds = TensorDataset(X_t)
        else:
            ds = TensorDataset(X_t, torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train, y_train, X_val=None, y_val=None, progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None):
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        train_loader = self._loader(X_train, y_train, shuffle=True)
        val_loader = self._loader(X_val, y_val, shuffle=False) if X_val is not None and y_val is not None else None
        best_metric = float("inf")
        best_state = None
        bad_epochs = 0
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            rec = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan")}
            if val_loader is not None:
                self.net.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        val_losses.append(loss_fn(self.net(xb), yb).item())
                rec["val_loss"] = float(np.mean(val_losses)) if val_losses else float("nan")
                monitor = rec["val_loss"]
            else:
                monitor = rec["train_loss"]
            self.history.append(rec)
            if progress_callback:
                progress_callback(epoch, self.epochs, rec)
            if np.isfinite(monitor) and monitor < best_metric - 1e-8:
                best_metric = monitor
                best_state = deepcopy(self.net.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
            if val_loader is not None and bad_epochs >= self.patience:
                rec["early_stopped"] = True
                if progress_callback:
                    progress_callback(epoch, self.epochs, {"message": f"early stop at epoch {epoch}"})
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return {"train_loss": self.history[-1]["train_loss"], "val_loss": self.history[-1].get("val_loss", float("nan"))}

    def predict(self, X):
        self.net.eval()
        preds = []
        with torch.no_grad():
            for batch in self._loader(X, None, shuffle=False):
                xb = batch[0].to(self.device)
                preds.append(self.net(xb).cpu().numpy())
        return np.clip(np.concatenate(preds, axis=0), 0.0, 1.0)

    def evaluate(self, X_test, y_test):
        return evaluate_predictions(y_test, self.predict(X_test))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "history": self.history}, path)
