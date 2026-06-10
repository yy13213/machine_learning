from __future__ import annotations

from copy import deepcopy
from pathlib import Path
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


class _RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        x = (x - mean) / stdev
        if self.affine:
            x = x * self.affine_weight.view(1, 1, -1)
            x = x + self.affine_bias.view(1, 1, -1)
        return x


class _SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho: {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = ((torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p))
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2) if norms else torch.tensor(0.0, device=shared_device)


class _SAMFormerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        memory_length: int,
        horizon: int,
        hid_dim: int = 16,
        dropout: float = 0.1,
        use_revin: bool = True,
    ):
        super().__init__()
        self.use_revin = use_revin
        self.revin = _RevIN(num_features=input_dim) if use_revin else nn.Identity()
        self.compute_keys = nn.Linear(memory_length, hid_dim)
        self.compute_queries = nn.Linear(memory_length, hid_dim)
        self.compute_values = nn.Linear(memory_length, memory_length)
        self.dropout = nn.Dropout(dropout)
        self.channel_forecaster = nn.Linear(memory_length, horizon)
        self.channel_projector = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F] -> [B, F, L]
        z = x.transpose(1, 2)
        if self.use_revin:
            z = self.revin(z.transpose(1, 2)).transpose(1, 2)
        queries = self.compute_queries(z)
        keys = self.compute_keys(z)
        values = self.compute_values(z)
        scale = queries.shape[-1] ** -0.5
        attn_scores = torch.matmul(queries, keys.transpose(1, 2)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, values)
        fused = z + self.dropout(attn_out)
        per_channel = self.channel_forecaster(fused)  # [B, F, H]
        y = self.channel_projector(per_channel.transpose(1, 2)).squeeze(-1)  # [B, H]
        return y


class SAMFormerPVModel:
    name = "SAMFormer"

    def __init__(
        self,
        input_dim: int,
        memory_length: int,
        horizon: int = 96,
        hid_dim: int = 16,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        rho: float = 0.5,
        batch_size: int = 64,
        epochs: int = 12,
        patience: int = 3,
        use_revin: bool = True,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        if torch is None:
            raise ImportError("请先安装 PyTorch：pip install torch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = _SAMFormerNet(
            input_dim=input_dim,
            memory_length=memory_length,
            horizon=horizon,
            hid_dim=hid_dim,
            dropout=dropout,
            use_revin=use_revin,
        ).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.rho = rho
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
        loss_fn = nn.MSELoss()
        optimizer = _SAM(
            self.net.parameters(),
            base_optimizer=torch.optim.Adam,
            rho=self.rho,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
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
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                pred_second = self.net(xb)
                second_loss = loss_fn(pred_second, yb)
                second_loss.backward()
                optimizer.second_step(zero_grad=True)
                losses.append(second_loss.item())

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
