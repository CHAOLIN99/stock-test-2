"""
PyTorch recurrent models for next-day log-return prediction (MPS backend).

Each sample is a window of the last `lookback` rows of the scaled feature
matrix; the target is that window's final-row y_reg (log(close[t+1]/close[t])).
Predictions are one-step-ahead from observed data, so they line up row-for-row
with the Tier-3 models.
"""

from __future__ import annotations

import copy
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, EPOCHS, PATIENCE, RANDOM_SEED


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class SeqSpec:
    cell: str = "lstm"  # "lstm" | "gru" | "rnn"
    hidden: Sequence[int] = (128, 64)
    dropout: float = 0.2
    bidirectional: bool = False
    head: Optional[int] = None  # extra dense layer before the output


PRESETS: Dict[str, SeqSpec] = {
    "RNN": SeqSpec(cell="rnn", hidden=(256, 128), dropout=0.2),
    "LSTM_A": SeqSpec(cell="lstm", hidden=(256, 128), dropout=0.2),
    "LSTM_B": SeqSpec(cell="lstm", hidden=(256, 256), dropout=0.3, head=32),
    "BiLSTM": SeqSpec(cell="lstm", hidden=(128, 64), dropout=0.2, bidirectional=True),
}

_CELLS = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}


class SeqRegressor(nn.Module):
    def __init__(self, n_features: int, spec: SeqSpec):
        super().__init__()
        dirs = 2 if spec.bidirectional else 1
        layers, size = [], n_features
        for h in spec.hidden:
            layers.append(_CELLS[spec.cell](size, h, batch_first=True, bidirectional=spec.bidirectional))
            size = h * dirs
        self.layers = nn.ModuleList(layers)
        self.drop = nn.Dropout(spec.dropout)
        self.head = nn.Sequential(nn.Linear(size, spec.head), nn.ReLU()) if spec.head else nn.Identity()
        self.out = nn.Linear(spec.head or size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x, _ = layer(x)
            x = self.drop(x)
        return self.out(self.head(x[:, -1, :])).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_windows(X: np.ndarray, lookback: int) -> np.ndarray:
    """(N, F) -> (N - lookback + 1, lookback, F); window k ends at row k + lookback - 1."""
    idx = np.arange(lookback)[None, :] + np.arange(len(X) - lookback + 1)[:, None]
    return X[idx].astype(np.float32)


@dataclass
class SeqResult:
    pred: np.ndarray  # length N; NaN for the first lookback-1 rows
    model: Optional[nn.Module]
    n_params: int
    best_epoch: int
    epochs_run: int
    history: Dict[str, List[float]] = field(default_factory=dict)


def _predict(model: nn.Module, W: np.ndarray, device: torch.device, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(W), batch):
            out.append(model(torch.from_numpy(W[i : i + batch]).to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def train_seq_model(
    spec: SeqSpec,
    X_all: np.ndarray,
    y_all: np.ndarray,
    n_train: int,
    lookback: int,
    device: torch.device,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = BATCH_SIZE,
    val_frac: float = 0.15,
) -> SeqResult:
    """
    X_all: scaled features for train rows followed by test rows (chronological).
    y_all: next-day log returns aligned with X_all; only y_all[:n_train] is used to fit.
    The last `val_frac` of training windows (chronological) drive early stopping.
    """
    torch.manual_seed(RANDOM_SEED)

    W = make_windows(X_all, lookback)
    targets = y_all[lookback - 1 :]
    n_train_w = n_train - lookback + 1
    n_val = max(int(n_train_w * val_frac), 1)
    n_fit = n_train_w - n_val
    if n_fit < 64:
        return SeqResult(np.full(len(X_all), np.nan), None, 0, 0, 0)

    mu, sd = float(targets[:n_fit].mean()), float(targets[:n_fit].std() + 1e-12)
    t_scaled = ((targets - mu) / sd).astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(W[:n_fit]), torch.from_numpy(t_scaled[:n_fit])),
        batch_size=batch_size,
        shuffle=True,
    )
    W_val = torch.from_numpy(W[n_fit:n_train_w]).to(device)
    t_val = torch.from_numpy(t_scaled[n_fit:n_train_w]).to(device)

    model = SeqRegressor(X_all.shape[1], spec).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(W_val), t_val).item())
        history["train_loss"].append(total / max(n, 1))
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    pred = np.full(len(X_all), np.nan)
    pred[lookback - 1 :] = _predict(model, W, device) * sd + mu
    return SeqResult(pred, model, count_params(model), best_epoch + 1, len(history["val_loss"]), history)


def cleanup_dl(model: Optional[nn.Module], device: torch.device) -> None:
    if model is None:
        return
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
