"""PyTorch RNN/LSTM models with MPS backend."""

from __future__ import annotations

import gc
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, EPOCHS, PATIENCE, RANDOM_SEED


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _early_stop_best(losses: List[float], patience: int) -> bool:
    if len(losses) <= patience:
        return False
    best = min(losses[: -patience])
    return min(losses[-patience:]) >= best


class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(1, 256, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.rnn2 = nn.RNN(256, 128, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.rnn1(x)
        o = self.drop1(o)
        o, _ = self.rnn2(o)
        o = self.drop2(o[:, -1, :])
        return self.fc(o).squeeze(-1)


class LSTM_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.LSTM(1, 256, batch_first=True)
        self.d1 = nn.Dropout(0.2)
        self.l2 = nn.LSTM(256, 128, batch_first=True)
        self.d2 = nn.Dropout(0.2)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.l1(x)
        o = self.d1(o)
        o, _ = self.l2(o)
        o = self.d2(o[:, -1, :])
        return self.fc(o).squeeze(-1)


class LSTM_B(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.LSTM(1, 256, batch_first=True)
        self.d1 = nn.Dropout(0.3)
        self.l2 = nn.LSTM(256, 256, batch_first=True)
        self.d2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.l1(x)
        o = self.d1(o)
        o, _ = self.l2(o)
        o = self.d2(o[:, -1, :])
        o = torch.relu(self.fc1(o))
        return self.fc2(o).squeeze(-1)


class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.LSTM(1, 128, batch_first=True, bidirectional=True)
        self.d1 = nn.Dropout(0.2)
        self.l2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.l1(x)
        o = self.d1(o)
        o, _ = self.l2(o)
        o = o[:, -1, :]
        return self.fc(o).squeeze(-1)


def make_sequences(series: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """series: scaled close prices. Predict next step (scaled)."""
    x, y = [], []
    for i in range(len(series) - lookback):
        x.append(series[i : i + lookback])
        y.append(series[i + lookback])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def train_dl_model(
    model_name: str,
    train_scaled: np.ndarray,
    full_scaled: np.ndarray,
    train_len: int,
    lookback: int,
    device: torch.device,
) -> Tuple[np.ndarray, nn.Module]:
    """
    train_scaled: scaled train close only (1D).
    full_scaled: scaled concatenation of train + test closes (1D), same scaler as train.
    train_len: length of train segment in full_scaled.
    Returns test predictions (scaled space), length = len(full_scaled) - train_len.
    """
    torch.manual_seed(RANDOM_SEED)
    if device.type == "mps":
        torch.mps.manual_seed(RANDOM_SEED)

    X_tr, y_tr = make_sequences(train_scaled, lookback)
    if len(X_tr) < 64:
        return np.array([]), None

    X_tr = X_tr.reshape(-1, lookback, 1)
    ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    if model_name == "RNN":
        model = RNNModel().to(device)
    elif model_name == "LSTM_A":
        model = LSTM_A().to(device)
    elif model_name == "LSTM_B":
        model = LSTM_B().to(device)
    elif model_name == "BiLSTM":
        model = BiLSTMModel().to(device)
    else:
        raise ValueError(model_name)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    losses: List[float] = []

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * len(xb)
            n += len(xb)
        losses.append(ep_loss / max(n, 1))
        if _early_stop_best(losses, PATIENCE):
            break

    model.eval()
    full = full_scaled.astype(np.float32)
    test_len = len(full) - train_len
    preds = []
    window = list(full[:train_len])
    with torch.no_grad():
        for _ in range(test_len):
            seq = np.array(window[-lookback:], dtype=np.float32).reshape(1, lookback, 1)
            xt = torch.from_numpy(seq).to(device)
            p = float(model(xt).cpu().numpy().ravel()[0])
            preds.append(p)
            window.append(p)
    return np.array(preds, dtype=np.float32), model


def cleanup_dl(model: Optional[nn.Module], device: torch.device) -> None:
    if model is None:
        return
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
