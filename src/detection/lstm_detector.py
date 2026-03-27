from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.logger import get_logger

log = get_logger(__name__)


class DriftLSTM(nn.Module):
    def __init__(self, input_dim: int = 3, hidden: int = 64, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


class LSTMDetector:
    def __init__(self, cfg: dict, seed: int = 42):
        lc = cfg["detection"]["lstm"]
        self.seq_len    = lc["sequence_len"]
        self.epochs     = lc["epochs"]
        self.batch_size = lc["batch_size"]
        self.lr         = lc["lr"]
        self.patience   = lc["patience"]
        self.seed       = seed
        self.device     = torch.device("cpu")
        self.model      = DriftLSTM(
            input_dim=3,
            hidden=lc["hidden_dim"],
            n_layers=lc["num_layers"],
            dropout=lc["dropout"],
        ).to(self.device)
        self.fitted = False

    def _make_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
    ) -> Tuple:
        seqs, targets = [], []
        for i in range(len(X) - self.seq_len):
            seqs.append(X[i: i + self.seq_len])
            if y is not None:
                targets.append(y[i + self.seq_len])
        seqs = np.array(seqs, dtype=np.float32)
        if y is not None:
            return seqs, np.array(targets, dtype=np.float32)
        return seqs, None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        seqs, targets = self._make_sequences(X, y)
        if len(seqs) < self.batch_size:
            log.warning("not enough sequences for LSTM training", n=len(seqs))
            return

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(seqs),
            torch.tensor(targets),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
        )

        opt   = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.BCELoss()

        best_loss  = float("inf")
        no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                preds = self.model(xb.to(self.device))
                loss  = loss_fn(preds, yb.to(self.device))
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            sched.step()
            avg_loss = epoch_loss / len(loader)

            if avg_loss < best_loss - 1e-4:
                best_loss  = avg_loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                log.info("early stopping", epoch=epoch, best_loss=round(best_loss, 5))
                break

        self.fitted = True
        log.info("LSTM fitted", epochs_run=epoch + 1, best_loss=round(best_loss, 5))

    def predict(self, X: np.ndarray) -> np.ndarray:
        seqs, _ = self._make_sequences(X)
        if seqs is None or len(seqs) == 0:
            return np.array([])
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(seqs, dtype=torch.float32)
            scores = self.model(t.to(self.device)).numpy()
        return scores
