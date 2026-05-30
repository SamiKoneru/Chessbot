"""Train the NNUE model on an extracted .npz dataset.

Loss:
    predicted logit  p_hat = model(features)
    teacher prob     p_t   = sigmoid(eval_cp / SCALE)
    outcome prob     p_w   = wdl  in {0, 0.5, 1}
    target           p*    = (1 - lambda) * p_t + lambda * p_w
    loss             = MSE(sigmoid(p_hat), p*)

The blend pulls the network toward Stockfish's eval (dense signal, fast to
learn from) while anchoring it to actual game results (sparse but ground-truth).
Lambda around 0.1-0.3 is typical; higher lambda = trust the engine less.

Usage:
    python scripts/train_nnue.py \
        --data data/positions_2025-04.npz \
        --output checkpoints/nnue.pt \
        --hidden 256 \
        --batch-size 8192 \
        --epochs 8 \
        --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bot.nnue.model import NNUE

EVAL_SCALE = 400.0


class PositionDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        # Keep as numpy arrays; tensors are built per-batch in the collate_fn so
        # variable-length features stay efficient.
        self.feats_stm = data["feats_stm"]
        self.offsets_stm = data["offsets_stm"]
        self.feats_oth = data["feats_oth"]
        self.offsets_oth = data["offsets_oth"]
        self.evals = data["evals"]
        self.wdls = data["wdls"]
        self.n = len(self.evals)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        a, b = self.offsets_stm[idx], self.offsets_stm[idx + 1]
        c, d = self.offsets_oth[idx], self.offsets_oth[idx + 1]
        return (
            self.feats_stm[a:b],
            self.feats_oth[c:d],
            np.float32(self.evals[idx]),
            np.float32(self.wdls[idx]),
        )


def collate(batch):
    flat_stm: list[int] = []
    flat_oth: list[int] = []
    off_stm: list[int] = [0]
    off_oth: list[int] = [0]
    evals: list[float] = []
    wdls: list[float] = []
    for fs, fo, ev, w in batch:
        flat_stm.extend(fs.tolist())
        flat_oth.extend(fo.tolist())
        off_stm.append(len(flat_stm))
        off_oth.append(len(flat_oth))
        evals.append(ev)
        wdls.append(w)
    return (
        torch.tensor(flat_stm, dtype=torch.long),
        torch.tensor(off_stm[:-1], dtype=torch.long),
        torch.tensor(flat_oth, dtype=torch.long),
        torch.tensor(off_oth[:-1], dtype=torch.long),
        torch.tensor(evals, dtype=torch.float32),
        torch.tensor(wdls, dtype=torch.float32),
    )


# Quantization-aware-training (QAT) clamps. Dense weights are quantized at
# export with scale 64 into int8 [-127, 127], so the float weights must stay
# within ±127/64 ≈ ±1.98 for lossless quantization. We use ±2.0 to leave a
# rounding margin. The feature transformer (stored as int16, scale 127) has
# vastly more headroom; we still clamp loosely (±256/127) to bound activations.
QAT_DENSE_CLAMP = 127.0 / 64.0  # ≈ 1.984
QAT_FT_CLAMP = 256.0 / 127.0    # ≈ 2.016


def apply_qat_clamps(model) -> None:
    """Constrain weights to the int8/int16 representable range. Call after each
    optimizer step so the network adapts to quantization-compatible weights."""
    with torch.no_grad():
        model.feature_transformer.weight.clamp_(-QAT_FT_CLAMP, QAT_FT_CLAMP)
        model.feature_bias.clamp_(-QAT_FT_CLAMP, QAT_FT_CLAMP)
        model.fc1.weight.clamp_(-QAT_DENSE_CLAMP, QAT_DENSE_CLAMP)
        model.fc2.weight.clamp_(-QAT_DENSE_CLAMP, QAT_DENSE_CLAMP)
        model.fc_out.weight.clamp_(-QAT_DENSE_CLAMP, QAT_DENSE_CLAMP)


def train(
    data_path: Path,
    output_path: Path,
    hidden: int,
    batch_size: int,
    epochs: int,
    lr: float,
    wdl_lambda: float,
    val_split: float,
    device: str,
    qat: bool,
) -> None:
    dev = torch.device(device)
    dataset = PositionDataset(data_path)
    n_val = int(val_split * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=2, pin_memory=(device != "cpu"),
    )

    model = NNUE(hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.3)

    def step_loss(batch) -> torch.Tensor:
        s_idx, s_off, o_idx, o_off, ev, w = [t.to(dev, non_blocking=True) for t in batch]
        pred_logit = model(s_idx, s_off, o_idx, o_off)
        pred = torch.sigmoid(pred_logit)
        teacher = torch.sigmoid(ev / EVAL_SCALE)
        target = (1.0 - wdl_lambda) * teacher + wdl_lambda * w
        return ((pred - target) ** 2).mean()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running, n_seen = 0.0, 0
        for i, batch in enumerate(train_loader):
            opt.zero_grad()
            loss = step_loss(batch)
            loss.backward()
            opt.step()
            if qat:
                apply_qat_clamps(model)
            running += loss.item() * batch[4].size(0)
            n_seen += batch[4].size(0)
            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}  loss={running / max(1, n_seen):.5f}")
        sched.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_running, v_seen = 0.0, 0
            for batch in val_loader:
                loss = step_loss(batch)
                v_running += loss.item() * batch[4].size(0)
                v_seen += batch[4].size(0)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch}  train={running / max(1, n_seen):.5f}  "
            f"val={v_running / max(1, v_seen):.5f}  ({elapsed:.1f}s)"
        )

        torch.save(
            {"state_dict": model.state_dict(), "hidden": hidden, "epoch": epoch},
            output_path,
        )
        print(f"  saved {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wdl-lambda", type=float, default=0.2)
    ap.add_argument("--val-split", type=float, default=0.02)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    ap.add_argument(
        "--qat",
        action="store_true",
        help="Quantization-aware training: clamp weights to int8/int16 range each step "
             "so the model quantizes cleanly when exported. Use this for production runs.",
    )
    args = ap.parse_args()
    train(
        args.data, args.output, args.hidden, args.batch_size, args.epochs,
        args.lr, args.wdl_lambda, args.val_split, args.device, args.qat,
    )


if __name__ == "__main__":
    main()
