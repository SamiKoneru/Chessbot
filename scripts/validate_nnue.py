"""Measure how well a trained NNUE predicts the Stockfish eval on HELD-OUT,
in-distribution positions from the training .npz.

This is the honest accuracy check (unlike the synthetic start-minus-a-piece test
in diagnose_nnue.py, which is out-of-distribution). It reproduces the exact
train/val split used by train_nnue.py (same seed + val fraction) and evaluates
only on the validation positions the model never trained on.

Reports correlation (scale-robust, the headline number), sign agreement, and
mean absolute error in both centipawns and win-probability space.

Usage:
    ./venv/bin/python scripts/validate_nnue.py \
        --checkpoint checkpoints/nnue_lambda0.pt \
        --data data/positions_2026-04.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from bot.nnue.model import NNUE, pack_batch
from bot.nnue.evaluator import EVAL_SCALE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-split", type=float, default=0.02, help="must match training")
    ap.add_argument("--max-eval", type=int, default=50000, help="cap positions for speed")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NNUE(hidden=ckpt.get("hidden", 256))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = np.load(args.data)
    feats_stm, off_stm = data["feats_stm"], data["offsets_stm"]
    feats_oth, off_oth = data["feats_oth"], data["offsets_oth"]
    evals = data["evals"]
    n = len(evals)

    # Reproduce train_nnue.py's split exactly: random_split lays out
    # randperm(n, seed=0) and takes the last n_val indices as validation.
    n_val = int(args.val_split * n)
    n_train = n - n_val
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    val_idx = perm[n_train:][: args.max_eval]

    preds, actuals = [], []
    B = 4096
    with torch.no_grad():
        for i in range(0, len(val_idx), B):
            batch = val_idx[i : i + B]
            fs = [feats_stm[off_stm[j] : off_stm[j + 1]].tolist() for j in batch]
            fo = [feats_oth[off_oth[j] : off_oth[j + 1]].tolist() for j in batch]
            si, so = pack_batch(fs)
            oi, oo = pack_batch(fo)
            raw = model(si, so, oi, oo)  # logit
            preds.append((raw * EVAL_SCALE).numpy())
            actuals.append(evals[batch])

    pred = np.concatenate(preds)
    act = np.concatenate(actuals).astype(np.float64)

    corr = float(np.corrcoef(pred, act)[0, 1])
    sign_agree = float(np.mean(np.sign(pred) == np.sign(act)))
    mae_cp = float(np.mean(np.abs(pred - act)))
    # win-prob space (what the model actually optimizes) is robust to sigmoid saturation
    p_pred = 1.0 / (1.0 + np.exp(-pred / EVAL_SCALE))
    p_act = 1.0 / (1.0 + np.exp(-act / EVAL_SCALE))
    mae_wp = float(np.mean(np.abs(p_pred - p_act)))

    print(f"held-out positions evaluated: {len(pred):,}")
    print(f"Pearson correlation (pred vs Stockfish): {corr:.3f}   <- headline number")
    print(f"sign agreement (same side better):       {sign_agree * 100:.1f}%")
    print(f"MAE in win-probability:                  {mae_wp:.3f}")
    print(f"MAE in centipawns:                       {mae_cp:.0f} cp")
    print("\ninterpretation:")
    print("  corr > 0.90 & sign > 90%  -> model tracks Stockfish well; synthetic-test")
    print("                               weirdness is OOD noise, more data unlikely to help much")
    print("  corr 0.75-0.90            -> decent but improvable; more/better data may help")
    print("  corr < 0.75               -> genuinely weak; more/better data or bigger net warranted")


if __name__ == "__main__":
    main()
