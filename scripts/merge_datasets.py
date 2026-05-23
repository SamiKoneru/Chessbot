"""Merge multiple NNUE .npz datasets into one by concatenating positions.

Combines complementary sources — e.g. game-PGN data (good material distribution)
+ eval-DB data (deep, clean labels). Output format is identical, so it's a
drop-in for train_nnue.py / validate_nnue.py.

The fiddly part is the offsets arrays: each source's offsets are cumulative from
0, so later sources must be shifted by the running feature-array length (and
their leading 0 dropped, since it coincides with the previous source's end).

Usage:
    ./venv/bin/python scripts/merge_datasets.py \
        --inputs data/positions_2026-04.npz data/eval_positions.npz \
        --output data/combined.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def merge(inputs: list[str]) -> dict[str, np.ndarray]:
    feats_stm_parts, feats_oth_parts = [], []
    off_stm_parts, off_oth_parts = [], []
    evals_parts, wdls_parts = [], []
    base_stm = base_oth = 0
    total = 0

    for i, path in enumerate(inputs):
        d = np.load(path)
        fs, os_ = d["feats_stm"], d["offsets_stm"]
        fo, oo = d["feats_oth"], d["offsets_oth"]
        ev, wd = d["evals"], d["wdls"]
        n = len(ev)

        feats_stm_parts.append(fs)
        feats_oth_parts.append(fo)
        # First file keeps its full offsets (incl. leading 0). Later files drop the
        # leading 0 and shift by the running base, so indices stay contiguous.
        off_stm_parts.append((os_ if i == 0 else os_[1:]) + base_stm)
        off_oth_parts.append((oo if i == 0 else oo[1:]) + base_oth)
        base_stm += len(fs)
        base_oth += len(fo)
        evals_parts.append(ev)
        wdls_parts.append(wd)
        total += n
        print(f"  {path}: {n:,} positions")

    out = dict(
        feats_stm=np.concatenate(feats_stm_parts),
        offsets_stm=np.concatenate(off_stm_parts),
        feats_oth=np.concatenate(feats_oth_parts),
        offsets_oth=np.concatenate(off_oth_parts),
        evals=np.concatenate(evals_parts),
        wdls=np.concatenate(wdls_parts),
    )
    # Sanity: offsets must have exactly n+1 entries and end at the feature length.
    n_total = len(out["evals"])
    assert len(out["offsets_stm"]) == n_total + 1, "offsets_stm length mismatch"
    assert len(out["offsets_oth"]) == n_total + 1, "offsets_oth length mismatch"
    assert out["offsets_stm"][-1] == len(out["feats_stm"]), "offsets_stm endpoint mismatch"
    assert out["offsets_oth"][-1] == len(out["feats_oth"]), "offsets_oth endpoint mismatch"
    print(f"merged total: {n_total:,} positions")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    out = merge(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
