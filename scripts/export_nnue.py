"""Export a trained NNUE checkpoint to a flat binary the Rust engine can load.

Format (all little-endian):
    magic        4 bytes   b"NNU1"
    hidden       u32       feature-transformer width
    then float32 arrays in this exact order (row-major / C order):
      feature_transformer.weight   [40960 * hidden]
      feature_bias                 [hidden]
      fc1.weight                   [32 * (2*hidden)]   (PyTorch Linear is [out, in])
      fc1.bias                     [32]
      fc2.weight                   [32 * 32]
      fc2.bias                     [32]
      fc_out.weight                [1 * 32]
      fc_out.bias                  [1]

The Rust loader (engine-rs/src/nnue.rs) reads exactly this layout.

Usage:
    ./venv/bin/python scripts/export_nnue.py \
        --checkpoint checkpoints/nnue_combined.pt \
        --output checkpoints/nnue_combined.bin
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

ORDER = [
    "feature_transformer.weight",
    "feature_bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc_out.weight",
    "fc_out.bias",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    hidden = int(ckpt.get("hidden", 256))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"NNU1")
        f.write(struct.pack("<I", hidden))
        for name in ORDER:
            arr = sd[name].detach().cpu().numpy().astype("<f4").ravel(order="C")
            f.write(arr.tobytes())

    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), hidden={hidden}")


if __name__ == "__main__":
    main()
