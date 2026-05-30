"""Export a trained NNUE checkpoint to a quantized int8/int16 binary that the
Rust engine loads.

The model was trained in float; this script does the quantization at export time
(post-training quantization, PTQ). For best accuracy retrain with `--qat` in
train_nnue.py, which keeps the weights in the representable range during training.

Scheme (per-layer fixed scales — matches the Rust loader exactly):
    feature transformer:    weights * 127 -> int16
                            bias    * 127 -> int16
    dense layers (fc1, fc2, out):
                            weights * 64  -> int8  (clamped to [-127, 127])
                            biases  * 127 * 64 -> int32

Format (all little-endian):
    magic         4 bytes   b"NNU2"
    hidden        u32
    then the arrays in order:
      feature_transformer.weight   int16 [40960 * hidden]
      feature_bias                 int16 [hidden]
      fc1.weight                   int8  [32 * (2*hidden)]
      fc1.bias                     int32 [32]
      fc2.weight                   int8  [32 * 32]
      fc2.bias                     int32 [32]
      fc_out.weight                int8  [32]
      fc_out.bias                  int32 [1]

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

# Must match Rust engine's constants in nnue.rs.
FT_SCALE = 127
W_SCALE = 64
BIAS_SCALE = FT_SCALE * W_SCALE  # 8128


def quantize_i16(arr: np.ndarray, scale: int) -> np.ndarray:
    """Scale by `scale`, round, clamp to int16 range, return int16."""
    q = np.round(arr.astype(np.float64) * scale)
    q = np.clip(q, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return q.astype(np.int16)


def quantize_i8(arr: np.ndarray, scale: int) -> np.ndarray:
    """Scale by `scale`, round, clamp to int8 range, return int8.

    We clamp to [-127, 127] (symmetric) rather than the full [-128, 127] so the
    Rust code can use signed arithmetic without sign-bit surprises.
    """
    q = np.round(arr.astype(np.float64) * scale)
    q = np.clip(q, -127, 127)
    return q.astype(np.int8)


def quantize_i32(arr: np.ndarray, scale: int) -> np.ndarray:
    q = np.round(arr.astype(np.float64) * scale)
    q = np.clip(q, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    return q.astype(np.int32)


def report_clipping(name: str, raw: np.ndarray, q: np.ndarray, scale: int, lo: int, hi: int) -> None:
    """Warn if quantization clipped a noticeable fraction of weights — that
    indicates the trained weights exceed the int range and PTQ accuracy will
    suffer. Use --qat in training to keep weights in range."""
    target = np.round(raw.astype(np.float64) * scale)
    clipped = int(np.sum((target < lo) | (target > hi)))
    total = int(target.size)
    if clipped:
        pct = 100.0 * clipped / total
        peak = float(np.max(np.abs(raw)))
        print(f"  warn: {name}: {clipped}/{total} weights clipped ({pct:.2f}%), max |w|={peak:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    hidden = int(ckpt.get("hidden", 256))

    def arr(name: str) -> np.ndarray:
        return sd[name].detach().cpu().numpy().astype(np.float32).ravel(order="C")

    # Raw float weights / biases.
    ft_w = arr("feature_transformer.weight")
    ft_b = arr("feature_bias")
    fc1_w = arr("fc1.weight")
    fc1_b = arr("fc1.bias")
    fc2_w = arr("fc2.weight")
    fc2_b = arr("fc2.bias")
    out_w = arr("fc_out.weight")
    out_b = arr("fc_out.bias")

    # Quantize. Report any clipping (warning: lots of clipping = accuracy hit;
    # retrain with --qat to fix).
    report_clipping("ft_weight", ft_w, None, FT_SCALE, -32768, 32767)
    report_clipping("fc1_w", fc1_w, None, W_SCALE, -127, 127)
    report_clipping("fc2_w", fc2_w, None, W_SCALE, -127, 127)
    report_clipping("out_w", out_w, None, W_SCALE, -127, 127)

    ft_w_q = quantize_i16(ft_w, FT_SCALE)
    ft_b_q = quantize_i16(ft_b, FT_SCALE)
    fc1_w_q = quantize_i8(fc1_w, W_SCALE)
    fc1_b_q = quantize_i32(fc1_b, BIAS_SCALE)
    fc2_w_q = quantize_i8(fc2_w, W_SCALE)
    fc2_b_q = quantize_i32(fc2_b, BIAS_SCALE)
    out_w_q = quantize_i8(out_w, W_SCALE)
    out_b_q = quantize_i32(out_b, BIAS_SCALE)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"NNU2")
        f.write(struct.pack("<I", hidden))
        f.write(ft_w_q.tobytes())
        f.write(ft_b_q.tobytes())
        f.write(fc1_w_q.tobytes())
        f.write(fc1_b_q.tobytes())
        f.write(fc2_w_q.tobytes())
        f.write(fc2_b_q.tobytes())
        f.write(out_w_q.tobytes())
        f.write(out_b_q.tobytes())

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.1f} MB, int8/int16 quantized, hidden={hidden})")


if __name__ == "__main__":
    main()
