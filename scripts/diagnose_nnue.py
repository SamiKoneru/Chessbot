"""Quick sanity diagnostics for a trained NNUE checkpoint.

Run this on any checkpoint to see, in a few seconds, whether the eval:
  1. respects MATERIAL (a knight should read clearly more than a pawn; +1 pawn
     should be clearly above equal), and
  2. has POSITIONAL sense (good opening moves ranked above pointless ones),
  3. and crucially, whether the positional spread is SMALL relative to material
     (if a positional swing rivals a whole pawn, the eval will trade material
     for dubious position — the failure we diagnosed).

Usage:
    ./venv/bin/python scripts/diagnose_nnue.py --checkpoint checkpoints/nnue.pt

Note: the material positions are the start position with a piece removed, which
is slightly out-of-distribution for a model trained on real middlegames. Treat
the numbers as a *relative* measure for comparing checkpoints, not absolute cp.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.board import Board
from bot.enums import Color
from bot.nnue.evaluator import NNUEEvaluator

MATERIAL_CASES = [
    ("equal (start)",  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0),
    ("white +1 pawn",  "rnbqkbnr/pp1ppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 100),
    ("white +2 pawns", "rnbqkbnr/1p1ppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 200),
    ("white +knight",  "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 320),
    ("white +rook",    "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 500),
    ("white +queen",   "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 900),
]


def material_scale(ev: NNUEEvaluator) -> None:
    print("Material valuation (White POV)        NNUE     (material eval would say)")
    pawn_val = None
    for name, fen, ref in MATERIAL_CASES:
        s = ev.evaluate(Board.from_fen(fen), Color.WHITE)
        if name == "white +1 pawn":
            pawn_val = s
        flag = ""
        if name == "white +1 pawn" and s <= ev.evaluate(Board.from_fen(MATERIAL_CASES[0][1]), Color.WHITE):
            flag = "  <-- BAD: +1 pawn not above equal"
        print(f"  {name:<18} {s:+6d}        {('+' + str(ref)) if ref else '0':>6}{flag}")
    return pawn_val


def positional_spread(ev: NNUEEvaluator):
    b = Board.starting_position()
    rows = []
    for mv in b.legal_moves():
        c = b.clone()
        c.apply_move(mv)
        rows.append((mv.uci, ev.evaluate(c, Color.WHITE)))
    rows.sort(key=lambda r: r[1], reverse=True)
    print("\nFirst-move ranking (White POV) — material is equal for all of these:")
    print("  best: " + ", ".join(f"{u}({s:+d})" for u, s in rows[:5]))
    print("  worst:" + ", ".join(f"{u}({s:+d})" for u, s in rows[-5:]))
    spread = rows[0][1] - rows[-1][1]
    return spread


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    ev = NNUEEvaluator.from_checkpoint(args.checkpoint)
    print(f"=== diagnostics for {args.checkpoint} ===\n")
    pawn_val = material_scale(ev)
    spread = positional_spread(ev)

    print(f"\npositional spread across first moves: {spread} cp")
    if pawn_val is not None:
        print(f"value of +1 pawn:                     {pawn_val} cp")
        if spread > abs(pawn_val):
            print("  >>> WARNING: positional spread exceeds a pawn — eval may trade material for position.")
        else:
            print("  OK: a pawn outweighs the positional spread.")


if __name__ == "__main__":
    main()
