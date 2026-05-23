"""Extract NNUE training samples from the Lichess **evaluations** database.

Input: `lichess_db_eval.jsonl.zst` from https://database.lichess.org — a stream
of JSON lines, each a position (FEN) with one or more Stockfish analyses. These
evals are deep (often depth 30+) and far cleaner than the shallow inline `[%eval]`
in game PGNs, which is why this is the better data source for raising eval quality.

Each line looks like:
    {"fen": "<fen>", "evals": [{"pvs": [{"cp": 31, "line": "..."}], "depth": 36, "knodes": ...}, ...]}

Differences from extract_lichess.py:
  - One position per line (no games, no per-game sampling).
  - cp is already in CENTIPAWNS (the PGN [%eval] was in pawns — do NOT ×100 here).
  - No game outcome, so there is no real WDL. We write a dummy wdl=0.5; you MUST
    train this data with `--wdl-lambda 0.0` (the WDL term must be off).

Output .npz layout matches extract_lichess.py exactly, so train_nnue.py /
validate_nnue.py read it unchanged:
    feats_stm, offsets_stm, feats_oth, offsets_oth (HalfKP features per perspective)
    evals (float32, STM-relative centipawns)
    wdls  (float32, all 0.5 — placeholder, unused when wdl_lambda=0)

SIGN SAFETY: the script assumes cp is white-relative by default (--cp-perspective
white) and converts to STM-relative. It also auto-checks the eval sign against
material balance on lopsided positions and warns if they look anti-correlated
(i.e. the perspective is flipped — rerun with --cp-perspective stm).

Usage:
    ./venv/bin/python scripts/extract_eval_db.py \
        --input ~/Downloads/lichess_db_eval.jsonl.zst \
        --output data/eval_positions.npz \
        --min-depth 16 --max-positions 10000000
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np
import zstandard as zstd

# Reuse the EXACT feature extraction the game-PGN path uses, so data from both
# sources is interchangeable and consistent with bot/nnue/features.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_lichess import active_features  # noqa: E402

_MATERIAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
             chess.ROOK: 500, chess.QUEEN: 900}


def material_white(board: chess.Board) -> int:
    s = 0
    for _, p in board.piece_map().items():
        v = _MATERIAL.get(p.piece_type, 0)
        s += v if p.color == chess.WHITE else -v
    return s


def parse_board(fen: str) -> chess.Board | None:
    parts = fen.split()
    if len(parts) == 4:
        fen = fen + " 0 1"
    elif len(parts) == 5:
        fen = fen + " 1"
    try:
        return chess.Board(fen)
    except (ValueError, KeyError):
        return None


def best_cp(entry: dict, min_depth: int) -> int | None:
    """Return centipawns (white/stm-relative as stored) from the deepest analysis.

    Skips positions whose best line is a forced mate (consistent with the PGN
    extractor) and positions analyzed shallower than min_depth.
    """
    evals = entry.get("evals")
    if not evals:
        return None
    best = max(evals, key=lambda e: e.get("depth", 0))
    if best.get("depth", 0) < min_depth:
        return None
    pvs = best.get("pvs")
    if not pvs:
        return None
    pv0 = pvs[0]
    if "cp" not in pv0:  # mate or malformed
        return None
    return pv0["cp"]


def iter_entries(input_path: str):
    """Stream JSON lines from a .jsonl.zst file (or stdin if input is '-')."""
    dctx = zstd.ZstdDecompressor()
    raw = sys.stdin.buffer if input_path == "-" else open(input_path, "rb")
    try:
        with dctx.stream_reader(raw) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    finally:
        if input_path != "-":
            raw.close()


def extract(input_path: str, output_path: Path, min_depth: int, max_abs_cp: float,
            max_positions: int, cp_perspective: str) -> None:
    feats_stm_chunks: list[np.ndarray] = []
    feats_oth_chunks: list[np.ndarray] = []
    offsets_stm: list[int] = [0]
    offsets_oth: list[int] = [0]
    evals: list[float] = []

    n = n_lines = 0
    # Sign-safety counters: on lopsided positions, does eval agree with material?
    sign_checked = sign_agree = 0
    t0 = time.time()

    for entry in iter_entries(input_path):
        n_lines += 1
        fen = entry.get("fen")
        if not fen:
            continue
        cp = best_cp(entry, min_depth)
        if cp is None or abs(cp) > max_abs_cp:
            continue
        board = parse_board(fen)
        if board is None:
            continue

        stm_white = board.turn == chess.WHITE

        # Interpret the stored cp and convert to white-relative for the sign check,
        # then to STM-relative for the label.
        cp_white = cp if cp_perspective == "white" else (cp if stm_white else -cp)
        cp_stm = cp_white if stm_white else -cp_white

        # Sign sanity: on materially + evaluatively lopsided positions, white-relative
        # eval and white material balance should mostly agree.
        mat = material_white(board)
        if abs(cp_white) >= 300 and abs(mat) >= 200:
            sign_checked += 1
            if (cp_white > 0) == (mat > 0):
                sign_agree += 1

        f_stm = active_features(board, perspective_white=stm_white)
        f_oth = active_features(board, perspective_white=not stm_white)
        feats_stm_chunks.append(np.asarray(f_stm, dtype=np.int32))
        feats_oth_chunks.append(np.asarray(f_oth, dtype=np.int32))
        offsets_stm.append(offsets_stm[-1] + len(f_stm))
        offsets_oth.append(offsets_oth[-1] + len(f_oth))
        evals.append(cp_stm)
        n += 1

        if n % 50_000 == 0:
            rate = n / max(1e-9, time.time() - t0)
            print(f"  ...{n:,} positions ({n_lines:,} lines, {rate:.0f} pos/s)", file=sys.stderr)
        if n >= max_positions:
            break

    print(f"Done. {n:,} positions from {n_lines:,} lines in {time.time() - t0:.1f}s", file=sys.stderr)

    # Report the sign check prominently — this is the guard against the perspective bug.
    if sign_checked:
        frac = sign_agree / sign_checked
        print(f"\nSIGN CHECK: eval agreed with material on {frac * 100:.1f}% of "
              f"{sign_checked:,} lopsided positions.", file=sys.stderr)
        if frac < 0.6:
            print("  >>> WARNING: low agreement — the cp perspective is likely FLIPPED.\n"
                  "      Re-run with the opposite --cp-perspective "
                  f"(you used '{cp_perspective}').", file=sys.stderr)
        else:
            print(f"  OK: '{cp_perspective}'-relative interpretation looks correct.", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        feats_stm=np.concatenate(feats_stm_chunks) if feats_stm_chunks else np.array([], dtype=np.int32),
        offsets_stm=np.asarray(offsets_stm, dtype=np.int64),
        feats_oth=np.concatenate(feats_oth_chunks) if feats_oth_chunks else np.array([], dtype=np.int32),
        offsets_oth=np.asarray(offsets_oth, dtype=np.int64),
        evals=np.asarray(evals, dtype=np.float32),
        wdls=np.full(len(evals), 0.5, dtype=np.float32),  # placeholder; train with --wdl-lambda 0.0
    )
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to lichess_db_eval.jsonl.zst, or '-' for stdin")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-depth", type=int, default=16, help="Skip analyses shallower than this")
    ap.add_argument("--max-abs-cp", type=float, default=1500.0, help="Drop near-decided positions")
    ap.add_argument("--max-positions", type=int, default=10_000_000)
    ap.add_argument("--cp-perspective", choices=["white", "stm"], default="white",
                    help="How the DB stores cp. Default 'white'; the sign check will tell you if wrong.")
    args = ap.parse_args()
    extract(args.input, args.output, args.min_depth, args.max_abs_cp,
            args.max_positions, args.cp_perspective)


if __name__ == "__main__":
    main()
