"""Extract NNUE training samples from a Lichess monthly PGN dump.

Reads a .pgn.zst file (streaming, no full decompression to disk), filters games
that have Stockfish eval annotations ([%eval ...] comments), samples positions
from each game, computes HalfKP features for both perspectives, and writes a
single .npz file with arrays the trainer can mmap.

Usage:
    python scripts/extract_lichess.py \
        --input /path/to/lichess_db_standard_rated_2025-04.pgn.zst \
        --output data/positions_2025-04.npz \
        --positions-per-game 4 \
        --max-positions 10_000_000

Output .npz layout:
    feats_stm     (int32, sum-N_active)  concatenated active features for STM perspective
    offsets_stm   (int64, N+1)           start indices into feats_stm (last entry = len)
    feats_oth     (int32, sum-N_active)  same for the other side
    offsets_oth   (int64, N+1)
    evals         (float32, N)           Stockfish eval in centipawns, STM-relative
    wdls          (float32, N)           game outcome in {1.0, 0.5, 0.0}, STM-relative
"""

from __future__ import annotations

import argparse
import io
import random
import re
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import zstandard as zstd

# HalfKP feature math, kept inline so this script doesn't depend on the engine
# Board (python-chess is much faster to parse from PGN).

PIECE_BASE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
NUM_SQUARES = 64
NUM_PIECE_KINDS = 10
STRIDE_KING = NUM_SQUARES * NUM_PIECE_KINDS

_EVAL_RE = re.compile(r"\[%eval ([^\]]+)\]")


def active_features(board: chess.Board, perspective_white: bool) -> list[int]:
    king_sq = board.king(chess.WHITE if perspective_white else chess.BLACK)
    if not perspective_white:
        king_sq ^= 56
    out: list[int] = []
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        psq = sq if perspective_white else (sq ^ 56)
        base = PIECE_BASE[piece.piece_type]
        is_own = (piece.color == chess.WHITE) == perspective_white
        kind = base * 2 + (0 if is_own else 1)
        out.append(king_sq * STRIDE_KING + psq * NUM_PIECE_KINDS + kind)
    return out


def parse_eval(comment: str) -> float | None:
    """Return eval in centipawns (white-relative), or None for mate / missing."""
    m = _EVAL_RE.search(comment or "")
    if not m:
        return None
    token = m.group(1).strip()
    if token.startswith("#"):
        return None  # mate scores: skip, too extreme to train on directly
    try:
        return float(token) * 100.0  # PGN evals are in pawns; convert to cp
    except ValueError:
        return None


def result_to_wdl(result: str) -> float | None:
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result)


def iter_games(input_path: Path):
    """Stream-decompress a .pgn.zst and yield chess.pgn.Game objects."""
    dctx = zstd.ZstdDecompressor()
    with open(input_path, "rb") as raw:
        with dctx.stream_reader(raw) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            while True:
                game = chess.pgn.read_game(text)
                if game is None:
                    return
                yield game


def extract(
    input_path: Path,
    output_path: Path,
    positions_per_game: int,
    max_positions: int,
    min_ply: int,
    max_abs_cp: float,
    seed: int,
) -> None:
    rng = random.Random(seed)

    feats_stm_chunks: list[np.ndarray] = []
    feats_oth_chunks: list[np.ndarray] = []
    offsets_stm: list[int] = [0]
    offsets_oth: list[int] = [0]
    evals: list[float] = []
    wdls: list[float] = []

    n_positions = 0
    n_games = 0
    n_games_with_eval = 0
    t0 = time.time()

    for game in iter_games(input_path):
        n_games += 1
        wdl_white = result_to_wdl(game.headers.get("Result", "*"))
        if wdl_white is None:
            continue

        # Collect all (board, eval_white_cp) pairs in this game where eval annotation exists
        candidates: list[tuple[chess.Board, float, int]] = []
        board = game.board()
        ply = 0
        for node in game.mainline():
            board.push(node.move)
            ply += 1
            if ply < min_ply:
                continue
            cp = parse_eval(node.comment)
            if cp is None:
                continue
            if abs(cp) > max_abs_cp:
                continue
            candidates.append((board.copy(stack=False), cp, ply))

        if not candidates:
            continue
        n_games_with_eval += 1

        sampled = rng.sample(candidates, k=min(positions_per_game, len(candidates)))
        for bd, cp_white, _ in sampled:
            stm_white = bd.turn == chess.WHITE
            cp_stm = cp_white if stm_white else -cp_white
            wdl_stm = wdl_white if stm_white else (1.0 - wdl_white)

            f_stm = active_features(bd, perspective_white=stm_white)
            f_oth = active_features(bd, perspective_white=not stm_white)

            feats_stm_chunks.append(np.asarray(f_stm, dtype=np.int32))
            feats_oth_chunks.append(np.asarray(f_oth, dtype=np.int32))
            offsets_stm.append(offsets_stm[-1] + len(f_stm))
            offsets_oth.append(offsets_oth[-1] + len(f_oth))
            evals.append(cp_stm)
            wdls.append(wdl_stm)
            n_positions += 1

            if n_positions >= max_positions:
                break

        if n_positions % 10_000 == 0 and n_positions:
            rate = n_positions / max(1e-9, time.time() - t0)
            print(
                f"  ...{n_positions:,} positions  "
                f"({n_games:,} games scanned, {rate:.0f} pos/s)",
                file=sys.stderr,
            )

        if n_positions >= max_positions:
            break

    print(
        f"Done. {n_positions:,} positions from {n_games_with_eval:,} games "
        f"(of {n_games:,} scanned) in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        feats_stm=np.concatenate(feats_stm_chunks) if feats_stm_chunks else np.array([], dtype=np.int32),
        offsets_stm=np.asarray(offsets_stm, dtype=np.int64),
        feats_oth=np.concatenate(feats_oth_chunks) if feats_oth_chunks else np.array([], dtype=np.int32),
        offsets_oth=np.asarray(offsets_oth, dtype=np.int64),
        evals=np.asarray(evals, dtype=np.float32),
        wdls=np.asarray(wdls, dtype=np.float32),
    )
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Path to .pgn.zst")
    ap.add_argument("--output", type=Path, required=True, help="Where to write .npz")
    ap.add_argument("--positions-per-game", type=int, default=4)
    ap.add_argument("--max-positions", type=int, default=10_000_000)
    ap.add_argument("--min-ply", type=int, default=10, help="Skip opening")
    ap.add_argument("--max-abs-cp", type=float, default=1500.0, help="Drop near-winning evals")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    extract(
        args.input,
        args.output,
        args.positions_per_game,
        args.max_positions,
        args.min_ply,
        args.max_abs_cp,
        args.seed,
    )


if __name__ == "__main__":
    main()
