"""Head-to-head match between two evaluators using the same search.

Plays N games at a fixed depth. Colors alternate between games so both
evaluators get equal turns with the white-to-move advantage. Reports a W/D/L
score and an Elo estimate.

Usage:
    python scripts/ab_play.py \
        --checkpoint checkpoints/nnue.pt \
        --depth 3 \
        --games 20

This is intentionally minimal — fixed depth, no time controls, no opening book.
The point is to isolate "does the eval help?" with everything else held constant.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import search
from bot.board import Board
from bot.enums import Color, PieceType
from bot.evaluation import evaluate_for_side_to_move as material_eval
from bot.nnue.evaluator import NNUEEvaluator


def random_opening(rng: random.Random, plies: int) -> Board:
    """Play `plies` uniformly-random legal moves from the start to seed a game.

    Both color-assignments of a pair start from the SAME opening, so the only
    variable between them is which engine plays which side — that keeps the
    comparison fair while giving every pair a different game.
    """
    board = Board.starting_position()
    for _ in range(plies):
        moves = board.legal_moves()
        if not moves or board.is_game_over():
            break
        board.apply_move(rng.choice(moves))
    return board


def play_one(start: Board, eval_white, eval_black, depth: int, max_plies: int = 200) -> str:
    """Play one game from `start`. Returns '1-0', '0-1', or '1/2-1/2'."""
    board = start.clone()
    for _ in range(max_plies):
        if board.is_checkmate():
            return "0-1" if board.state.side_to_move is Color.WHITE else "1-0"
        if board.is_stalemate():
            return "1/2-1/2"
        if board.state.halfmove_clock >= 100:
            return "1/2-1/2"

        # Swap in the right evaluator for whichever side is on move.
        search.set_evaluator(eval_white if board.state.side_to_move is Color.WHITE else eval_black)
        result = search.iterative_deepening_search(board, depth)
        if result.best_move is None:
            return "1/2-1/2"
        board.apply_move(result.best_move)

    return "1/2-1/2"  # ran out of plies


def elo_diff(score: float, n: int) -> float:
    """Rough Elo estimate from a fractional score (0..1). 0.5 -> 0 Elo."""
    score = min(max(score, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / score - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--opening-plies", type=int, default=4,
                    help="Random legal moves used to seed each game pair (0 = always start position)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    nnue = NNUEEvaluator.from_checkpoint(args.checkpoint)
    nnue_eval = nnue.evaluate_for_side_to_move
    base_eval = material_eval
    rng = random.Random(args.seed)

    # From NNUE's perspective: wins, draws, losses
    w = d = l = 0
    t0 = time.time()
    for i in range(args.games):
        # New random opening at the start of each color-alternating pair, so the
        # two games in a pair share an opening but swap colors (fair + diverse).
        if i % 2 == 0:
            opening = random_opening(rng, args.opening_plies)
        nnue_plays_white = (i % 2 == 0)
        if nnue_plays_white:
            result = play_one(opening, nnue_eval, base_eval, args.depth, args.max_plies)
            outcome = {"1-0": "W", "0-1": "L", "1/2-1/2": "D"}[result]
        else:
            result = play_one(opening, base_eval, nnue_eval, args.depth, args.max_plies)
            outcome = {"1-0": "L", "0-1": "W", "1/2-1/2": "D"}[result]
        if outcome == "W":
            w += 1
        elif outcome == "L":
            l += 1
        else:
            d += 1
        print(f"game {i + 1:>3}  NNUE={'W' if nnue_plays_white else 'B'}  {result}  -> NNUE {outcome}   running: {w}W-{d}D-{l}L")

    n = args.games
    score = (w + 0.5 * d) / n
    elapsed = time.time() - t0
    print()
    print(f"Final (NNUE vs material, depth {args.depth}, {n} games):")
    print(f"  {w}W - {d}D - {l}L   score={score:.3f}   est. Elo diff={elo_diff(score, n):+.0f}")
    print(f"  elapsed {elapsed:.1f}s ({elapsed / n:.1f}s/game)")


if __name__ == "__main__":
    main()
