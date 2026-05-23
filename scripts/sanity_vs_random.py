"""Sanity check: can an evaluator+search beat a random-move opponent?

If a functioning eval can't crush random play, something is broken in the
eval/integration. If it crushes random but only draws the material baseline,
then material-at-shallow-depth is just a strong wall and the NNUE-vs-material
test is the problem, not the model.

Runs BOTH the NNUE and the material eval against random, as a reference.

Usage:
    ./venv/bin/python scripts/sanity_vs_random.py --checkpoint checkpoints/nnue_lambda0.pt --depth 3 --games 10
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import search
from bot.board import Board
from bot.enums import Color
from bot.evaluation import evaluate_for_side_to_move as material_eval
from bot.nnue.evaluator import NNUEEvaluator


def play_vs_random(engine_eval, engine_color: Color, depth: int, rng: random.Random,
                   max_plies: int = 200) -> str:
    """One game: engine_color uses engine_eval+search, the other side plays random."""
    board = Board.starting_position()
    for _ in range(max_plies):
        if board.is_checkmate():
            winner = board.state.side_to_move.opposite
            return "W" if winner is engine_color else "L"
        if board.is_stalemate() or board.state.halfmove_clock >= 100:
            return "D"

        if board.state.side_to_move is engine_color:
            search.set_evaluator(engine_eval)
            res = search.iterative_deepening_search(board, depth)
            mv = res.best_move
            if mv is None:
                return "D"
        else:
            mv = rng.choice(board.legal_moves())
        board.apply_move(mv)
    return "D"


def run(label: str, engine_eval, depth: int, games: int, seed: int) -> None:
    w = d = l = 0
    rng = random.Random(seed)
    for i in range(games):
        color = Color.WHITE if i % 2 == 0 else Color.BLACK
        out = play_vs_random(engine_eval, color, depth, rng)
        w += out == "W"
        d += out == "D"
        l += out == "L"
        print(f"  {label} game {i + 1:>2}: engine={'W' if color is Color.WHITE else 'B'}  -> {out}")
    print(f"{label} vs random: {w}W-{d}D-{l}L  (score {(w + 0.5 * d) / games:.2f})\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    nnue = NNUEEvaluator.from_checkpoint(args.checkpoint)
    print("=== material eval vs random (reference) ===")
    run("material", material_eval, args.depth, args.games, args.seed)
    print("=== NNUE eval vs random ===")
    run("NNUE", nnue.evaluate_for_side_to_move, args.depth, args.games, args.seed)


if __name__ == "__main__":
    main()
