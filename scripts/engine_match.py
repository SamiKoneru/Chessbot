"""Strength test: play the Rust engine against an opponent UCI engine over many
games (from randomized openings, alternating colors) and report the score.

The opponent defaults to Stockfish (`brew install stockfish`), optionally clamped
to a target Elo so the match is informative. You can also point `--opponent` at
the engine itself for a self-play sanity check (no Stockfish needed).

Usage:
    # vs Stockfish limited to ~1500 Elo, 0.1s/move, 20 games:
    ./venv/bin/python scripts/engine_match.py --opponent stockfish --opponent-elo 1500 --games 20

    # self-play smoke test (no external engine needed):
    ./venv/bin/python scripts/engine_match.py --opponent engine-rs/target/release/chessbot-engine --games 4
"""

from __future__ import annotations

import argparse
import math
import os
import random

import chess
import chess.engine
import chess.pgn

DEFAULT_ENGINE = "engine-rs/target/release/chessbot-engine"
DEFAULT_NNUE = "checkpoints/nnue_combined.bin"


def elo_diff(score: float, n: int) -> float:
    s = min(max(score, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / s - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--nnue", default=DEFAULT_NNUE)
    ap.add_argument("--no-nnue", action="store_true")
    ap.add_argument("--opponent", default="stockfish", help="path/command of the opponent UCI engine")
    ap.add_argument("--opponent-elo", type=int, default=None, help="clamp opponent strength (Stockfish)")
    ap.add_argument("--opponent-nnue", default=None, help="NnuePath for the opponent if it's this engine")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--movetime", type=float, default=0.1, help="seconds per move")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pgn", default=None, help="write all games to this PGN file (for later analysis)")
    args = ap.parse_args()

    ours = chess.engine.SimpleEngine.popen_uci(args.engine)
    if not args.no_nnue:
        ours.configure({"NnuePath": args.nnue})

    try:
        opp = chess.engine.SimpleEngine.popen_uci(args.opponent)
    except FileNotFoundError:
        print(f"could not launch opponent '{args.opponent}'. For Stockfish: brew install stockfish")
        ours.quit()
        return
    if args.opponent_elo is not None:
        try:
            opp.configure({"UCI_LimitStrength": True, "UCI_Elo": args.opponent_elo})
        except chess.engine.EngineError:
            print("opponent does not support UCI_Elo; running at full strength.")
    if args.opponent_nnue:
        opp.configure({"NnuePath": args.opponent_nnue})

    rng = random.Random(args.seed)
    limit = chess.engine.Limit(time=args.movetime)
    opp_name = "Stockfish" if "stockfish" in args.opponent.lower() else os.path.basename(args.opponent)
    w = d = l = 0
    games_pgn = []

    for i in range(args.games):
        board = chess.Board()
        for _ in range(args.opening_plies):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))

        ours_white = (i % 2 == 0)
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < args.max_plies:
            engine = ours if (board.turn == chess.WHITE) == ours_white else opp
            board.push(engine.play(board, limit).move)
            plies += 1

        result = board.result(claim_draw=True)  # "1-0" / "0-1" / "1/2-1/2" / "*"
        if result == "1/2-1/2" or result == "*":
            d += 1
            outcome = "D"
        else:
            white_won = result == "1-0"
            ours_won = white_won == ours_white
            if ours_won:
                w += 1
                outcome = "W"
            else:
                l += 1
                outcome = "L"
        print(f"game {i + 1:>3}: ours={'W' if ours_white else 'B'}  {result:<8} -> {outcome}   running {w}W-{d}D-{l}L")

        if args.pgn:
            g = chess.pgn.Game.from_board(board)
            g.headers["White"] = "Chessbot" if ours_white else opp_name
            g.headers["Black"] = opp_name if ours_white else "Chessbot"
            g.headers["Result"] = result
            g.headers["Event"] = f"match game {i + 1}"
            games_pgn.append(g)

    if args.pgn:
        with open(args.pgn, "w") as f:
            for g in games_pgn:
                print(g, file=f, end="\n\n")
        print(f"wrote {len(games_pgn)} games to {args.pgn}")

    ours.quit()
    opp.quit()

    n = args.games
    score = (w + 0.5 * d) / n
    print(f"\nResult ({n} games, {args.movetime}s/move): {w}W-{d}D-{l}L   score={score:.3f}   est. Elo diff {elo_diff(score, n):+.0f}")


if __name__ == "__main__":
    main()
