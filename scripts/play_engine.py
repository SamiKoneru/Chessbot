"""Play a game against the Rust UCI engine in the terminal.

Drives the engine over UCI via python-chess. You type moves in UCI ("e2e4",
"e7e8q") or SAN ("Nf3", "O-O"); the engine replies.

Usage:
    ./venv/bin/python scripts/play_engine.py                 # you are White, engine depth 8
    ./venv/bin/python scripts/play_engine.py --human black --movetime 1.0
    ./venv/bin/python scripts/play_engine.py --no-nnue       # engine uses material eval

Commands during play: a move, or `quit` / `resign` / `board` / `undo`.
"""

from __future__ import annotations

import argparse

import chess
import chess.engine

DEFAULT_ENGINE = "engine-rs/target/release/chessbot-engine"
DEFAULT_NNUE = "checkpoints/nnue_combined.bin"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--nnue", default=DEFAULT_NNUE)
    ap.add_argument("--no-nnue", action="store_true", help="use the material eval instead")
    ap.add_argument("--human", choices=["white", "black"], default="white")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--movetime", type=float, default=None, help="seconds per engine move (overrides depth)")
    args = ap.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    if not args.no_nnue:
        engine.configure({"NnuePath": args.nnue})
    limit = chess.engine.Limit(time=args.movetime) if args.movetime else chess.engine.Limit(depth=args.depth)

    board = chess.Board()
    human_white = args.human == "white"

    def show() -> None:
        print()
        print(board.unicode(borders=False, empty_square="."))
        print(f"FEN: {board.fen()}")

    print("You are", "White" if human_white else "Black", "— type a move (UCI or SAN), or quit/resign/board/undo.")
    while not board.is_game_over(claim_draw=True):
        human_turn = (board.turn == chess.WHITE) == human_white
        if human_turn:
            show()
            raw = input("your move> ").strip()
            if raw in ("quit", "exit"):
                break
            if raw == "resign":
                print("you resigned.")
                break
            if raw == "board":
                continue
            if raw == "undo":
                if len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                else:
                    print("nothing to undo.")
                continue
            move = None
            try:
                move = board.parse_san(raw)
            except ValueError:
                try:
                    move = chess.Move.from_uci(raw)
                except ValueError:
                    move = None
            if move is None or move not in board.legal_moves:
                print("illegal/unparseable move; try again (e.g. e4, Nf3, e2e4).")
                continue
            board.push(move)
        else:
            result = engine.play(board, limit)
            print(f"engine plays: {board.san(result.move)}")
            board.push(result.move)

    show()
    print("\ngame over:", board.result(claim_draw=True))
    engine.quit()


if __name__ == "__main__":
    main()
