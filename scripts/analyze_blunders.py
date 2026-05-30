"""Classify whether one side's losses are TACTICAL or POSITIONAL.

For each move by the chosen player, a reference engine (Stockfish) measures the
"centipawn loss" — how much worse the played move was than the best move. Then:
  - if most of the total loss is concentrated in a few big blunders -> TACTICAL
  - if loss is spread thinly across many small-mistake moves          -> POSITIONAL

Generate the games first with PGN output, ideally with NO random opening so every
move is the engine's own decision:

    ./venv/bin/python scripts/engine_match.py --opponent stockfish --opponent-elo 1500 \
        --games 20 --opening-plies 0 --pgn games.pgn

Then analyze (needs Stockfish):

    ./venv/bin/python scripts/analyze_blunders.py --pgn games.pgn --side Chessbot
"""

from __future__ import annotations

import argparse

import chess
import chess.engine
import chess.pgn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--side", default="Chessbot", help="player name (from PGN headers) to analyze")
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--depth", type=int, default=14, help="reference analysis depth")
    ap.add_argument("--blunder-cp", type=int, default=150, help="a move losing >= this is a 'blunder'")
    ap.add_argument("--losses-only", action="store_true", help="only analyze games our side lost")
    args = ap.parse_args()

    try:
        sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    except FileNotFoundError:
        print(f"could not launch '{args.stockfish}'. brew install stockfish")
        return
    limit = chess.engine.Limit(depth=args.depth)

    def ref(board: chess.Board, pov: chess.Color) -> int:
        info = sf.analyse(board, limit)
        return info["score"].pov(pov).score(mate_score=10000)

    total_moves = 0
    total_loss = 0
    blunder_loss = 0
    blunders = 0

    with open(args.pgn) as f:
        while (game := chess.pgn.read_game(f)) is not None:
            if args.side in game.headers.get("White", ""):
                our = chess.WHITE
            elif args.side in game.headers.get("Black", ""):
                our = chess.BLACK
            else:
                continue
            if args.losses_only:
                res = game.headers.get("Result", "*")
                our_lost = (res == "0-1") if our == chess.WHITE else (res == "1-0")
                if not our_lost:
                    continue

            board = game.board()
            for move in game.mainline_moves():
                if board.turn == our:
                    best = ref(board, our)        # value before our move (best play)
                    board.push(move)
                    after = ref(board, our)       # value after the move we actually played
                    loss = max(0, best - after)
                    total_moves += 1
                    total_loss += loss
                    if loss >= args.blunder_cp:
                        blunders += 1
                        blunder_loss += loss
                else:
                    board.push(move)

    sf.quit()

    if total_moves == 0:
        print(f"no moves found for side '{args.side}' (check the player name in the PGN headers).")
        return

    acpl = total_loss / total_moves
    blunder_frac = blunder_loss / total_loss if total_loss else 0.0
    print(f"analyzed {total_moves} moves by '{args.side}'")
    print(f"  ACPL (avg centipawn loss/move): {acpl:.0f}")
    print(f"  blunders (loss >= {args.blunder_cp}cp): {blunders}  ({100 * blunders / total_moves:.1f}% of moves)")
    print(f"  share of total loss from blunders: {100 * blunder_frac:.0f}%")
    print()
    if blunder_frac >= 0.6:
        print("  VERDICT: mostly TACTICAL — loss concentrated in big blunders.")
        print("           Lever: more search depth (int8 quantization).")
    elif acpl <= 40 and blunder_frac < 0.4:
        print("  VERDICT: mostly POSITIONAL — steady small losses, few blunders.")
        print("           Lever: a better eval (self-play training data).")
    else:
        print("  VERDICT: mixed — both tactical blunders and positional drift contribute.")


if __name__ == "__main__":
    main()
