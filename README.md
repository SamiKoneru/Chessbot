# Chessbot

A small chess engine and board-logic package written in Python: legal move generation, search, and a minimal GUI for trying it by hand.

## Project layout

| Path | Purpose |
|------|---------|
| `bot/` | Core engine: board state, moves, evaluation, search, Zobrist hashing, transposition table. |
| `app/` | Tkinter UI (`app/main.py`). **Barely functional** — useful for quick interactive play (moves, search depth), not a polished product. |

Run the GUI from the **repository root**:

```bash
python app/main.py
```

Ensure the current working directory is the `chessbot` folder so imports resolve, or run with `PYTHONPATH` set to that root.

## Engine features (`bot/`)

Verified against the current codebase:

### Rules & representation

- **FEN** parse/serialize, starting position.
- **Board state**: piece grid, side to move, castling rights, en passant target, halfmove clock (where used for FEN).
- **Special moves**: **castling**, **en passant**, **promotions** (board applies and updates rights / EP target).
- **Attack detection** and **check** (via `attacks.py`).
- **Legal move generation** (pseudo-legal + filter; see `move_generation.py`).
- **Zobrist hashing** (`zobrist.py`): pieces, side to move, castling, en passant file — used for transposition entries (does not fold in full move counters as position identity).

### Search

- **Negamax** with **alpha–beta pruning** (`search.py`).
- **Quiescence search**: stand-pat, extensions in check, tactical moves (captures, promotions, en passant) when quiet.
- **Transposition table** (`transposition_table.py`): depth-aware store/lookup with **EXACT / LOWER / UPPER** bounds, keyed by board Zobrist hash; **best move** stored for move ordering.
- **Iterative deepening** reusing a shared TT, feeding the previous iteration’s best move as **preferred move** for ordering.
- **Move ordering**: MVV-LVA-style capture scoring, promotion/castling bonuses, **hash move / previous-depth best move** boost.

### Evaluation

- **Material-only** scoring from the side to move (`evaluation.py`), plus mate/stalemate handling with a large mate score and **ply-adjusted** mate values in search terminals.
- Intended as a placeholder: **no** piece-square tables, mobility, king safety, or neural net — those are natural next steps.

### Public API (`bot/__init__.py`)

Exports include `Board`, `Move`, `legal_moves`, evaluation helpers, `alpha_beta_search`, `iterative_deepening_search`, `choose_move`, `TranspositionTable`, Zobrist helpers, etc.

## GUI app (`app/`)

The Tkinter app is **mainly a sandbox**: board display, applying human moves, optional engine replies with configurable search depth, and light analysis hooks. Expect rough edges (UX, timing, edge cases). Treat **`bot/`** as the serious component; **`app/`** as disposable UI glue.

## Development

- Python 3.10+ recommended (uses `list[X]`, `X | Y` union types in places).

## Roadmap ideas

- Richer evaluation (PST, tactics, endgame) or a **small neural net** for leaf evaluation.
- Opening book, time control, UCI protocol.
- Harder search tweaks: aspiration windows, SEE, repetition detection in search scores.
