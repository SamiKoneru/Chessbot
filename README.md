# Chessbot

A small chess engine and board-logic package written in Python: legal move
generation, an alpha–beta search with the usual optimizations, two interchangeable
evaluators (a material baseline and an optional **NNUE** neural-network evaluator),
and a minimal GUI for trying it by hand.

## Project layout

| Path | Purpose |
|------|---------|
| `bot/` | Core engine: board state, moves, evaluation, search, Zobrist hashing, transposition table. |
| `bot/nnue/` | Optional NNUE evaluator: HalfKP feature extraction, a PyTorch model, and an inference adapter that drops into the engine in place of the material evaluator. |
| `scripts/` | Offline tooling: extract training data from Lichess PGNs, train the NNUE, and run engine-vs-engine A/B matches. |
| `tests/` | Unit tests for board, move generation, evaluation, search, and the transposition table. |
| `app/` | Tkinter UI (`app/main.py`). **Barely functional** — useful for quick interactive play, not a polished product. |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `torch` (NNUE training/inference), `python-chess` + `zstandard`
(reading Lichess dumps during data extraction), and `numpy`. The pure engine
(`bot/`, minus `bot/nnue/`) only needs the standard library; torch is required
once you use the NNUE evaluator or the training scripts.

Run the GUI from the **repository root** so imports resolve:

```bash
python app/main.py
```

## Engine features (`bot/`)

### Rules & representation

- **FEN** parse/serialize, starting position.
- **Board state**: piece grid, side to move, castling rights, en passant target, halfmove clock.
- **Special moves**: castling, en passant, promotions (with rights / EP-target updates).
- **Attack & check detection** (`attacks.py`). `is_square_attacked` scans outward
  from the target square and short-circuits, rather than enumerating every
  enemy piece's attack set.
- **Legal move generation** (pseudo-legal + king-safety filter; `move_generation.py`),
  validated with **perft** (matches the standard reference through depth 4:
  20 / 400 / 8902 / 197281).
- **Zobrist hashing** (`zobrist.py`): pieces, side to move, castling rights, en
  passant file — used as the transposition-table key.

### Search (`search.py`)

- **Negamax** with **alpha–beta pruning**.
- **Iterative deepening** over a shared transposition table, feeding the previous
  iteration's best move as the **preferred move** for ordering.
- **Quiescence search**: stand-pat with beta cutoff, full evasions when in check,
  noisy moves (captures / promotions / en passant) otherwise.
- **Transposition table** (`transposition_table.py`): depth-aware store/lookup with
  **EXACT / LOWER / UPPER** bounds and a stored best move; **size-capped** with
  FIFO eviction so memory can't grow without bound.
- **Move ordering**: MVV-LVA capture scoring, promotion/castling bonuses, hash-move
  boost, and a **killer-move** heuristic (quiet moves that caused cutoffs at the
  same ply).
- **Null-move pruning** (R=2, depth ≥ 3), guarded against check and against
  zugzwang (skipped when the side to move has only king + pawns).
- **Check extensions**: moves that give check are searched one ply deeper,
  bounded by ply so perpetual checks can't extend the search forever.

### Evaluation

The leaf evaluator is **swappable** at runtime via `bot.search.set_evaluator(fn)`
(and `reset_evaluator()`), so the search code is identical regardless of which
evaluator is in use.

- **Material (default)** — `evaluation.py`: material balance from the side to
  move, plus ply-adjusted mate/stalemate handling. No positional terms.
- **NNUE (optional)** — `bot/nnue/`: a trained neural network. See below.

## NNUE evaluator (`bot/nnue/`)

An [NNUE](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network)-style
evaluator. It is **optional and experimental** — the engine defaults to the
material evaluator; the NNUE is opted into by loading a checkpoint and calling
`set_evaluator`.

- **Features** (`features.py`): HalfKP — 40,960 sparse features per perspective
  (own-king square × piece square × piece kind), computed for both the side to
  move and the opponent.
- **Model** (`model.py`): a shared feature transformer (`nn.EmbeddingBag`, summing
  the active features into a per-perspective *accumulator*), the two accumulators
  concatenated (side-to-move first), then a small dense head
  (`2·hidden → 32 → 32 → 1`) with clipped-ReLU activations. Default hidden = 256.
- **Inference** (`evaluator.py`): `NNUEEvaluator` mirrors the
  `evaluate(board, perspective)` API so it's a drop-in replacement. Terminal
  (mate/stalemate) scoring still routes through the engine, not the network.

```python
from bot import search
from bot.nnue.evaluator import NNUEEvaluator

nnue = NNUEEvaluator.from_checkpoint("checkpoints/nnue.pt")
search.set_evaluator(nnue.evaluate_for_side_to_move)   # engine now uses the NNUE
# search.reset_evaluator()                               # back to material
```

**Known limitations**: inference recomputes the accumulator from scratch each
call (no incremental updates) and runs in float32 (no quantization), so it is
considerably slower per node than the material evaluator. The architecture is
correct NNUE shape; those two optimizations are future work.

### Training pipeline (`scripts/`)

The repository does **not** include a trained model or training data (both are
gitignored — the data file is too large for GitHub and the model is re-derivable).
To produce a model:

```bash
# 1. Download a monthly PGN dump (.pgn.zst) from https://database.lichess.org
#    (use the standard rated games; many have Stockfish [%eval] annotations).

# 2. Extract training positions (HalfKP features + STM-relative eval + WDL).
python scripts/extract_lichess.py \
    --input ~/Downloads/lichess_db_standard_rated_YYYY-MM.pgn.zst \
    --output data/positions.npz \
    --positions-per-game 4 \
    --max-positions 10000000

# 3. Train. Loss = MSE between sigmoid(model output) and a blend of
#    sigmoid(eval/400) (Stockfish teacher) and the game's win/draw/loss outcome.
python scripts/train_nnue.py \
    --data data/positions.npz \
    --output checkpoints/nnue.pt \
    --hidden 256 --epochs 8 --batch-size 8192 --lr 1e-3 --wdl-lambda 0.2

# 4. A/B test the NNUE against the material baseline (same search, only the eval
#    differs). Openings are randomized per game pair for fair, varied games.
python scripts/ab_play.py \
    --checkpoint checkpoints/nnue.pt --depth 4 --games 20 --opening-plies 4
```

## Testing

```bash
python -m pytest tests/        # unit tests
```

Move generation can additionally be spot-checked with perft (counting leaf nodes
from the start position against known-correct values).

## GUI app (`app/`)

The Tkinter app is **mainly a sandbox**: board display, applying human moves,
optional engine replies with configurable search depth, and light analysis hooks.
Expect rough edges. Treat **`bot/`** as the serious component; **`app/`** as
disposable UI glue.

## Development

- Python 3.10+ recommended (`list[X]`, `X | Y` unions are used throughout).

## Roadmap ideas

- **NNUE**: incremental accumulator updates (needs make/unmake), int8 quantization,
  larger networks, self-play fine-tuning.
- **Search**: aspiration windows, late-move reductions (LMR), SEE-based capture
  ordering and quiescence pruning, repetition/threefold detection in search scores.
- **Performance**: incremental Zobrist hashing, integer-square board representation,
  make/unmake instead of copy-make.
- **Eval (material path)**: piece-square tables, mobility, king safety, pawn structure.
- Opening book, time control, UCI protocol.
