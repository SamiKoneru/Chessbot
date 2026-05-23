# Chessbot

A chess engine in two parts:

- a **Python** implementation — the engine plus a full **NNUE** neural-network
  training and evaluation pipeline, and a minimal GUI; and
- a **Rust** port (`engine-rs/`) being built for the raw speed needed to search
  deeply enough that a learned evaluation actually outplays plain material counting.

The Python side is the reference implementation and the ML lab; the Rust side is
the fast engine. (Why the port: a pure-Python search tops out around a few hundred
nodes/sec — the interpreter, not the algorithm, is the wall. The Rust move
generator already does ~14.7M nodes/sec.)

## Project layout

| Path | Purpose |
|------|---------|
| `bot/` | Python engine: board state, moves, evaluation, search, Zobrist hashing, transposition table. |
| `bot/nnue/` | NNUE evaluator: HalfKP feature extraction, a PyTorch model, and an inference adapter that drops into the engine in place of the material evaluator. |
| `scripts/` | Offline ML pipeline: extract/merge training data, train the NNUE, and validate/diagnose/A-B-test models. |
| `tests/` | Unit tests for board, move generation, evaluation, search, and the transposition table. |
| `app/` | Tkinter UI (`app/main.py`). **Barely functional** — useful for quick interactive play, not a polished product. |
| `engine-rs/` | Rust port of the engine (bitboards, make/unmake). Move generation is **perft-validated**; search, NNUE eval, and a UCI interface are in progress. |

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

The full set of pipeline scripts:

| Script | Purpose |
|--------|---------|
| `extract_lichess.py` | Game PGN dump → training positions (good material distribution; shallow inline evals). |
| `extract_eval_db.py` | Lichess **evaluations** DB → positions (deep, clean Stockfish evals; no game outcome). Includes a sign-safety check. |
| `merge_datasets.py` | Concatenate datasets (e.g. game + eval-DB) into one combined set. |
| `train_nnue.py` | Train the model (eval/WDL blend loss; `--wdl-lambda` controls the mix). |
| `validate_nnue.py` | Held-out correlation between the model and Stockfish — the cheap quality gate. |
| `diagnose_nnue.py` | Material and positional sanity check on a checkpoint. |
| `ab_play.py` | Head-to-head match: NNUE vs material, or NNUE vs another NNUE (`--opponent`). |
| `sanity_vs_random.py` | Functional check — does the eval beat random play? |

**Lessons baked into these tools:** train with `--wdl-lambda 0.0` for human-game data
(the win/loss signal is noisy at amateur level and flattens material values); the
eval-DB labels are cleaner but skew toward near-equal positions, so a *merged*
dataset gives the best material understanding. Practically, a learned eval that
correlates ~0.8 with Stockfish is roughly material-strength at shallow depth —
its positional knowledge only becomes a decisive edge with more search depth,
which is what the Rust port is for.

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

## Rust engine (`engine-rs/`)

A from-scratch Rust port for the speed the Python engine can't reach. Built and
validated so far:

- **Bitboard board representation**, FEN parsing, attack generation.
- **Make/unmake** moves (no per-node cloning — the main thing that crippled the
  Python search's depth).
- **Move generation** validated with **perft** against standard reference values
  (start position through depth 5 = 4,865,609; Kiwipete depth 4 = 4,085,603; plus
  an endgame position) — so castling, en passant, promotions, pins, and check
  evasion are all provably correct.
- **~14.7M nodes/sec** in perft, versus a few hundred for the Python engine.

Square indexing is Little-Endian Rank-File (a1 = 0 … h8 = 63), matching the
Python NNUE feature layout so trained weights transfer directly.

```bash
cd engine-rs
cargo build --release
cargo test --release        # perft correctness tests
./target/release/chessbot-engine   # perft benchmark (temporary entry point)
```

**In progress:** alpha-beta search (with the Python engine's optimizations), NNUE
inference (loading exported weights, with int8 quantization + incremental
accumulator), and a **UCI** interface so it can plug into chess GUIs and play
other engines.

## Development

- Python 3.10+ recommended (`list[X]`, `X | Y` unions are used throughout).
- Rust: stable toolchain (built with 1.94). `cargo build`/`cargo test` from `engine-rs/`.

## Roadmap

**Rust engine (the active path to a genuinely stronger engine):**
- Port the search (alpha-beta, TT, null-move, killers, check extensions, quiescence) + Zobrist hashing.
- NNUE inference: export the trained `.pt` weights to a flat binary, load in Rust, add int8 quantization + incremental accumulator updates.
- UCI protocol → playable in Arena / Cute Chess, testable against Stockfish at fixed nodes.

**Python NNUE (largely explored):**
- Self-play / Stockfish-binpack training data (accurate outcomes), larger networks.
- The data experiments above showed diminishing returns — depth, not eval quality, is the current ceiling, hence the Rust port.

**Python engine (lower priority):**
- Aspiration windows, LMR, SEE; opening book; time control; piece-square tables for the material eval.
