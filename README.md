# Chessbot

A chess engine in two parts:

- a **Python** implementation — the reference engine plus a full **NNUE**
  neural-network training and evaluation pipeline; and
- a **Rust** port (`engine-rs/`) — the actual fast engine, with a complete
  alpha-beta search, an NNUE evaluator with an incrementally-updated accumulator,
  and a **UCI** interface so it plugs into chess GUIs and plays other engines.

The Python side is the reference + ML lab. The Rust side is where the engine
actually plays at strength: a pure-Python search tops out at a few hundred nodes
per second (the interpreter, not the algorithm), so a learned eval has nowhere
to land. Rust does **millions** of nodes per second, deep enough that the eval's
positional knowledge becomes a winning advantage. The Tkinter GUI drives the
Rust engine over UCI so playing it interactively gets the full strength.

## Project layout

| Path | Purpose |
|------|---------|
| `engine-rs/` | **The fast engine** — Rust port with bitboards, make/unmake, full search, NNUE inference, and UCI. Perft-validated. |
| `bot/` | Python reference engine — board, moves, evaluation, search, Zobrist, TT. Used as the oracle for porting and validation. |
| `bot/nnue/` | NNUE training/inference in PyTorch: HalfKP features, the trainable model, and a Python evaluator. |
| `scripts/` | Offline ML pipeline + match/diagnosis tooling: extract/merge data, train, validate, export to the Rust engine, play it, measure its strength. |
| `app/` | Tkinter GUI that drives the Rust UCI engine non-blocking (background-thread analysis, responsive UI). |
| `tests/` | Python unit tests for board, move generation, evaluation, search, TT. |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd engine-rs && cargo build --release && cd ..
```

Dependencies:
- **Python**: `torch` (NNUE training/inference), `python-chess` (drives the Rust
  engine over UCI from the GUI and the match scripts), `numpy`, `zstandard`
  (for reading Lichess dumps).
- **Rust**: stable toolchain (built with 1.94). Just `cargo build --release`.

If you want to actually play the engine, you also need a trained NNUE binary at
`checkpoints/nnue_combined.bin`. The repo doesn't ship one (gitignored — too big
and re-derivable). See [Training](#nnue-training-pipeline) below.

## Play it

**In the Tkinter GUI** (the GUI drives the Rust engine over UCI in a background
thread, so the board stays responsive while the engine thinks):

```bash
python app/main.py
```

**In the terminal**:

```bash
./venv/bin/python scripts/play_engine.py --human white --depth 8
```

**Plug into a chess GUI**: the Rust binary is a standard UCI engine. Point
Arena, Cute Chess, or BanksiaGUI at `engine-rs/target/release/chessbot-engine`,
and set the `NnuePath` option to your `checkpoints/nnue_combined.bin`.

## Rust engine (`engine-rs/`)

The actual playing engine. Built and validated:

### Board + move generation

- **Bitboard board** with FEN parsing and full attack generation.
- **Make/unmake** moves (no per-node cloning), incremental Zobrist hashing.
- **Perft-validated** against standard references: startpos through depth 5
  (4,865,609), Kiwipete depth 4 (4,085,603), and an endgame position — so
  castling, en passant, promotions, pins, and check evasion are all provably
  correct.

### Search

- **Negamax + alpha-beta** with iterative deepening and a sized transposition
  table (~1M entries, depth-preferred replacement, EXACT/LOWER/UPPER bounds).
- **Move ordering**: TT/hash move, MVV-LVA captures, killer moves, **history
  heuristic** (`[from][to]` table accumulated on quiet cutoffs).
- **Principal-Variation Search (PVS)** + **Late-Move Reductions (LMR)** with a
  full-depth re-search safety net.
- **Aspiration windows** around the previous iteration's score, widening on a
  fail.
- **Null-move pruning** (R=2, depth ≥ 3), guarded against check and zugzwang.
- **Check extensions** (ply-bounded).
- **Quiescence**: stand-pat with beta cutoff, full evasions in check, captures
  pruned by **Static Exchange Evaluation (SEE)**, and non-losing **quiet checks
  at the horizon** so short forcing tactics aren't truncated.
- Hard recursion cap so check sequences can never overflow the stack.
- **Time management**: `go movetime`/`go wtime btime`/`go depth` all supported;
  iterative deepening stops at the deadline and returns the best move from the
  last completed depth.

### NNUE eval

- **HalfKP** features (40,960 per perspective), shared feature transformer,
  small dense head (`2·256 → 32 → 32 → 1`) with clipped-ReLU.
- **Int8 / int16 quantized inference** (Stockfish-style scheme): feature
  transformer weights int16 (scale 127), accumulator int16, clipped-ReLU output
  int8, dense weights int8 (scale 64), bias int32. ~5× faster than the
  equivalent float32 path; weights file is half the size (21 MB).
- **Incremental accumulator** maintained across make/unmake: a non-king move
  toggles the changed feature rows; a king move refreshes that side's
  accumulator (kings re-key every HalfKP feature). The standard scheme.
- Weights exported by `scripts/export_nnue.py` (PTQ from the existing float
  checkpoint, or use `--qat` in training for cleaner quantization). Validated
  against the Python evaluator: int8 leaf evals match within ±12 cp on test
  positions (essentially noise vs. the 150 cp blunder threshold).
- Binary format magic: `NNU2`. Re-export old `.pt` checkpoints to use the
  current engine.

### UCI

Standard `uci`/`isready`/`ucinewgame`/`position`/`go`/`stop`/`quit`. Custom
option `NnuePath` to load the exported weights.

### Run / test

```bash
cd engine-rs
cargo build --release
cargo test --release   # perft + search + SEE + incremental-accumulator validation

# Standalone perft + search-demo entry point:
./target/release/chessbot-engine bench checkpoints/nnue_combined.bin

# Or as a UCI engine (the normal interface):
./target/release/chessbot-engine
```

Square indexing is Little-Endian Rank-File (a1 = 0 … h8 = 63), matching the
Python NNUE feature layout so trained weights transfer directly with no
remapping.

## Measured strength

Against Stockfish capped to ~1500 UCI Elo at 0.1s/move (small-sample matches
have a wide error bar — roughly ±150 Elo at 20 games — so the W/D/L number
should be read as directional):

- A 20-game match scored ~60–70% (≈ +100 to +150 Elo relative to Stockfish at
  that setting).
- The losses are predominantly **tactical** by `analyze_blunders.py`: ~83% of
  total centipawn loss in losing games comes from individual blunders (single
  moves losing ≥150 cp). That's the depth-limited signature — not the eval's
  fault, the search just can't see deep enough at 0.1s.

The search optimizations brought ACPL (in losing games) from ~201 to ~157 (−22%)
versus the baseline before SEE / PVS / LMR / history / aspiration landed. At
longer time controls the engine reaches deeper and tactical blunders drop
further. See the diagnostic scripts below to re-measure.

## Python engine (`bot/`)

The reference implementation: bitless mailbox board, alpha-beta search with the
same Python-side optimizations (null-move pruning, killer moves, check
extensions, quiescence, TT). It's slow — hundreds of nodes/sec, the
interpreter — but it's the proven oracle for the Rust port and supports the
swappable Python NNUE evaluator for experiments.

```python
from bot import search
from bot.nnue.evaluator import NNUEEvaluator

nnue = NNUEEvaluator.from_checkpoint("checkpoints/nnue_combined.pt")
search.set_evaluator(nnue.evaluate_for_side_to_move)
# search.reset_evaluator() goes back to material
```

Unit tests + perft validation:

```bash
python -m pytest tests/
```

## NNUE training pipeline

The repo doesn't ship a trained model or training data (both gitignored — too
large for GitHub, both re-derivable). To produce a model and load it into the
Rust engine:

```bash
# 1. Download a monthly PGN dump (.pgn.zst) from https://database.lichess.org
#    (standard rated; many games have Stockfish [%eval] annotations). OR use
#    Lichess's evaluations DB (deeper, cleaner labels, no game outcome).

# 2. Extract training positions
python scripts/extract_lichess.py \
    --input ~/Downloads/lichess_db_standard_rated_YYYY-MM.pgn.zst \
    --output data/positions.npz \
    --positions-per-game 4 --max-positions 10000000

# 3. (Optional) merge multiple datasets for better distribution + labels
python scripts/merge_datasets.py data/positions.npz data/eval_positions.npz \
    --output data/combined.npz

# 4. Train (use --wdl-lambda 0.0 for human-game data — the win/loss signal at
#    amateur level is noisy and flattens material; pure Stockfish-eval target
#    respects material monotonically).
python scripts/train_nnue.py \
    --data data/combined.npz --output checkpoints/nnue_combined.pt \
    --hidden 256 --epochs 20 --batch-size 8192 --lr 1e-3 --wdl-lambda 0.0

# 5. Export to the quantized int8/int16 binary the Rust engine loads.
#    For cleaner quantization, retrain with --qat in step 4.
python scripts/export_nnue.py \
    --checkpoint checkpoints/nnue_combined.pt \
    --output checkpoints/nnue_combined.bin
```

The full set of tools in `scripts/`:

| Script | Purpose |
|--------|---------|
| `extract_lichess.py` | Lichess game-PGN dump → training positions (good material distribution; shallow inline evals). |
| `extract_eval_db.py` | Lichess **evaluations** DB → positions (deep clean Stockfish evals, no game outcome). Sign-safety check included. |
| `merge_datasets.py` | Concatenate datasets — combine game + eval-DB for distribution + label quality. |
| `train_nnue.py` | Train the NNUE (eval/WDL blend loss; `--wdl-lambda` controls the mix). |
| `export_nnue.py` | Export `.pt` → `.bin` for the Rust engine. |
| `validate_nnue.py` | Held-out Pearson correlation between the model and its labels — the cheap quality gate. |
| `diagnose_nnue.py` | Material and positional sanity check on a checkpoint. |
| `ab_play.py` | Python head-to-head: NNUE vs material, or NNUE vs another NNUE. |
| `sanity_vs_random.py` | Functional check — does the eval beat random play? |
| `play_engine.py` | Play the Rust engine in the terminal (drives it via python-chess UCI). |
| `engine_match.py` | Strength test — Rust engine vs Stockfish (or self-play). Writes PGN. |
| `analyze_blunders.py` | Per-move centipawn-loss analysis with a tactical-vs-positional verdict (needs local Stockfish). |

**Lessons baked into these tools:**
- `--wdl-lambda 0.0` for human-game data — the win/loss signal at amateur
  rating levels is too noisy and flattens the material gradient.
- Eval-DB labels are deeper but skew toward near-equal positions. *Merging*
  game data + eval-DB gives the best of both: material variety + clean labels.
- A learned eval that correlates ~0.8 with Stockfish is roughly material-strength
  at shallow depth — its positional knowledge only becomes a decisive edge
  with the depth the Rust engine provides.

## GUI app (`app/`)

The Tkinter GUI launches the Rust UCI engine as a subprocess and drives it via
`python-chess`. Analysis runs on a **background thread** with a result queue
that the UI polls (25 ms), so the board stays fully responsive while the engine
thinks. A generation token discards stale results (board changed, new game).
Falls back to the Python search if the Rust binary isn't built. Defaults to
depth 6 with a 3-second safety cap; depth is editable in the controls.

```bash
python app/main.py
```

## NNUE evaluator (`bot/nnue/`) — Python reference

The Python implementation of the same network, used for training and for
validating that the Rust inference matches:

- **Features** (`features.py`): HalfKP — 40,960 sparse features per perspective.
- **Model** (`model.py`): shared feature transformer (`nn.EmbeddingBag`,
  summing active features into a per-perspective accumulator), two accumulators
  concatenated, small dense head (`2·hidden → 32 → 32 → 1`) with clipped-ReLU.
  Default hidden = 256.
- **Inference** (`evaluator.py`): `NNUEEvaluator` mirrors the engine's
  `evaluate(board, perspective)` API; recomputes the accumulator from scratch
  each call (the Rust side is the fast one).

The Rust inference is bit-for-bit equivalent on the same weights — verified by
running both on identical FENs and checking the eval matches exactly.

## Development

- Python 3.10+ (`list[X]`, `X | Y` unions used throughout).
- Rust: stable, built with 1.94. `cargo build`/`cargo test` from `engine-rs/`.

## Roadmap

**Search refinements (incremental):**
- Repetition + 50-move draw detection in search (currently the search doesn't
  track position history — can misplay repeatable/drawn positions).
- Proper mate-score adjustment in the TT (currently simplistic).
- Lazy SMP (multi-threaded search) — big single-machine speedup.
- Magic bitboards for faster sliding-piece move generation.

**NNUE improvements (lower priority — eval isn't the current bottleneck):**
- Self-play training data (the Stockfish bootstrapping loop): now that the
  engine is fast, it can generate its own labeled positions.
- Bigger network (hidden 512/1024); HalfKAv2 + output buckets (game-phase-aware
  sub-nets).

**Python engine (lowest priority — it's a reference, not a competitor):**
- Aspiration windows, LMR, SEE; piece-square tables for the material eval.
