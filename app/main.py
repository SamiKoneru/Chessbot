"""Small Tkinter application for playing with the chessbot engine.

Run from the project root with:
    python3 app/main.py
"""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from time import perf_counter
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.board import Board
from bot.coordinates import square_name
from bot.enums import Color, PieceType
from bot.evaluation import CHECKMATE_SCORE
from bot.move import Move
from bot.search import SearchResult, alpha_beta_search
from bot.transposition_table import TranspositionTable

PLAYER_VS_COMPUTER_MODE = "Player vs Computer"
TESTING_MODE = "Testing"
AUTO_MOVE_DELAY_MS = 75

# Paths to the fast Rust UCI engine and its exported NNUE weights.
RUST_ENGINE_PATH = PROJECT_ROOT / "engine-rs" / "target" / "release" / "chessbot-engine"
RUST_NNUE_PATH = PROJECT_ROOT / "checkpoints" / "nnue_combined.bin"
# Safety cap on engine think time per position (also bounds the Python fallback freeze).
MAX_THINK_SECONDS = 3.0
# How often the UI polls for finished background analyses (ms).
POLL_MS = 25

LIGHT_SQUARE = "#f3e6c8"
DARK_SQUARE = "#b58863"
SELECTED_LIGHT_SQUARE = "#dec5ab"
SELECTED_DARK_SQUARE = "#8c6540"
TARGET_SQUARE = "#cde9b0"

PIECE_TO_TEXT = {
    (Color.WHITE, PieceType.PAWN): "♙",
    (Color.WHITE, PieceType.KNIGHT): "♘",
    (Color.WHITE, PieceType.BISHOP): "♗",
    (Color.WHITE, PieceType.ROOK): "♖",
    (Color.WHITE, PieceType.QUEEN): "♕",
    (Color.WHITE, PieceType.KING): "♔",
    (Color.BLACK, PieceType.PAWN): "♟",
    (Color.BLACK, PieceType.KNIGHT): "♞",
    (Color.BLACK, PieceType.BISHOP): "♝",
    (Color.BLACK, PieceType.ROOK): "♜",
    (Color.BLACK, PieceType.QUEEN): "♛",
    (Color.BLACK, PieceType.KING): "♚",
}

PROMOTION_CHOICES = {
    "q": PieceType.QUEEN,
    "r": PieceType.ROOK,
    "b": PieceType.BISHOP,
    "n": PieceType.KNIGHT,
}


class ChessbotApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Chessbot App")
        self.geometry("1180x760")
        self.minsize(1080, 720)

        self.board = Board.starting_position()
        self.selected_square: str | None = None
        self.thinking = False
        self.move_history: list[str] = []
        self.square_buttons: dict[str, tk.Label] = {}
        self.analysis_cache: tuple[str, int, SearchResult] | None = None
        self.eval_display_cache: tuple[str, int, SearchResult] | None = None
        self.transposition_table = TranspositionTable()
        self.auto_move_job: str | None = None

        # Use the fast Rust UCI engine (with its NNUE) for play/analysis if it's
        # built; otherwise fall back to the built-in Python search. The Rust engine
        # is thousands of times faster, so it gets a much higher default depth.
        self.rust_engine = None
        self.eval_label = self._load_engine()
        default_depth = "6" if self.rust_engine is not None else "4"

        # Background-analysis plumbing (Rust path only): a worker thread runs the
        # blocking engine query and posts results to this queue; the UI polls it on
        # the main thread. `_gen` is an invalidation token bumped whenever in-flight
        # analyses should be discarded (board change, new game, mode/color change).
        self._result_q: queue.Queue = queue.Queue()
        self._engine_lock = threading.Lock()
        self._gen = 0
        self._pending_move_info = None  # (gen, fen, depth, actor, start_time) or None
        self._eval_pending_gen = None

        self.mode_var = tk.StringVar(value=PLAYER_VS_COMPUTER_MODE)
        self.human_color_var = tk.StringVar(value="White")
        self.single_depth_var = tk.StringVar(value=default_depth)
        self.white_depth_var = tk.StringVar(value=default_depth)
        self.black_depth_var = tk.StringVar(value=default_depth)
        self.show_eval_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value=f"Ready — engine eval: {self.eval_label}")
        self.eval_label_var = tk.StringVar(value="Eval +0.00")
        self.selection_var = tk.StringVar(value="Selected: none")

        self._build_ui()
        self._refresh_controls()
        self._refresh_view()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(POLL_MS, self._poll_engine_results)

    def _load_engine(self) -> str:
        """Launch the fast Rust UCI engine if it's built; else fall back to the
        Python search. Sets self.rust_engine and returns a status label.
        """
        self.rust_engine = None
        if not RUST_ENGINE_PATH.exists():
            return "Python search (build Rust: cd engine-rs && cargo build --release)"
        try:
            import chess.engine

            engine = chess.engine.SimpleEngine.popen_uci(str(RUST_ENGINE_PATH))
            suffix = ""
            if RUST_NNUE_PATH.exists():
                try:
                    engine.configure({"NnuePath": str(RUST_NNUE_PATH)})
                    suffix = " + NNUE"
                except Exception:  # noqa: BLE001
                    pass
            self.rust_engine = engine
            return f"Rust engine{suffix}"
        except Exception as exc:  # noqa: BLE001 - missing python-chess / launch failure
            return f"Python search (Rust launch failed: {exc})"

    def _on_close(self) -> None:
        self._cancel_pending_analysis()
        if self.rust_engine is not None:
            try:
                self.rust_engine.quit()
            except Exception:  # noqa: BLE001
                pass
        self.destroy()

    # --- Background analysis (Rust path) ------------------------------------

    def _cancel_pending_analysis(self) -> None:
        """Invalidate any in-flight analysis so its result is ignored on arrival."""
        self._gen += 1
        self._pending_move_info = None
        self._eval_pending_gen = None
        self.thinking = False

    def _spawn_analysis(self, gen: int, fen: str, depth: int, purpose: str) -> None:
        """Run an engine analysis on a daemon thread; post the result to the queue.
        The worker never touches Tk or the board — only the engine and `fen`."""
        engine = self.rust_engine

        def work() -> None:
            raw = None
            with self._engine_lock:
                if engine is not None:
                    try:
                        raw = self._engine_analyse_raw(engine, fen, depth)
                    except Exception:  # noqa: BLE001 - engine died / protocol error
                        try:
                            engine.quit()
                        except Exception:  # noqa: BLE001
                            pass
                        self.rust_engine = None
                        raw = None
            self._result_q.put((gen, fen, depth, purpose, raw))

        threading.Thread(target=work, daemon=True).start()

    def _engine_analyse_raw(self, engine, fen: str, depth: int):
        """Returns (best_uci, score_stm_relative, nodes). Runs on the worker thread."""
        import chess
        import chess.engine

        pyboard = chess.Board(fen)
        limit = chess.engine.Limit(depth=depth, time=MAX_THINK_SECONDS)
        info = engine.analyse(pyboard, limit)
        score = info["score"].pov(pyboard.turn).score(mate_score=2 * CHECKMATE_SCORE)
        nodes = int(info.get("nodes", 0) or 0)
        pv = info.get("pv") or []
        best_uci = pv[0].uci() if pv else None
        return (best_uci, score, nodes)

    def _poll_engine_results(self) -> None:
        try:
            while True:
                gen, fen, depth, purpose, raw = self._result_q.get_nowait()
                if purpose == "move":
                    self._on_move_ready(gen, fen, depth, raw)
                else:
                    self._on_eval_ready(gen, fen, depth, raw)
        except queue.Empty:
            pass
        self.after(POLL_MS, self._poll_engine_results)

    def _on_move_ready(self, gen: int, fen: str, depth: int, raw) -> None:
        info = self._pending_move_info
        if info is None or gen != info[0]:
            return  # superseded request; ignore (don't disturb a newer pending move)
        self.thinking = False
        self._pending_move_info = None
        if gen != self._gen:
            self._refresh_view()
            return  # board changed after dispatch; discard
        _g, _f, _d, actor, started = info
        if raw is None or raw[0] is None:
            self._set_status("No move available.")
            self._refresh_view()
            return
        best_uci, score, nodes = raw
        move = self._uci_to_app_move(best_uci)
        if move is None:
            self._set_status(f"Engine returned an unexpected move: {best_uci}")
            self._refresh_view()
            return
        elapsed = perf_counter() - started
        self._apply_move(move, actor=actor, refresh=False)
        self._set_status(
            f"{actor} played {best_uci} at depth {depth} "
            f"(score {score}, nodes {nodes}, {elapsed:.2f}s)."
        )
        self._refresh_view()
        self._schedule_auto_move_if_needed()

    def _on_eval_ready(self, gen: int, fen: str, depth: int, raw) -> None:
        if gen != self._gen or raw is None or raw[0] is None:
            return
        self._eval_pending_gen = None
        best_uci, score, nodes = raw
        result = SearchResult(score=score, best_move=self._uci_to_app_move(best_uci), nodes_searched=nodes)
        self.eval_display_cache = (fen, depth, result)
        if self.show_eval_var.get():
            self._draw_eval_bar(result, fen)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=12)
        controls.grid(row=0, column=0, sticky="ns")

        board_area = ttk.Frame(self, padding=(0, 12, 12, 12))
        board_area.grid(row=0, column=1, sticky="nsew")
        board_area.columnconfigure(0, weight=0)
        board_area.columnconfigure(1, weight=0)
        board_area.columnconfigure(2, weight=1)
        board_area.rowconfigure(0, weight=1)

        ttk.Label(controls, text="Mode").grid(row=0, column=0, sticky="w")
        mode_picker = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=(PLAYER_VS_COMPUTER_MODE, TESTING_MODE),
            state="readonly",
            width=18,
        )
        mode_picker.grid(row=1, column=0, sticky="ew", pady=(4, 12))
        mode_picker.bind("<<ComboboxSelected>>", lambda _event: self._refresh_controls())

        self.human_controls = ttk.LabelFrame(controls, text="Player vs Computer", padding=10)
        self.human_controls.grid(row=2, column=0, sticky="ew")
        ttk.Label(self.human_controls, text="Human plays").grid(row=0, column=0, sticky="w")
        human_color_picker = ttk.Combobox(
            self.human_controls,
            textvariable=self.human_color_var,
            values=("White", "Black"),
            state="readonly",
            width=10,
        )
        human_color_picker.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        human_color_picker.bind("<<ComboboxSelected>>", lambda _event: self._on_human_color_changed())

        ttk.Label(self.human_controls, text="Bot depth").grid(row=2, column=0, sticky="w")
        ttk.Entry(self.human_controls, textvariable=self.single_depth_var, width=10).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(4, 0),
        )

        self.self_play_controls = ttk.LabelFrame(controls, text="Testing", padding=10)
        self.self_play_controls.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(self.self_play_controls, text="White depth").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.self_play_controls, textvariable=self.white_depth_var, width=10).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(4, 10),
        )
        ttk.Label(self.self_play_controls, text="Black depth").grid(row=2, column=0, sticky="w")
        ttk.Entry(self.self_play_controls, textvariable=self.black_depth_var, width=10).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(4, 0),
        )

        ttk.Checkbutton(
            controls,
            text="Show evaluation bar",
            variable=self.show_eval_var,
            command=self._refresh_eval_bar_visibility,
        ).grid(row=4, column=0, sticky="w", pady=(12, 0))

        ttk.Button(controls, text="New Game", command=self._new_game).grid(
            row=5,
            column=0,
            sticky="ew",
            pady=(12, 6),
        )
        self.generate_button = ttk.Button(
            controls,
            text="Generate Move",
            command=self._handle_generate_move,
        )
        self.generate_button.grid(row=6, column=0, sticky="ew", pady=6)
        ttk.Button(controls, text="Clear Selection", command=self._clear_selection).grid(
            row=7,
            column=0,
            sticky="ew",
            pady=6,
        )

        ttk.Label(controls, textvariable=self.selection_var, wraplength=220).grid(
            row=8,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )

        ttk.Label(controls, textvariable=self.status_var, wraplength=220).grid(
            row=9,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )

        board_frame = tk.Frame(board_area, bg="#3a3025", padx=6, pady=6)
        board_frame.grid(row=0, column=0, sticky="n")
        for row in range(8):
            board_frame.rowconfigure(row, weight=1)
            board_frame.columnconfigure(row, weight=1)

        for rank in range(8):
            for file in range(8):
                square = square_name(rank, file)
                button = tk.Label(
                    board_frame,
                    text="",
                    font=("Arial Unicode MS", 28),
                    width=3,
                    height=1,
                    bd=0,
                    relief="flat",
                    anchor="center",
                )
                button.grid(row=rank, column=file, sticky="nsew", padx=0, pady=0)
                button.bind("<Button-1>", lambda _event, current_square=square: self._on_square_clicked(current_square))
                self.square_buttons[square] = button

        self.eval_frame = ttk.Frame(board_area, padding=(12, 0))
        self.eval_frame.grid(row=0, column=1, sticky="ns")
        ttk.Label(self.eval_frame, text="Evaluation").grid(row=0, column=0, pady=(0, 6))
        self.eval_canvas = tk.Canvas(self.eval_frame, width=42, height=420, bg="#e5e5e5", highlightthickness=1)
        self.eval_canvas.grid(row=1, column=0)
        ttk.Label(self.eval_frame, textvariable=self.eval_label_var).grid(row=2, column=0, pady=(8, 0))

        history_frame = ttk.LabelFrame(board_area, text="Move Log", padding=10)
        history_frame.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        history_frame.rowconfigure(0, weight=1)
        history_frame.columnconfigure(0, weight=1)
        self.move_log = tk.Listbox(history_frame, font=("Menlo", 12), width=38)
        self.move_log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.move_log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.move_log.configure(yscrollcommand=scrollbar.set)

    def _refresh_controls(self) -> None:
        player_mode = self._is_player_vs_computer_mode()
        if player_mode:
            self.human_controls.grid()
            self.self_play_controls.grid_remove()
            self.generate_button.configure(text="Bot Plays Automatically", state="disabled")
        else:
            self.human_controls.grid_remove()
            self.self_play_controls.grid()
            self.generate_button.configure(text="Bot Move", state="normal")
        self.selected_square = None
        self._cancel_auto_move()
        self._cancel_pending_analysis()
        self._refresh_eval_bar_visibility()
        self._reset_eval_display()
        self._refresh_view()
        self._schedule_auto_move_if_needed()

    def _refresh_eval_bar_visibility(self) -> None:
        if self.show_eval_var.get():
            self.eval_frame.grid()
        else:
            self.eval_frame.grid_remove()

    def _new_game(self) -> None:
        self._cancel_auto_move()
        self._cancel_pending_analysis()
        self.board = Board.starting_position()
        self.selected_square = None
        self.thinking = False
        self.analysis_cache = None
        self.eval_display_cache = None
        self.transposition_table.clear()
        self.move_history.clear()
        self.move_log.delete(0, tk.END)
        self._set_status("New game started.")
        self._reset_eval_display()
        self._refresh_view()
        self._schedule_auto_move_if_needed()

    def _clear_selection(self) -> None:
        self.selected_square = None
        self._set_status("Selection cleared.")
        self._refresh_view()

    def _handle_generate_move(self) -> None:
        if self._is_player_vs_computer_mode():
            self._set_status("Bot moves automatically in Player vs Computer mode.")
            return
        self._make_engine_move(use_cached_analysis=True)

    def _on_square_clicked(self, square: str) -> None:
        if self.thinking:
            self._set_status("The engine is thinking.")
            return
        if self.board.is_game_over():
            self._set_status("The game is over. Start a new game to continue.")
            return
        if self._is_player_vs_computer_mode() and self._current_human_color() is not self.board.state.side_to_move:
            self._set_status("Wait for the bot to move.")
            return

        piece = self.board.piece_at(square)
        active_human_color = self._active_human_color()
        if self.selected_square is None:
            if piece is None or piece.color is not active_human_color:
                self._set_status("Select one of your own pieces.")
                return
            self.selected_square = square
            self._set_status(f"Selected {square}.")
            self._refresh_view()
            return

        if square == self.selected_square:
            self._clear_selection()
            return

        if piece is not None and piece.color is active_human_color:
            self.selected_square = square
            self._set_status(f"Selected {square}.")
            self._refresh_view()
            return

        move = self._build_move(self.selected_square, square)
        if move is None:
            return
        if not self.board.is_legal_move(move):
            self._set_status(f"Illegal move: {move.uci}")
            return

        self._apply_move(move, actor="Human", refresh=False)
        self.selected_square = None
        if self._is_player_vs_computer_mode() and self.rust_engine is None:
            # Rust path updates the eval bar asynchronously via _refresh_eval_bar.
            self._update_eval_display_for_current_position()
        self._refresh_view()
        self._schedule_auto_move_if_needed()

    def _build_move(self, from_square: str, to_square: str) -> Move | None:
        piece = self.board.piece_at(from_square)
        if piece is None:
            self._set_status(f"No piece on {from_square}.")
            return None

        promotion = None
        if piece.piece_type is PieceType.PAWN and to_square[1] in {"1", "8"}:
            choice = simpledialog.askstring(
                "Promotion",
                "Promote to (q, r, b, n). Leave blank for queen.",
                parent=self,
            )
            if choice:
                promotion = PROMOTION_CHOICES.get(choice.strip().lower())
                if promotion is None:
                    messagebox.showwarning("Promotion", "Invalid promotion piece. Defaulting to queen.")
            if promotion is None:
                promotion = PieceType.QUEEN

        return Move(from_square, to_square, promotion=promotion)

    def _apply_move(self, move: Move, actor: str, *, refresh: bool = True) -> None:
        side = self.board.state.side_to_move.value
        self.board.apply_move(move, validate_legality=True)
        self._gen += 1  # board changed: invalidate in-flight analyses
        self.analysis_cache = None
        self.selected_square = None
        entry = f"{len(self.move_history) + 1:>2}. {actor:<5} {side:<5} {move.uci}"
        self.move_history.append(entry)
        self.move_log.insert(tk.END, entry)
        self.move_log.yview_moveto(1.0)
        if refresh:
            self._refresh_view()

    def _make_engine_move(self, *, use_cached_analysis: bool) -> None:
        if self.thinking:
            return
        if self.board.is_game_over():
            self._set_status("The game is already over.")
            return
        self._cancel_auto_move()

        depth = self._current_engine_depth()
        actor = f"{self.board.state.side_to_move.value.title()} bot" if self._is_testing_mode() else "Bot"

        if self.rust_engine is not None:
            # Non-blocking: dispatch to the worker; the move is applied in
            # _on_move_ready when the result arrives. The UI stays responsive.
            self.thinking = True
            self._pending_move_info = (self._gen, self.board.to_fen(), depth, actor, perf_counter())
            self._set_status(f"{actor} thinking at depth {depth}...")
            self._spawn_analysis(self._gen, self.board.to_fen(), depth, "move")
            return

        # Synchronous fallback (built-in Python search). Fast enough to block briefly.
        self.thinking = True
        self._set_status(f"{actor} thinking at depth {depth}...")
        self.update_idletasks()
        started = perf_counter()
        result = self._get_cached_analysis() if use_cached_analysis else self._analyze_current_position(depth)
        elapsed = perf_counter() - started
        self.thinking = False
        if result.best_move is None:
            self._set_status(f"No legal moves available after {elapsed:.2f}s.")
            self._refresh_view()
            return
        self._apply_move(result.best_move, actor=actor, refresh=False)
        self._set_status(
            f"{actor} played {result.best_move.uci} at depth {depth} "
            f"(score {result.score}, nodes {result.nodes_searched}, {elapsed:.2f}s)."
        )
        self._refresh_view()

    def _refresh_view(self) -> None:
        legal_targets = self._selected_legal_targets()
        for rank in range(8):
            for file in range(8):
                square = square_name(rank, file)
                button = self.square_buttons[square]
                piece = self.board.piece_at(square)
                default_color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
                color = default_color
                if square == self.selected_square:
                    color = SELECTED_LIGHT_SQUARE if (rank + file) % 2 == 0 else SELECTED_DARK_SQUARE
                elif square in legal_targets:
                    color = TARGET_SQUARE

                button.configure(
                    text=self._piece_text(piece),
                    bg=color,
                    fg="#1f1f1f",
                    relief="flat",
                    bd=0,
                )

        self.selection_var.set(self._selection_text())
        self._refresh_eval_bar()
        if self.board.is_checkmate():
            winner = self.board.state.side_to_move.opposite.value.title()
            self._set_status(f"Checkmate. {winner} wins.")
        elif self.board.is_stalemate():
            self._set_status("Stalemate.")
        elif not self.thinking:
            current = self.board.state.side_to_move.value.title()
            self._set_status(f"{current} to move.")

    def _refresh_eval_bar(self) -> None:
        if not self.show_eval_var.get():
            return

        fen = self.board.to_fen()
        depth = self._current_engine_depth()

        if self.rust_engine is None:
            # Synchronous fallback (fast Python search): original behavior.
            eval_context = self._current_eval_context()
            if eval_context is None:
                self.eval_label_var.set("Eval --")
                self.eval_canvas.delete("all")
                return
            analysis_fen, _depth, analysis = eval_context
            self._draw_eval_bar(analysis, analysis_fen)
            return

        # Rust path: draw from cache if it matches the current position, else
        # request a background eval (without blocking) and show a placeholder.
        if self.eval_display_cache is not None:
            cached_fen, _cached_depth, cached_result = self.eval_display_cache
            if cached_fen == fen:
                self._draw_eval_bar(cached_result, cached_fen)
                return
        self.eval_label_var.set("Eval …")
        # Don't compute an eval while a move is being searched — the board is
        # about to change, and the move search already occupies the engine.
        if not self.thinking and self._eval_pending_gen != self._gen:
            self._eval_pending_gen = self._gen
            self._spawn_analysis(self._gen, fen, depth, "eval")

    def _draw_eval_bar(self, analysis: SearchResult, fen: str) -> None:
        score = self._score_from_white_perspective(
            analysis.score,
            side_to_move=self._side_to_move_from_fen(fen),
        )
        best_move = None if analysis.best_move is None else analysis.best_move.uci
        self.eval_label_var.set(self._format_eval_label(score, best_move))

        height = 420
        width = 42
        normalized = max(-1.0, min(1.0, score / 1000.0))
        white_ratio = (normalized + 1.0) / 2.0
        white_height = int(height * white_ratio)
        black_height = height - white_height

        self.eval_canvas.delete("all")
        self.eval_canvas.create_rectangle(0, 0, width, black_height, fill="#1f1f1f", outline="")
        self.eval_canvas.create_rectangle(0, black_height, width, height, fill="#f8f8f8", outline="")
        self.eval_canvas.create_rectangle(0, 0, width, height, outline="#666666")

    def _format_eval_label(self, score: int, best_move: str | None) -> str:
        if abs(score) >= CHECKMATE_SCORE:
            mate_text = "+MATE" if score > 0 else "-MATE"
            if best_move is None:
                return f"Eval {mate_text}"
            return f"Eval {mate_text} ({best_move})"

        if best_move is None:
            return f"Eval {score / 100:.2f}"
        return f"Eval {score / 100:.2f} ({best_move})"

    def _get_cached_analysis(self) -> SearchResult:
        depth = self._current_engine_depth()
        fen = self.board.to_fen()
        if self.analysis_cache is not None:
            cached_fen, cached_depth, cached_result = self.analysis_cache
            if cached_fen == fen and cached_depth == depth:
                return cached_result

        result = self._analyze_current_position(depth)
        self.analysis_cache = (fen, depth, result)
        return result

    def _analyze_current_position(self, depth: int) -> SearchResult:
        if self.rust_engine is not None:
            result = self._analyze_with_rust(depth)
            if result is not None:
                self.analysis_cache = (self.board.to_fen(), depth, result)
                return result
        # Fallback: built-in Python search.
        result = alpha_beta_search(
            self.board,
            depth=depth,
            transposition_table=self.transposition_table,
        )
        self.analysis_cache = (self.board.to_fen(), depth, result)
        return result

    def _analyze_with_rust(self, depth: int) -> SearchResult | None:
        """Query the Rust UCI engine. Returns a SearchResult (score side-to-move
        relative, matching the Python search), or None if the engine errors —
        in which case we drop it and fall back to the Python search."""
        try:
            import chess
            import chess.engine

            pyboard = chess.Board(self.board.to_fen())
            limit = chess.engine.Limit(depth=depth, time=MAX_THINK_SECONDS)
            info = self.rust_engine.analyse(pyboard, limit)
            # Score from the side-to-move's perspective (negamax convention).
            score = info["score"].pov(pyboard.turn).score(mate_score=2 * CHECKMATE_SCORE)
            nodes = int(info.get("nodes", 0) or 0)
            pv = info.get("pv") or []
            best_move = self._uci_to_app_move(pv[0].uci()) if pv else None
            return SearchResult(score=score, best_move=best_move, nodes_searched=nodes)
        except Exception:  # noqa: BLE001 - engine died / protocol error
            try:
                self.rust_engine.quit()
            except Exception:  # noqa: BLE001
                pass
            self.rust_engine = None
            return None

    def _uci_to_app_move(self, uci_str: str) -> Move | None:
        for move in self.board.legal_moves():
            if move.uci == uci_str:
                return move
        return None

    def _current_eval_context(self) -> tuple[str, int, SearchResult] | None:
        if self._is_testing_mode():
            depth = self._current_engine_depth()
            return (self.board.to_fen(), depth, self._get_cached_analysis())

        if self.eval_display_cache is None:
            self._update_eval_display_for_current_position()
        return self.eval_display_cache

    def _update_eval_display_for_current_position(self) -> None:
        depth = self._current_engine_depth()
        result = self._get_cached_analysis()
        self.eval_display_cache = (self.board.to_fen(), depth, result)

    def _reset_eval_display(self) -> None:
        self.eval_display_cache = None
        # Rust path repopulates the eval bar asynchronously; only the synchronous
        # Python fallback precomputes it here.
        if self._is_player_vs_computer_mode() and self.rust_engine is None:
            self._update_eval_display_for_current_position()

    def _is_player_vs_computer_mode(self) -> bool:
        return self.mode_var.get() == PLAYER_VS_COMPUTER_MODE

    def _is_testing_mode(self) -> bool:
        return self.mode_var.get() == TESTING_MODE

    def _active_human_color(self) -> Color:
        if self._is_player_vs_computer_mode():
            return self._current_human_color()
        return self.board.state.side_to_move

    def _on_human_color_changed(self) -> None:
        self.selected_square = None
        self._cancel_auto_move()
        self._cancel_pending_analysis()
        self._reset_eval_display()
        self._refresh_view()
        self._schedule_auto_move_if_needed()

    def _schedule_auto_move_if_needed(self) -> None:
        if not self._is_player_vs_computer_mode():
            return
        if self.thinking or self.board.is_game_over():
            return
        if self.board.state.side_to_move is self._current_human_color():
            return
        if self.auto_move_job is not None:
            return
        self.auto_move_job = self.after(AUTO_MOVE_DELAY_MS, self._run_scheduled_auto_move)

    def _run_scheduled_auto_move(self) -> None:
        self.auto_move_job = None
        if not self._is_player_vs_computer_mode():
            return
        if self.board.is_game_over() or self.thinking:
            return
        if self.board.state.side_to_move is self._current_human_color():
            return
        self._make_engine_move(use_cached_analysis=False)

    def _cancel_auto_move(self) -> None:
        if self.auto_move_job is not None:
            self.after_cancel(self.auto_move_job)
            self.auto_move_job = None

    def _score_from_white_perspective(self, score: int, *, side_to_move: Color) -> int:
        if side_to_move is Color.WHITE:
            return score
        return -score

    def _side_to_move_from_fen(self, fen: str) -> Color:
        return Color.WHITE if fen.split()[1] == "w" else Color.BLACK

    def _current_human_color(self) -> Color:
        return Color.WHITE if self.human_color_var.get() == "White" else Color.BLACK

    def _current_engine_depth(self) -> int:
        if self._is_player_vs_computer_mode():
            return self._read_depth(self.single_depth_var, fallback=2)

        if self.board.state.side_to_move is Color.WHITE:
            return self._read_depth(self.white_depth_var, fallback=2)
        return self._read_depth(self.black_depth_var, fallback=2)

    def _read_depth(self, variable: tk.StringVar, fallback: int) -> int:
        raw_value = variable.get().strip()
        try:
            depth = int(raw_value)
        except ValueError:
            depth = fallback
            variable.set(str(fallback))
        return max(0, depth)

    def _piece_text(self, piece) -> str:
        if piece is None:
            return ""
        return PIECE_TO_TEXT[(piece.color, piece.piece_type)]

    def _selection_text(self) -> str:
        if self.selected_square is None:
            return "Selected: none"

        piece = self.board.piece_at(self.selected_square)
        if piece is None:
            return f"Selected: {self.selected_square}"

        return (
            f"Selected: {self.selected_square} "
            f"{PIECE_TO_TEXT[(piece.color, piece.piece_type)]}"
        )

    def _selected_legal_targets(self) -> set[str]:
        if self.selected_square is None:
            return set()
        return {
            move.to_square
            for move in self.board.legal_moves()
            if move.from_square == self.selected_square
        }

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)


def main() -> None:
    ChessbotApp().mainloop()


if __name__ == "__main__":
    main()
