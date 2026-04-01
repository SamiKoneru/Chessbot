"""Small Tkinter application for playing with the chessbot engine.

Run from the project root with:
    python3 app/main.py
"""

from __future__ import annotations

import sys
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

        self.mode_var = tk.StringVar(value=PLAYER_VS_COMPUTER_MODE)
        self.human_color_var = tk.StringVar(value="White")
        self.single_depth_var = tk.StringVar(value="2")
        self.white_depth_var = tk.StringVar(value="2")
        self.black_depth_var = tk.StringVar(value="2")
        self.show_eval_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready.")
        self.eval_label_var = tk.StringVar(value="Eval +0.00")
        self.selection_var = tk.StringVar(value="Selected: none")

        self._build_ui()
        self._refresh_controls()
        self._refresh_view()

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
        if self._is_player_vs_computer_mode():
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
        self.thinking = True
        if self._is_testing_mode():
            actor = f"{self.board.state.side_to_move.value.title()} bot"
        else:
            actor = "Bot"

        self._set_status(f"{actor} thinking at depth {depth}...")
        self.update_idletasks()

        started = perf_counter()
        if use_cached_analysis:
            result = self._get_cached_analysis()
        else:
            result = self._analyze_current_position(depth)
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

        eval_context = self._current_eval_context()
        if eval_context is None:
            self.eval_label_var.set("Eval --")
            self.eval_canvas.delete("all")
            return

        analysis_fen, _depth, analysis = eval_context
        score = self._score_from_white_perspective(
            analysis.score,
            side_to_move=self._side_to_move_from_fen(analysis_fen),
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
        result = alpha_beta_search(
            self.board,
            depth=depth,
            transposition_table=self.transposition_table,
        )
        self.analysis_cache = (self.board.to_fen(), depth, result)
        return result

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
        if self._is_player_vs_computer_mode():
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
