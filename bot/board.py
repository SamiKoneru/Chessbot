"""Board container and state-transition helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .attacks import (
    attacked_squares_by_color,
    attacked_squares_for_piece,
    find_king_square,
    is_square_attacked,
)
from .constants import STARTING_FEN
from .coordinates import square_index, square_name
from .enums import Color, PieceType
from .fen import BoardGrid, empty_grid, parse_fen, to_fen
from .move import Move
from .move_generation import is_legal_move, legal_moves, normalize_move, pseudo_legal_moves
from .piece import Piece
from .state import BoardState, CastlingRights
from .zobrist import compute_zobrist_hash


def _copy_grid(grid: BoardGrid) -> BoardGrid:
    return [row.copy() for row in grid]


@dataclass(slots=True)
class Board:
    grid: BoardGrid = field(default_factory=empty_grid)
    state: BoardState = field(default_factory=BoardState)
    zobrist_hash: int = field(init=False)

    def __post_init__(self) -> None:
        self._refresh_zobrist_hash()

    @classmethod
    def starting_position(cls) -> "Board":
        return cls.from_fen(STARTING_FEN)

    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        grid, state = parse_fen(fen)
        return cls(grid=grid, state=state)

    def clone(self) -> "Board":
        state = BoardState(
            side_to_move=self.state.side_to_move,
            castling_rights=CastlingRights(
                white_kingside=self.state.castling_rights.white_kingside,
                white_queenside=self.state.castling_rights.white_queenside,
                black_kingside=self.state.castling_rights.black_kingside,
                black_queenside=self.state.castling_rights.black_queenside,
            ),
            en_passant_target=self.state.en_passant_target,
            halfmove_clock=self.state.halfmove_clock,
            fullmove_number=self.state.fullmove_number,
        )
        clone = Board(grid=_copy_grid(self.grid), state=state)
        clone.zobrist_hash = self.zobrist_hash
        return clone

    def to_fen(self) -> str:
        return to_fen(self.grid, self.state)

    def iter_pieces(self, color: Color | None = None) -> Iterator[tuple[str, Piece]]:
        for rank, row in enumerate(self.grid):
            for file, piece in enumerate(row):
                if piece is None:
                    continue
                if color is not None and piece.color is not color:
                    continue
                yield square_name(rank, file), piece

    def attacked_squares(self, color: Color) -> set[str]:
        return attacked_squares_by_color(self, color)

    def attacks_from(self, square: str) -> set[str]:
        piece = self.piece_at(square)
        if piece is None:
            raise ValueError(f"No piece at square {square!r}")
        return attacked_squares_for_piece(self, square, piece)

    def is_square_attacked(self, square: str, by_color: Color) -> bool:
        return is_square_attacked(self, square, by_color)

    def king_square(self, color: Color) -> str:
        return find_king_square(self, color)

    def is_in_check(self, color: Color | None = None) -> bool:
        target_color = self.state.side_to_move if color is None else color
        return self.is_square_attacked(self.king_square(target_color), target_color.opposite)

    def normalize_move(self, move: Move) -> Move:
        return normalize_move(self, move)

    def pseudo_legal_moves(self) -> list[Move]:
        return pseudo_legal_moves(self)

    def legal_moves(self) -> list[Move]:
        return legal_moves(self)

    def is_legal_move(self, move: Move) -> bool:
        return is_legal_move(self, move)

    def has_legal_moves(self) -> bool:
        return bool(self.legal_moves())

    def is_checkmate(self) -> bool:
        return self.is_in_check() and not self.has_legal_moves()

    def is_stalemate(self) -> bool:
        return not self.is_in_check() and not self.has_legal_moves()

    def is_game_over(self) -> bool:
        return self.is_checkmate() or self.is_stalemate()

    def piece_at(self, square: str) -> Piece | None:
        rank, file = square_index(square)
        return self.grid[rank][file]

    def set_piece(self, square: str, piece: Piece | None, *, refresh_hash: bool = True) -> None:
        rank, file = square_index(square)
        self.grid[rank][file] = piece
        if refresh_hash:
            self._refresh_zobrist_hash()

    def remove_piece(self, square: str, *, refresh_hash: bool = True) -> Piece | None:
        existing_piece = self.piece_at(square)
        self.set_piece(square, None, refresh_hash=refresh_hash)
        return existing_piece

    def apply_move(self, move: Move, validate_legality: bool = False) -> None:
        move = self.normalize_move(move)
        if validate_legality and not self.is_legal_move(move):
            raise ValueError(f"Illegal move for current position: {move.uci}")

        piece = self.piece_at(move.from_square)
        if piece is None:
            raise ValueError(f"No piece at source square {move.from_square!r}")
        if piece.color is not self.state.side_to_move:
            raise ValueError("Cannot move a piece belonging to the side not on move")

        captured_piece = self.piece_at(move.to_square)
        if captured_piece is not None and captured_piece.color is piece.color:
            raise ValueError("Cannot capture a piece of the same color")
        if captured_piece is not None and captured_piece.piece_type is PieceType.KING:
            raise ValueError("Cannot capture the opposing king")

        self._clear_en_passant_target()
        self._update_castling_rights_for_departure(move.from_square, piece)

        if captured_piece is not None:
            self._update_castling_rights_for_capture(move.to_square, captured_piece)

        self.set_piece(move.from_square, None, refresh_hash=False)

        if move.is_en_passant:
            self._apply_en_passant_capture(move, piece)
        elif move.is_castling:
            self._apply_castling_rook_move(move, piece)

        moved_piece = piece
        if move.promotion is not None:
            moved_piece = Piece(color=piece.color, piece_type=move.promotion)

        self.set_piece(move.to_square, moved_piece, refresh_hash=False)
        self._update_state_after_move(move, piece, captured_piece)
        self._refresh_zobrist_hash()

    def _apply_en_passant_capture(self, move: Move, piece: Piece) -> None:
        target_rank, target_file = square_index(move.to_square)
        direction = 1 if piece.color is Color.WHITE else -1
        captured_square = square_name(target_rank + direction, target_file)
        self.set_piece(captured_square, None, refresh_hash=False)

    def _apply_castling_rook_move(self, move: Move, piece: Piece) -> None:
        rook_from, rook_to = self._castle_rook_squares(move.to_square, piece.color)
        rook = self.piece_at(rook_from)
        if rook is None or rook.piece_type is not PieceType.ROOK:
            raise ValueError("Castling move requires a rook on the correct starting square")
        self.set_piece(rook_from, None, refresh_hash=False)
        self.set_piece(rook_to, rook, refresh_hash=False)

    def _castle_rook_squares(self, king_destination: str, color: Color) -> tuple[str, str]:
        if color is Color.WHITE:
            mapping = {"g1": ("h1", "f1"), "c1": ("a1", "d1")}
        else:
            mapping = {"g8": ("h8", "f8"), "c8": ("a8", "d8")}
        try:
            return mapping[king_destination]
        except KeyError as exc:
            raise ValueError(f"Unsupported castling destination: {king_destination!r}") from exc

    def _clear_en_passant_target(self) -> None:
        self.state.en_passant_target = None

    def _update_castling_rights_for_departure(self, square: str, piece: Piece) -> None:
        rights = self.state.castling_rights
        if piece.piece_type is PieceType.KING:
            if piece.color is Color.WHITE:
                rights.white_kingside = False
                rights.white_queenside = False
            else:
                rights.black_kingside = False
                rights.black_queenside = False
        elif piece.piece_type is PieceType.ROOK:
            if square == "h1":
                rights.white_kingside = False
            elif square == "a1":
                rights.white_queenside = False
            elif square == "h8":
                rights.black_kingside = False
            elif square == "a8":
                rights.black_queenside = False

    def _update_castling_rights_for_capture(self, square: str, piece: Piece) -> None:
        if piece.piece_type is not PieceType.ROOK:
            return

        rights = self.state.castling_rights
        if square == "h1":
            rights.white_kingside = False
        elif square == "a1":
            rights.white_queenside = False
        elif square == "h8":
            rights.black_kingside = False
        elif square == "a8":
            rights.black_queenside = False

    def _update_state_after_move(
        self,
        move: Move,
        piece: Piece,
        captured_piece: Piece | None,
    ) -> None:
        self._update_en_passant_target(move, piece)

        if piece.piece_type is PieceType.PAWN or captured_piece is not None or move.is_en_passant:
            self.state.halfmove_clock = 0
        else:
            self.state.halfmove_clock += 1

        if self.state.side_to_move is Color.BLACK:
            self.state.fullmove_number += 1
        self.state.side_to_move = self.state.side_to_move.opposite

    def _update_en_passant_target(self, move: Move, piece: Piece) -> None:
        if piece.piece_type is not PieceType.PAWN:
            return

        from_rank, from_file = square_index(move.from_square)
        to_rank, _ = square_index(move.to_square)
        if abs(to_rank - from_rank) != 2:
            return

        middle_rank = (from_rank + to_rank) // 2
        self.state.en_passant_target = square_name(middle_rank, from_file)

    def _refresh_zobrist_hash(self) -> None:
        self.zobrist_hash = compute_zobrist_hash(self)
