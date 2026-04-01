"""FEN parsing and serialization helpers."""

from __future__ import annotations

from .constants import BOARD_SIZE, FEN_TO_PIECE, PIECE_TO_FEN
from .enums import Color
from .piece import Piece
from .state import BoardState, CastlingRights

BoardGrid = list[list[Piece | None]]


def empty_grid() -> BoardGrid:
    return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def parse_fen(fen: str) -> tuple[BoardGrid, BoardState]:
    parts = fen.split()
    if len(parts) != 6:
        raise ValueError(f"Expected 6 FEN fields, got {len(parts)} in {fen!r}")

    board_part, side_part, castling_part, en_passant_part, halfmove_part, fullmove_part = parts
    rows = board_part.split("/")
    if len(rows) != BOARD_SIZE:
        raise ValueError(f"Expected 8 FEN board rows, got {len(rows)}")

    grid = empty_grid()
    for rank, row in enumerate(rows):
        file = 0
        for symbol in row:
            if symbol.isdigit():
                file += int(symbol)
                continue

            if symbol not in FEN_TO_PIECE:
                raise ValueError(f"Unexpected FEN piece symbol: {symbol!r}")
            if file >= BOARD_SIZE:
                raise ValueError(f"Too many squares in FEN row: {row!r}")

            color, piece_type = FEN_TO_PIECE[symbol]
            grid[rank][file] = Piece(color=color, piece_type=piece_type)
            file += 1

        if file != BOARD_SIZE:
            raise ValueError(f"FEN row does not describe 8 squares: {row!r}")

    state = BoardState(
        side_to_move=Color.from_fen_symbol(side_part),
        castling_rights=CastlingRights.from_fen(castling_part),
        en_passant_target=None if en_passant_part == "-" else en_passant_part,
        halfmove_clock=int(halfmove_part),
        fullmove_number=int(fullmove_part),
    )
    return grid, state


def to_fen(grid: BoardGrid, state: BoardState) -> str:
    row_fragments: list[str] = []
    for row in grid:
        empty_count = 0
        row_text: list[str] = []
        for piece in row:
            if piece is None:
                empty_count += 1
                continue

            if empty_count:
                row_text.append(str(empty_count))
                empty_count = 0
            row_text.append(PIECE_TO_FEN[(piece.color, piece.piece_type)])

        if empty_count:
            row_text.append(str(empty_count))
        row_fragments.append("".join(row_text))

    en_passant = state.en_passant_target or "-"
    return " ".join(
        [
            "/".join(row_fragments),
            state.side_to_move.fen_symbol,
            state.castling_rights.to_fen(),
            en_passant,
            str(state.halfmove_clock),
            str(state.fullmove_number),
        ]
    )
