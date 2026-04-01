"""Deterministic Zobrist hashing helpers for board positions.

The hash includes piece placement, side to move, castling rights, and the
en-passant file. It intentionally excludes move clocks because those are
bookkeeping fields rather than move-generation state.
"""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

from .coordinates import FILES, square_index
from .enums import Color, PieceType

if TYPE_CHECKING:
    from .board import Board
    from .piece import Piece
    from .state import CastlingRights

ZOBRIST_SEED = 0x5EEDC0DE

_RANDOM = Random(ZOBRIST_SEED)
_PIECE_TYPES = (
    PieceType.PAWN,
    PieceType.KNIGHT,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.QUEEN,
    PieceType.KING,
)


def _next_key() -> int:
    return _RANDOM.getrandbits(64)


PIECE_KEYS = {
    color: {
        piece_type: tuple(_next_key() for _ in range(64))
        for piece_type in _PIECE_TYPES
    }
    for color in (Color.WHITE, Color.BLACK)
}
SIDE_TO_MOVE_KEY = _next_key()
CASTLING_KEYS = {
    "K": _next_key(),
    "Q": _next_key(),
    "k": _next_key(),
    "q": _next_key(),
}
EN_PASSANT_FILE_KEYS = {
    file_name: _next_key()
    for file_name in FILES
}


def piece_square_hash(piece: "Piece", square: str) -> int:
    rank, file = square_index(square)
    square_id = (rank * 8) + file
    return PIECE_KEYS[piece.color][piece.piece_type][square_id]


def castling_rights_hash(castling_rights: "CastlingRights") -> int:
    key = 0
    if castling_rights.white_kingside:
        key ^= CASTLING_KEYS["K"]
    if castling_rights.white_queenside:
        key ^= CASTLING_KEYS["Q"]
    if castling_rights.black_kingside:
        key ^= CASTLING_KEYS["k"]
    if castling_rights.black_queenside:
        key ^= CASTLING_KEYS["q"]
    return key


def en_passant_hash(target_square: str | None) -> int:
    if target_square is None:
        return 0
    return EN_PASSANT_FILE_KEYS[target_square[0]]


def compute_zobrist_hash(board: "Board") -> int:
    key = 0

    for square, piece in board.iter_pieces():
        key ^= piece_square_hash(piece, square)

    if board.state.side_to_move is Color.BLACK:
        key ^= SIDE_TO_MOVE_KEY

    key ^= castling_rights_hash(board.state.castling_rights)
    key ^= en_passant_hash(board.state.en_passant_target)
    return key
