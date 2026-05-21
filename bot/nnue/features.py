"""HalfKP feature extraction for NNUE.

A HalfKP feature is a triple (own_king_square, piece_square, piece_kind),
indexed from a single side's ("perspective") point of view. The piece kind
encodes both piece type and whether the piece belongs to that perspective
(own) or the opponent. Kings are NOT part of the piece set because the
own king's square already conditions the entire feature.

Index layout:
    feature = own_king_sq * (NUM_SQUARES * NUM_PIECE_KINDS)
            + piece_sq * NUM_PIECE_KINDS
            + piece_kind

For each position we extract two sparse feature lists, one per perspective
(white and black). At inference time, the perspective whose turn it is
("side to move") goes first when we concatenate the two accumulators.
"""

from __future__ import annotations

from ..board import Board
from ..coordinates import square_index
from ..enums import Color, PieceType

NUM_SQUARES = 64
NUM_PIECE_KINDS = 10  # 5 piece types (excl. king) * {own, opp}
FEATURES_PER_PERSPECTIVE = NUM_SQUARES * NUM_SQUARES * NUM_PIECE_KINDS  # 40,960

# Stable order: pawn, knight, bishop, rook, queen.
_PIECE_TYPE_TO_INDEX = {
    PieceType.PAWN: 0,
    PieceType.KNIGHT: 1,
    PieceType.BISHOP: 2,
    PieceType.ROOK: 3,
    PieceType.QUEEN: 4,
}


def _square_to_white_index(square: str) -> int:
    """Convert a square like 'a1' to a 0..63 index, a1=0, h1=7, a8=56, h8=63."""
    rank, file = square_index(square)
    rank_from_white = 7 - rank
    return rank_from_white * 8 + file


def _orient(sq_white: int, perspective: Color) -> int:
    """Flip the square vertically when viewing from black's side."""
    return sq_white if perspective is Color.WHITE else sq_white ^ 56


def _piece_kind(piece_color: Color, piece_type: PieceType, perspective: Color) -> int:
    base = _PIECE_TYPE_TO_INDEX[piece_type]
    is_own = piece_color is perspective
    return base * 2 + (0 if is_own else 1)


def active_features(board: Board, perspective: Color) -> list[int]:
    """Return the sparse feature indices that are active for one perspective."""
    king_sq_white = _square_to_white_index(board.king_square(perspective))
    king_sq = _orient(king_sq_white, perspective)
    stride_king = NUM_SQUARES * NUM_PIECE_KINDS

    features: list[int] = []
    for square, piece in board.iter_pieces():
        if piece.piece_type is PieceType.KING:
            continue
        piece_sq = _orient(_square_to_white_index(square), perspective)
        kind = _piece_kind(piece.color, piece.piece_type, perspective)
        features.append(king_sq * stride_king + piece_sq * NUM_PIECE_KINDS + kind)
    return features


def both_perspectives(board: Board) -> tuple[list[int], list[int]]:
    """Return (white_features, black_features)."""
    return active_features(board, Color.WHITE), active_features(board, Color.BLACK)
