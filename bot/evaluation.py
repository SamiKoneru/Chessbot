"""Very small evaluation model for the current board.

The scoring convention is:
- positive values are good for the chosen perspective
- negative values are good for the opponent

This keeps the module easy to understand and ready for a simple negamax search.
"""

from __future__ import annotations

from .board import Board
from .enums import Color, PieceType

CHECKMATE_SCORE = 1_000_000
STALEMATE_SCORE = 0

PIECE_VALUES = {
    PieceType.PAWN: 100,
    PieceType.KNIGHT: 320,
    PieceType.BISHOP: 330,
    PieceType.ROOK: 500,
    PieceType.QUEEN: 900,
    PieceType.KING: 0,
}


def material_balance(board: Board) -> int:
    """Return raw material balance from White's perspective."""
    score = 0
    for _, piece in board.iter_pieces():
        value = PIECE_VALUES[piece.piece_type]
        if piece.color is Color.WHITE:
            score += value
        else:
            score -= value
    return score


def evaluate(board: Board, perspective: Color) -> int:
    """Evaluate a position from the chosen side's perspective."""
    if board.is_checkmate():
        return -CHECKMATE_SCORE if perspective is board.state.side_to_move else CHECKMATE_SCORE

    if board.is_stalemate():
        return STALEMATE_SCORE

    score = material_balance(board)
    return score if perspective is Color.WHITE else -score


def evaluate_for_side_to_move(board: Board) -> int:
    """Convenience wrapper for search code."""
    return evaluate(board, board.state.side_to_move)
