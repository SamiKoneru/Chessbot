"""Readable board-logic package for the chessbot project."""

from .attacks import attacked_squares_by_color, is_square_attacked
from .board import Board
from .evaluation import (
    CHECKMATE_SCORE,
    PIECE_VALUES,
    STALEMATE_SCORE,
    evaluate,
    evaluate_for_side_to_move,
    material_balance,
)
from .enums import Color, PieceType
from .move import Move
from .move_generation import legal_moves, normalize_move, pseudo_legal_moves
from .piece import Piece
from .search import (
    SearchResult,
    alpha_beta_search,
    choose_move,
    iterative_deepening_search,
    ordered_moves,
)
from .state import BoardState, CastlingRights
from .transposition_table import BoundType, TranspositionEntry, TranspositionTable
from .zobrist import ZOBRIST_SEED, compute_zobrist_hash

__all__ = [
    "alpha_beta_search",
    "attacked_squares_by_color",
    "Board",
    "BoardState",
    "BoundType",
    "CastlingRights",
    "CHECKMATE_SCORE",
    "choose_move",
    "Color",
    "compute_zobrist_hash",
    "evaluate",
    "evaluate_for_side_to_move",
    "is_square_attacked",
    "iterative_deepening_search",
    "legal_moves",
    "material_balance",
    "Move",
    "ordered_moves",
    "PIECE_VALUES",
    "normalize_move",
    "Piece",
    "PieceType",
    "pseudo_legal_moves",
    "SearchResult",
    "STALEMATE_SCORE",
    "TranspositionEntry",
    "TranspositionTable",
    "ZOBRIST_SEED",
]
