"""Shared constants for board representation."""

from __future__ import annotations

from .enums import Color, PieceType

BOARD_SIZE = 8
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

PIECE_TO_FEN = {
    (Color.WHITE, PieceType.PAWN): "P",
    (Color.WHITE, PieceType.KNIGHT): "N",
    (Color.WHITE, PieceType.BISHOP): "B",
    (Color.WHITE, PieceType.ROOK): "R",
    (Color.WHITE, PieceType.QUEEN): "Q",
    (Color.WHITE, PieceType.KING): "K",
    (Color.BLACK, PieceType.PAWN): "p",
    (Color.BLACK, PieceType.KNIGHT): "n",
    (Color.BLACK, PieceType.BISHOP): "b",
    (Color.BLACK, PieceType.ROOK): "r",
    (Color.BLACK, PieceType.QUEEN): "q",
    (Color.BLACK, PieceType.KING): "k",
}

FEN_TO_PIECE = {symbol: key for key, symbol in PIECE_TO_FEN.items()}
