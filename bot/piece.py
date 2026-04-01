"""Piece model used by the board state."""

from __future__ import annotations

from dataclasses import dataclass

from .enums import Color, PieceType


@dataclass(frozen=True, slots=True)
class Piece:
    color: Color
    piece_type: PieceType

    @property
    def is_slider(self) -> bool:
        return self.piece_type in {
            PieceType.BISHOP,
            PieceType.ROOK,
            PieceType.QUEEN,
        }
