"""Small enums used across the board package."""

from __future__ import annotations

from enum import Enum


class Color(str, Enum):
    WHITE = "white"
    BLACK = "black"

    @property
    def opposite(self) -> "Color":
        return Color.BLACK if self is Color.WHITE else Color.WHITE

    @property
    def fen_symbol(self) -> str:
        return "w" if self is Color.WHITE else "b"

    @classmethod
    def from_fen_symbol(cls, value: str) -> "Color":
        mapping = {"w": cls.WHITE, "b": cls.BLACK}
        try:
            return mapping[value]
        except KeyError as exc:
            raise ValueError(f"Unsupported side-to-move symbol: {value!r}") from exc


class PieceType(str, Enum):
    PAWN = "pawn"
    KNIGHT = "knight"
    BISHOP = "bishop"
    ROOK = "rook"
    QUEEN = "queen"
    KING = "king"
