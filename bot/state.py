"""Dataclasses for metadata attached to a board position."""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import Color


@dataclass(slots=True)
class CastlingRights:
    white_kingside: bool = True
    white_queenside: bool = True
    black_kingside: bool = True
    black_queenside: bool = True

    @classmethod
    def from_fen(cls, value: str) -> "CastlingRights":
        if value == "-":
            return cls(False, False, False, False)

        return cls(
            white_kingside="K" in value,
            white_queenside="Q" in value,
            black_kingside="k" in value,
            black_queenside="q" in value,
        )

    def to_fen(self) -> str:
        symbols = []
        if self.white_kingside:
            symbols.append("K")
        if self.white_queenside:
            symbols.append("Q")
        if self.black_kingside:
            symbols.append("k")
        if self.black_queenside:
            symbols.append("q")
        return "".join(symbols) or "-"


@dataclass(slots=True)
class BoardState:
    side_to_move: Color = Color.WHITE
    castling_rights: CastlingRights = field(default_factory=CastlingRights)
    en_passant_target: str | None = None
    halfmove_clock: int = 0
    fullmove_number: int = 1
