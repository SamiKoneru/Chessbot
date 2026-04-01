"""Move model for board updates.

This is intentionally lightweight. Search and move-generation code can extend
or wrap this later without forcing board storage to change.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import PieceType


@dataclass(frozen=True, slots=True)
class Move:
    from_square: str
    to_square: str
    promotion: PieceType | None = None
    is_castling: bool = False
    is_en_passant: bool = False

    @property
    def uci(self) -> str:
        suffix = ""
        if self.promotion is not None:
            suffix = self.promotion.value[0]
            if self.promotion is PieceType.KNIGHT:
                suffix = "n"
        return f"{self.from_square}{self.to_square}{suffix}"
