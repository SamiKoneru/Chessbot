"""Transposition-table storage for alpha-beta search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .move import Move


class BoundType(str, Enum):
    """How the stored score relates to the true minimax value."""

    EXACT = "exact"
    LOWER = "lower"
    UPPER = "upper"


@dataclass(frozen=True, slots=True)
class TranspositionEntry:
    """Cached search result for one Zobrist-hashed position."""

    zobrist_hash: int
    depth: int
    score: int
    bound: BoundType
    best_move: Move | None = None


class TranspositionTable:
    """Simple depth-aware transposition table.

    For readability, this first version keeps a single dictionary and replaces
    existing entries only when the new search reached at least the same depth.
    """

    def __init__(self) -> None:
        self._entries: dict[int, TranspositionEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def lookup(self, zobrist_hash: int) -> TranspositionEntry | None:
        return self._entries.get(zobrist_hash)

    def store_entry(self, entry: TranspositionEntry) -> None:
        existing_entry = self.lookup(entry.zobrist_hash)
        if existing_entry is not None and existing_entry.depth > entry.depth:
            return
        self._entries[entry.zobrist_hash] = entry

    def store(
        self,
        zobrist_hash: int,
        depth: int,
        score: int,
        bound: BoundType,
        best_move: Move | None = None,
    ) -> None:
        self.store_entry(
            TranspositionEntry(
                zobrist_hash=zobrist_hash,
                depth=depth,
                score=score,
                bound=bound,
                best_move=best_move,
            )
        )
