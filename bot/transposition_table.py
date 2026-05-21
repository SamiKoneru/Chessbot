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


DEFAULT_MAX_ENTRIES = 1_000_000


class TranspositionTable:
    """Simple depth-aware transposition table.

    Replaces an existing entry only when the new search reached at least the
    same depth. Bounded in size: once full, the oldest-inserted entry is evicted
    (FIFO) before a new key is added, so memory can't grow without limit during
    long games or deep searches.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._entries: dict[int, TranspositionEntry] = {}
        self._max_entries = max_entries

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def lookup(self, zobrist_hash: int) -> TranspositionEntry | None:
        return self._entries.get(zobrist_hash)

    def store_entry(self, entry: TranspositionEntry) -> None:
        existing_entry = self._entries.get(entry.zobrist_hash)
        if existing_entry is not None:
            # Keep the deeper search; otherwise refresh in place (no size change).
            if existing_entry.depth > entry.depth:
                return
            self._entries[entry.zobrist_hash] = entry
            return
        if len(self._entries) >= self._max_entries:
            # Evict the oldest inserted key. Dicts preserve insertion order.
            self._entries.pop(next(iter(self._entries)))
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
