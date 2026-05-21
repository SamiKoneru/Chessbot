"""Helpers for converting between board coordinates and algebraic squares."""

from __future__ import annotations

from .constants import BOARD_SIZE

FILES = "abcdefgh"
RANKS = "12345678"


def is_valid_coordinate(rank: int, file: int) -> bool:
    return 0 <= rank < BOARD_SIZE and 0 <= file < BOARD_SIZE


# Precomputed lookup tables. There are only 64 squares, and these conversions
# sit in the hottest part of move generation / hashing (tens of millions of
# calls per search). Replacing per-call string parsing and f-string building
# with dict lookups is a large speedup with no behavior change.
_NAME_TABLE: dict[tuple[int, int], str] = {}
_INDEX_TABLE: dict[str, tuple[int, int]] = {}
for _rank in range(BOARD_SIZE):
    for _file in range(BOARD_SIZE):
        _name = f"{FILES[_file]}{RANKS[7 - _rank]}"
        _NAME_TABLE[(_rank, _file)] = _name
        _INDEX_TABLE[_name] = (_rank, _file)
del _rank, _file, _name


def square_name(rank: int, file: int) -> str:
    try:
        return _NAME_TABLE[(rank, file)]
    except KeyError:
        raise ValueError(f"Invalid board coordinate: {(rank, file)!r}") from None


def square_index(square: str) -> tuple[int, int]:
    try:
        return _INDEX_TABLE[square]
    except KeyError:
        raise ValueError(f"Invalid algebraic square: {square!r}") from None
