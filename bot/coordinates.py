"""Helpers for converting between board coordinates and algebraic squares."""

from __future__ import annotations

from .constants import BOARD_SIZE

FILES = "abcdefgh"
RANKS = "12345678"


def is_valid_coordinate(rank: int, file: int) -> bool:
    return 0 <= rank < BOARD_SIZE and 0 <= file < BOARD_SIZE


def square_name(rank: int, file: int) -> str:
    if not is_valid_coordinate(rank, file):
        raise ValueError(f"Invalid board coordinate: {(rank, file)!r}")
    return f"{FILES[file]}{RANKS[7 - rank]}"


def square_index(square: str) -> tuple[int, int]:
    if len(square) != 2 or square[0] not in FILES or square[1] not in RANKS:
        raise ValueError(f"Invalid algebraic square: {square!r}")

    file = FILES.index(square[0])
    rank = 7 - RANKS.index(square[1])
    return rank, file
