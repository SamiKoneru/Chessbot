"""Attack and check helpers.

The functions here answer "which squares does this side control?" rather than
"which squares can this side legally move to?". That distinction matters later
when legal move generation is added.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .coordinates import is_valid_coordinate, square_index, square_name
from .enums import Color, PieceType

if TYPE_CHECKING:
    from .board import Board
    from .piece import Piece

ORTHOGONAL_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONAL_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT_OFFSETS = (
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
)
KING_OFFSETS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def find_king_square(board: "Board", color: Color) -> str:
    for square, piece in board.iter_pieces(color=color):
        if piece.piece_type is PieceType.KING:
            return square
    raise ValueError(f"Could not find {color.value} king on the board")


def attacked_squares_by_color(board: "Board", color: Color) -> set[str]:
    attacks: set[str] = set()
    for square, piece in board.iter_pieces(color=color):
        attacks.update(attacked_squares_for_piece(board, square, piece))
    return attacks


def is_square_attacked(board: "Board", square: str, by_color: Color) -> bool:
    """Targeted attack test: scan outward from `square` and stop at the first
    attacker, instead of enumerating every enemy piece's full attack set.

    This is the hot path for legality / check detection, so it avoids set
    allocation, inlines bounds checks, and short-circuits.
    """
    rank, file = square_index(square)
    grid = board.grid

    # Pawn attackers sit one rank "behind" the diagonal that hits `square`.
    pawn_dir = -1 if by_color is Color.WHITE else 1
    pr = rank - pawn_dir
    if 0 <= pr < 8:
        for pf in (file - 1, file + 1):
            if 0 <= pf < 8:
                p = grid[pr][pf]
                if p is not None and p.color is by_color and p.piece_type is PieceType.PAWN:
                    return True

    # Knight attackers.
    for dr, df in KNIGHT_OFFSETS:
        tr, tf = rank + dr, file + df
        if 0 <= tr < 8 and 0 <= tf < 8:
            p = grid[tr][tf]
            if p is not None and p.color is by_color and p.piece_type is PieceType.KNIGHT:
                return True

    # Adjacent enemy king.
    for dr, df in KING_OFFSETS:
        tr, tf = rank + dr, file + df
        if 0 <= tr < 8 and 0 <= tf < 8:
            p = grid[tr][tf]
            if p is not None and p.color is by_color and p.piece_type is PieceType.KING:
                return True

    # Sliding attackers: orthogonal rays (rook / queen).
    for dr, df in ORTHOGONAL_DIRECTIONS:
        tr, tf = rank + dr, file + df
        while 0 <= tr < 8 and 0 <= tf < 8:
            p = grid[tr][tf]
            if p is not None:
                if p.color is by_color and p.piece_type in (PieceType.ROOK, PieceType.QUEEN):
                    return True
                break
            tr += dr
            tf += df

    # Sliding attackers: diagonal rays (bishop / queen).
    for dr, df in DIAGONAL_DIRECTIONS:
        tr, tf = rank + dr, file + df
        while 0 <= tr < 8 and 0 <= tf < 8:
            p = grid[tr][tf]
            if p is not None:
                if p.color is by_color and p.piece_type in (PieceType.BISHOP, PieceType.QUEEN):
                    return True
                break
            tr += dr
            tf += df

    return False


def attacked_squares_for_piece(board: "Board", square: str, piece: "Piece") -> set[str]:
    rank, file = square_index(square)

    if piece.piece_type is PieceType.PAWN:
        return _pawn_attacks(rank, file, piece.color)
    if piece.piece_type is PieceType.KNIGHT:
        return _jump_attacks(rank, file, KNIGHT_OFFSETS)
    if piece.piece_type is PieceType.BISHOP:
        return _ray_attacks(board, rank, file, DIAGONAL_DIRECTIONS)
    if piece.piece_type is PieceType.ROOK:
        return _ray_attacks(board, rank, file, ORTHOGONAL_DIRECTIONS)
    if piece.piece_type is PieceType.QUEEN:
        return _ray_attacks(board, rank, file, ORTHOGONAL_DIRECTIONS + DIAGONAL_DIRECTIONS)
    if piece.piece_type is PieceType.KING:
        return _jump_attacks(rank, file, KING_OFFSETS)
    raise ValueError(f"Unsupported piece type: {piece.piece_type!r}")


def _pawn_attacks(rank: int, file: int, color: Color) -> set[str]:
    direction = -1 if color is Color.WHITE else 1
    attacks = set()
    for file_offset in (-1, 1):
        target_rank = rank + direction
        target_file = file + file_offset
        if is_valid_coordinate(target_rank, target_file):
            attacks.add(square_name(target_rank, target_file))
    return attacks


def _jump_attacks(rank: int, file: int, offsets: tuple[tuple[int, int], ...]) -> set[str]:
    attacks = set()
    for rank_offset, file_offset in offsets:
        target_rank = rank + rank_offset
        target_file = file + file_offset
        if is_valid_coordinate(target_rank, target_file):
            attacks.add(square_name(target_rank, target_file))
    return attacks


def _ray_attacks(
    board: "Board",
    rank: int,
    file: int,
    directions: tuple[tuple[int, int], ...],
) -> set[str]:
    attacks = set()

    for rank_step, file_step in directions:
        target_rank = rank + rank_step
        target_file = file + file_step

        while is_valid_coordinate(target_rank, target_file):
            target_square = square_name(target_rank, target_file)
            attacks.add(target_square)
            if board.grid[target_rank][target_file] is not None:
                break
            target_rank += rank_step
            target_file += file_step

    return attacks
