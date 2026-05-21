"""Pseudo-legal and legal move generation for the current position."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .attacks import DIAGONAL_DIRECTIONS, KING_OFFSETS, KNIGHT_OFFSETS, ORTHOGONAL_DIRECTIONS
from .coordinates import is_valid_coordinate, square_index, square_name
from .enums import Color, PieceType
from .move import Move

if TYPE_CHECKING:
    from .board import Board
    from .piece import Piece

PROMOTION_PIECES = (
    PieceType.QUEEN,
    PieceType.ROOK,
    PieceType.BISHOP,
    PieceType.KNIGHT,
)


def normalize_move(board: "Board", move: Move) -> Move:
    """Infer special move flags from the current board position."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return move

    from_rank, from_file = square_index(move.from_square)
    to_rank, to_file = square_index(move.to_square)

    is_castling = move.is_castling
    is_en_passant = move.is_en_passant

    if piece.piece_type is PieceType.KING and abs(to_file - from_file) == 2:
        is_castling = True

    if (
        piece.piece_type is PieceType.PAWN
        and from_file != to_file
        and board.piece_at(move.to_square) is None
        and board.state.en_passant_target == move.to_square
    ):
        is_en_passant = True

    return Move(
        from_square=move.from_square,
        to_square=move.to_square,
        promotion=move.promotion,
        is_castling=is_castling,
        is_en_passant=is_en_passant,
    )


def pseudo_legal_moves(board: "Board") -> list[Move]:
    moves: list[Move] = []

    for square, piece in board.iter_pieces(color=board.state.side_to_move):
        if piece.piece_type is PieceType.PAWN:
            moves.extend(_pawn_moves(board, square, piece))
        elif piece.piece_type is PieceType.KNIGHT:
            moves.extend(_jump_moves(board, square, piece, KNIGHT_OFFSETS))
        elif piece.piece_type is PieceType.BISHOP:
            moves.extend(_ray_moves(board, square, piece, DIAGONAL_DIRECTIONS))
        elif piece.piece_type is PieceType.ROOK:
            moves.extend(_ray_moves(board, square, piece, ORTHOGONAL_DIRECTIONS))
        elif piece.piece_type is PieceType.QUEEN:
            moves.extend(_ray_moves(board, square, piece, ORTHOGONAL_DIRECTIONS + DIAGONAL_DIRECTIONS))
        elif piece.piece_type is PieceType.KING:
            moves.extend(_jump_moves(board, square, piece, KING_OFFSETS))
            moves.extend(_castle_moves(board, square, piece.color))
        else:
            raise ValueError(f"Unsupported piece type: {piece.piece_type!r}")

    return moves


def legal_moves(board: "Board") -> list[Move]:
    moving_color = board.state.side_to_move
    return [move for move in pseudo_legal_moves(board) if _move_keeps_king_safe(board, move, moving_color)]


def is_legal_move(board: "Board", move: Move) -> bool:
    candidate = normalize_move(board, move)
    return candidate in legal_moves(board)


def _pawn_moves(board: "Board", from_square: str, piece: "Piece") -> list[Move]:
    rank, file = square_index(from_square)
    direction = -1 if piece.color is Color.WHITE else 1
    start_rank = 6 if piece.color is Color.WHITE else 1
    promotion_rank = 0 if piece.color is Color.WHITE else 7
    moves: list[Move] = []

    one_step_rank = rank + direction
    if is_valid_coordinate(one_step_rank, file):
        one_step_square = square_name(one_step_rank, file)
        if board.piece_at(one_step_square) is None:
            moves.extend(_pawn_destination_moves(from_square, one_step_square, one_step_rank, promotion_rank))

            two_step_rank = rank + (2 * direction)
            if rank == start_rank and is_valid_coordinate(two_step_rank, file):
                two_step_square = square_name(two_step_rank, file)
                if board.piece_at(two_step_square) is None:
                    moves.append(Move(from_square, two_step_square))

    for file_offset in (-1, 1):
        target_rank = rank + direction
        target_file = file + file_offset
        if not is_valid_coordinate(target_rank, target_file):
            continue

        target_square = square_name(target_rank, target_file)
        target_piece = board.piece_at(target_square)

        if _is_capturable_enemy_piece(target_piece, piece.color):
            moves.extend(_pawn_destination_moves(from_square, target_square, target_rank, promotion_rank))
            continue

        if board.state.en_passant_target == target_square and _has_en_passant_capture(board, target_square, piece.color):
            moves.append(Move(from_square, target_square, is_en_passant=True))

    return moves


def _pawn_destination_moves(
    from_square: str,
    to_square: str,
    destination_rank: int,
    promotion_rank: int,
) -> list[Move]:
    if destination_rank != promotion_rank:
        return [Move(from_square, to_square)]

    return [Move(from_square, to_square, promotion=piece_type) for piece_type in PROMOTION_PIECES]


def _has_en_passant_capture(board: "Board", target_square: str, moving_color: Color) -> bool:
    target_rank, target_file = square_index(target_square)
    captured_rank = target_rank + (1 if moving_color is Color.WHITE else -1)
    captured_square = square_name(captured_rank, target_file)
    captured_piece = board.piece_at(captured_square)
    return captured_piece is not None and (
        captured_piece.color is moving_color.opposite and captured_piece.piece_type is PieceType.PAWN
    )


def _jump_moves(
    board: "Board",
    from_square: str,
    piece: "Piece",
    offsets: tuple[tuple[int, int], ...],
) -> list[Move]:
    rank, file = square_index(from_square)
    moves: list[Move] = []

    for rank_offset, file_offset in offsets:
        target_rank = rank + rank_offset
        target_file = file + file_offset
        if not is_valid_coordinate(target_rank, target_file):
            continue

        target_square = square_name(target_rank, target_file)
        target_piece = board.piece_at(target_square)
        if target_piece is None or _is_capturable_enemy_piece(target_piece, piece.color):
            moves.append(Move(from_square, target_square))

    return moves


def _ray_moves(
    board: "Board",
    from_square: str,
    piece: "Piece",
    directions: tuple[tuple[int, int], ...],
) -> list[Move]:
    rank, file = square_index(from_square)
    moves: list[Move] = []

    for rank_step, file_step in directions:
        target_rank = rank + rank_step
        target_file = file + file_step

        while is_valid_coordinate(target_rank, target_file):
            target_square = square_name(target_rank, target_file)
            target_piece = board.piece_at(target_square)

            if target_piece is None:
                moves.append(Move(from_square, target_square))
            else:
                if _is_capturable_enemy_piece(target_piece, piece.color):
                    moves.append(Move(from_square, target_square))
                break

            target_rank += rank_step
            target_file += file_step

    return moves


def _castle_moves(board: "Board", king_square: str, color: Color) -> list[Move]:
    if board.is_in_check(color):
        return []

    if color is Color.WHITE:
        rights = board.state.castling_rights
        options = []
        if king_square == "e1" and rights.white_kingside:
            options.append(("g1", "h1", ("f1", "g1"), ("f1", "g1")))
        if king_square == "e1" and rights.white_queenside:
            options.append(("c1", "a1", ("d1", "c1", "b1"), ("d1", "c1")))
    else:
        rights = board.state.castling_rights
        options = []
        if king_square == "e8" and rights.black_kingside:
            options.append(("g8", "h8", ("f8", "g8"), ("f8", "g8")))
        if king_square == "e8" and rights.black_queenside:
            options.append(("c8", "a8", ("d8", "c8", "b8"), ("d8", "c8")))

    moves: list[Move] = []
    for destination, rook_square, empty_squares, travel_squares in options:
        rook = board.piece_at(rook_square)
        if rook is None or rook.color is not color or rook.piece_type is not PieceType.ROOK:
            continue
        if any(board.piece_at(square) is not None for square in empty_squares):
            continue
        if any(board.is_square_attacked(square, by_color=color.opposite) for square in travel_squares):
            continue
        moves.append(Move(king_square, destination, is_castling=True))
    return moves


def _move_keeps_king_safe(board: "Board", move: Move, moving_color: Color) -> bool:
    candidate = board.clone()
    # Legality only needs to know whether the king is left in check; the Zobrist
    # hash is never read here, so skip the (expensive) full recompute.
    candidate.apply_move(move, refresh_hash=False)
    return not candidate.is_in_check(moving_color)


def _is_capturable_enemy_piece(target_piece: "Piece | None", moving_color: Color) -> bool:
    return target_piece is not None and (
        target_piece.color is not moving_color and target_piece.piece_type is not PieceType.KING
    )
