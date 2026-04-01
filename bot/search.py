"""Simple alpha-beta search built on top of legal move generation."""

from __future__ import annotations

from dataclasses import dataclass

from .board import Board
from .evaluation import CHECKMATE_SCORE, evaluate_for_side_to_move
from .enums import PieceType
from .move import Move
from .transposition_table import BoundType, TranspositionEntry, TranspositionTable

SEARCH_INFINITY = CHECKMATE_SCORE + 1

# Keep move-order values local to search so evaluation can evolve separately.
ORDERING_PIECE_VALUES = {
    PieceType.PAWN: 100,
    PieceType.KNIGHT: 320,
    PieceType.BISHOP: 330,
    PieceType.ROOK: 500,
    PieceType.QUEEN: 900,
    PieceType.KING: 0,
}
PROMOTION_BONUS = 10_000
CAPTURE_BONUS = 5_000
CASTLING_BONUS = 100
PREFERRED_MOVE_BONUS = 100_000


@dataclass(frozen=True, slots=True)
class SearchResult:
    score: int
    best_move: Move | None
    nodes_searched: int


def choose_move(board: Board, depth: int) -> Move | None:
    """Return the best move found with iterative deepening up to the requested depth."""
    return iterative_deepening_search(board, depth).best_move


def ordered_moves(
    board: Board,
    moves: list[Move] | None = None,
    preferred_move: Move | None = None,
) -> list[Move]:
    """Order moves using a cheap search-specific heuristic."""
    candidates = board.legal_moves() if moves is None else moves
    return sorted(
        candidates,
        key=lambda move: _move_order_score(board, move, preferred_move),
        reverse=True,
    )


def alpha_beta_search(
    board: Board,
    depth: int,
    preferred_move: Move | None = None,
    transposition_table: TranspositionTable | None = None,
) -> SearchResult:
    """Search the current position and return the best root move."""
    if depth < 0:
        raise ValueError(f"Search depth must be non-negative, got {depth}")
    transposition_table = TranspositionTable() if transposition_table is None else transposition_table

    if board.is_game_over():
        return SearchResult(
            score=_terminal_score(board, ply=0),
            best_move=None,
            nodes_searched=1,
        )
    if depth == 0:
        score, nodes = _quiescence(
            board,
            alpha=-SEARCH_INFINITY,
            beta=SEARCH_INFINITY,
            ply=0,
        )
        return SearchResult(score=score, best_move=None, nodes_searched=nodes)

    cached_entry, _, _, cached_score = _probe_transposition_table(
        transposition_table,
        board,
        depth,
        alpha=-SEARCH_INFINITY,
        beta=SEARCH_INFINITY,
    )
    if cached_score is not None:
        return SearchResult(
            score=cached_score,
            best_move=None if cached_entry is None else cached_entry.best_move,
            nodes_searched=1,
        )

    root_preferred_move = preferred_move
    if root_preferred_move is None and cached_entry is not None:
        root_preferred_move = cached_entry.best_move

    legal_moves = ordered_moves(board, preferred_move=root_preferred_move)
    if not legal_moves:
        return SearchResult(
            score=evaluate_for_side_to_move(board),
            best_move=None,
            nodes_searched=1,
        )

    alpha = -SEARCH_INFINITY
    beta = SEARCH_INFINITY
    best_score = -SEARCH_INFINITY
    best_move: Move | None = None
    nodes_searched = 0

    for move in legal_moves:
        candidate = board.clone()
        candidate.apply_move(move)

        child_score, child_nodes = _alpha_beta(
            candidate,
            depth=depth - 1,
            alpha=-beta,
            beta=-alpha,
            ply=1,
            transposition_table=transposition_table,
        )
        score = -child_score

        nodes_searched += child_nodes
        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

    transposition_table.store(
        zobrist_hash=board.zobrist_hash,
        depth=depth,
        score=best_score,
        bound=BoundType.EXACT,
        best_move=best_move,
    )

    return SearchResult(
        score=best_score,
        best_move=best_move,
        nodes_searched=nodes_searched,
    )


def iterative_deepening_search(
    board: Board,
    max_depth: int,
    transposition_table: TranspositionTable | None = None,
) -> SearchResult:
    """Search depths 1..N, feeding the previous best move into the next pass."""
    if max_depth < 0:
        raise ValueError(f"Search depth must be non-negative, got {max_depth}")
    transposition_table = TranspositionTable() if transposition_table is None else transposition_table

    if max_depth == 0:
        return alpha_beta_search(board, depth=0, transposition_table=transposition_table)

    preferred_move: Move | None = None
    latest_result = alpha_beta_search(board, depth=0, transposition_table=transposition_table)
    total_nodes = latest_result.nodes_searched

    for depth in range(1, max_depth + 1):
        latest_result = alpha_beta_search(
            board,
            depth=depth,
            preferred_move=preferred_move,
            transposition_table=transposition_table,
        )
        total_nodes += latest_result.nodes_searched
        preferred_move = latest_result.best_move

    return SearchResult(
        score=latest_result.score,
        best_move=latest_result.best_move,
        nodes_searched=total_nodes,
    )


def _alpha_beta(
    board: Board,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    transposition_table: TranspositionTable,
) -> tuple[int, int]:
    """Negamax alpha-beta from the side-to-move perspective."""
    original_alpha = alpha
    original_beta = beta

    if board.is_game_over():
        return _terminal_score(board, ply), 1
    if depth == 0:
        return _quiescence(board, alpha, beta, ply)

    cached_entry, alpha, beta, cached_score = _probe_transposition_table(
        transposition_table,
        board,
        depth,
        alpha,
        beta,
    )
    if cached_score is not None:
        return cached_score, 1

    preferred_move = None if cached_entry is None else cached_entry.best_move
    legal_moves = ordered_moves(board, preferred_move=preferred_move)
    if not legal_moves:
        return evaluate_for_side_to_move(board), 1

    best_score = -SEARCH_INFINITY
    best_move: Move | None = None
    nodes_searched = 0

    for move in legal_moves:
        candidate = board.clone()
        candidate.apply_move(move)

        child_score, child_nodes = _alpha_beta(
            candidate,
            depth=depth - 1,
            alpha=-beta,
            beta=-alpha,
            ply=ply + 1,
            transposition_table=transposition_table,
        )
        score = -child_score

        nodes_searched += child_nodes
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    transposition_table.store(
        zobrist_hash=board.zobrist_hash,
        depth=depth,
        score=best_score,
        bound=_score_bound_type(best_score, original_alpha, original_beta),
        best_move=best_move,
    )

    return best_score, nodes_searched


def _quiescence(board: Board, alpha: int, beta: int, ply: int) -> tuple[int, int]:
    """Extend noisy leaf positions until they become quiet enough to evaluate."""
    if board.is_game_over():
        return _terminal_score(board, ply), 1

    nodes_searched = 1

    if board.is_in_check():
        best_score = -SEARCH_INFINITY
        moves = ordered_moves(board)
    else:
        stand_pat = evaluate_for_side_to_move(board)
        if stand_pat >= beta:
            return beta, nodes_searched
        if stand_pat > alpha:
            alpha = stand_pat
        best_score = stand_pat
        moves = _quiescence_moves(board)

    if not moves:
        return best_score, nodes_searched

    for move in moves:
        candidate = board.clone()
        candidate.apply_move(move)

        child_score, child_nodes = _quiescence(candidate, -beta, -alpha, ply + 1)
        score = -child_score

        nodes_searched += child_nodes
        if score > best_score:
            best_score = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            return beta, nodes_searched

    return best_score, nodes_searched


def _move_order_score(
    board: Board,
    move: Move,
    preferred_move: Move | None,
) -> int:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0

    score = 0

    if preferred_move is not None and move == preferred_move:
        score += PREFERRED_MOVE_BONUS

    if move.promotion is not None:
        score += PROMOTION_BONUS + ORDERING_PIECE_VALUES[move.promotion]

    captured_piece = board.piece_at(move.to_square)
    if move.is_en_passant:
        score += CAPTURE_BONUS + (10 * ORDERING_PIECE_VALUES[PieceType.PAWN]) - ORDERING_PIECE_VALUES[piece.piece_type]
    elif captured_piece is not None:
        victim_value = ORDERING_PIECE_VALUES[captured_piece.piece_type]
        attacker_value = ORDERING_PIECE_VALUES[piece.piece_type]
        score += CAPTURE_BONUS + (10 * victim_value) - attacker_value

    if move.is_castling:
        score += CASTLING_BONUS

    return score


def _terminal_score(board: Board, ply: int) -> int:
    """Return terminal score from the side-to-move perspective.

    Mate scores are ply-adjusted so the engine prefers faster mates and slower losses.
    """
    if board.is_checkmate():
        return ply - CHECKMATE_SCORE
    if board.is_stalemate():
        return 0
    return evaluate_for_side_to_move(board)


def _quiescence_moves(board: Board) -> list[Move]:
    moves = [
        move
        for move in board.legal_moves()
        if move.is_en_passant
        or move.promotion is not None
        or board.piece_at(move.to_square) is not None
    ]
    return ordered_moves(board, moves=moves)


def _probe_transposition_table(
    transposition_table: TranspositionTable,
    board: Board,
    depth: int,
    alpha: int,
    beta: int,
) -> tuple[TranspositionEntry | None, int, int, int | None]:
    entry = transposition_table.lookup(board.zobrist_hash)
    if entry is None:
        return None, alpha, beta, None

    if entry.depth < depth:
        return entry, alpha, beta, None

    if entry.bound is BoundType.EXACT:
        return entry, alpha, beta, entry.score

    if entry.bound is BoundType.LOWER:
        alpha = max(alpha, entry.score)
    elif entry.bound is BoundType.UPPER:
        beta = min(beta, entry.score)

    if alpha >= beta:
        return entry, alpha, beta, entry.score

    return entry, alpha, beta, None


def _score_bound_type(score: int, alpha: int, beta: int) -> BoundType:
    if score <= alpha:
        return BoundType.UPPER
    if score >= beta:
        return BoundType.LOWER
    return BoundType.EXACT
