"""Simple alpha-beta search built on top of legal move generation."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Callable

from .board import Board
from .evaluation import CHECKMATE_SCORE, evaluate_for_side_to_move as _default_evaluator
from .enums import Color, PieceType
from .move import Move
from .transposition_table import BoundType, TranspositionEntry, TranspositionTable

SEARCH_INFINITY = CHECKMATE_SCORE + 1

# Null-move pruning parameters.
# R is how many plies we reduce the null-move search by. R=2 is the textbook
# default; R=3 is more aggressive but more error-prone. NULL_MIN_DEPTH stops
# us from null-moving at depths where it isn't worth the risk.
NULL_MOVE_R = 2
NULL_MIN_DEPTH = 3

# Check extensions: search a move that gives check one ply deeper, so forced
# tactical sequences aren't truncated at the horizon. Bounded by ply so a
# perpetual-check line can't extend the search forever (it would otherwise keep
# depth constant). Past this ply, extensions stop and depth strictly decreases.
MAX_EXTENSION_PLY = 48

# The eval function used by search. Module-level so it can be swapped at runtime
# (e.g. material baseline vs NNUE) without threading it through every function.
_evaluator: Callable[[Board], int] = _default_evaluator


def set_evaluator(fn: Callable[[Board], int]) -> None:
    """Swap the leaf evaluation function used by alpha-beta and quiescence."""
    global _evaluator
    _evaluator = fn


def reset_evaluator() -> None:
    """Restore the material-only evaluator."""
    set_evaluator(_default_evaluator)


def evaluate_for_side_to_move(board: Board) -> int:
    return _evaluator(board)

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
KILLER_BONUS = 4_000
CASTLING_BONUS = 100
PREFERRED_MOVE_BONUS = 100_000


class KillerTable:
    """Per-ply store of quiet moves that recently caused a beta cutoff.

    A "killer" at ply N is a quiet move that proved strong in a sibling subtree
    at the same ply. Trying it right after the hash move (and before generic
    quiet moves) tends to produce more cutoffs in similar positions.

    Captures aren't tracked — MVV-LVA already promotes them above quiet moves.
    """

    __slots__ = ("_killers",)

    def __init__(self) -> None:
        self._killers: dict[int, list[Move]] = {}

    def add(self, ply: int, move: Move) -> None:
        slots = self._killers.setdefault(ply, [])
        if slots and slots[0] == move:
            return
        if move in slots:
            slots.remove(move)
        slots.insert(0, move)
        del slots[2:]  # keep two killers per ply

    def is_killer(self, ply: int, move: Move) -> bool:
        return move in self._killers.get(ply, ())


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
    killer_table: KillerTable | None = None,
    ply: int = 0,
) -> list[Move]:
    """Order moves using a cheap search-specific heuristic."""
    candidates = board.legal_moves() if moves is None else moves
    return sorted(
        candidates,
        key=lambda move: _move_order_score(board, move, preferred_move, killer_table, ply),
        reverse=True,
    )


def alpha_beta_search(
    board: Board,
    depth: int,
    preferred_move: Move | None = None,
    transposition_table: TranspositionTable | None = None,
    killer_table: KillerTable | None = None,
) -> SearchResult:
    """Search the current position and return the best root move."""
    if depth < 0:
        raise ValueError(f"Search depth must be non-negative, got {depth}")
    transposition_table = TranspositionTable() if transposition_table is None else transposition_table
    killer_table = KillerTable() if killer_table is None else killer_table

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

    legal_moves = ordered_moves(
        board,
        preferred_move=root_preferred_move,
        killer_table=killer_table,
        ply=0,
    )
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
            killer_table=killer_table,
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
    # Killer table is shared across iterations: killers found at depth N typically
    # remain strong at depth N+1, so reusing them gives the deeper pass better ordering.
    killer_table = KillerTable()

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
            killer_table=killer_table,
        )
        total_nodes += latest_result.nodes_searched
        preferred_move = latest_result.best_move

    return SearchResult(
        score=latest_result.score,
        best_move=latest_result.best_move,
        nodes_searched=total_nodes,
    )


def _has_non_pawn_material(board: Board, color: Color) -> bool:
    """True if `color` has any piece other than pawns and the king.

    Used to gate null-move pruning: in king-and-pawn endgames the "passing is at
    least as good as moving" premise of null-move can fail (zugzwang).
    """
    for _, piece in board.iter_pieces(color):
        if piece.piece_type not in (PieceType.PAWN, PieceType.KING):
            return True
    return False


def _apply_null_move(board: Board) -> Board:
    """Return a clone with side-to-move flipped and en-passant rights cleared.

    A "null move" is just passing the turn — no piece is moved. Clearing EP
    matches what would happen after any real non-pawn move.
    """
    clone = board.clone()
    clone.state.en_passant_target = None
    clone.state.side_to_move = clone.state.side_to_move.opposite
    clone._refresh_zobrist_hash()
    return clone


def _alpha_beta(
    board: Board,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    transposition_table: TranspositionTable,
    killer_table: KillerTable,
) -> tuple[int, int]:
    """Negamax alpha-beta from the side-to-move perspective."""
    original_alpha = alpha
    original_beta = beta

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

    # Generate legal moves once and reuse them for terminal detection, null-move
    # gating, and ordering. Terminal (no-move) detection must run BEFORE null-move
    # pruning, otherwise we could "pass" in a stalemate and prune a drawn node.
    legal_moves = board.legal_moves()
    if not legal_moves:
        return _terminal_score_no_moves(board, ply), 1

    # Null-move pruning. If we can skip our turn and the opponent still can't
    # reach beta, our position is so good they wouldn't have allowed it — prune.
    # Guards: must not be in check (passing while in check is illegal and
    # disastrous), must have non-pawn material (zugzwang protection), and we
    # need enough remaining depth that the reduced search isn't trivially shallow.
    if (
        depth >= NULL_MIN_DEPTH
        and not board.is_in_check()
        and _has_non_pawn_material(board, board.state.side_to_move)
    ):
        null_board = _apply_null_move(board)
        null_score, null_nodes = _alpha_beta(
            null_board,
            depth=depth - 1 - NULL_MOVE_R,
            alpha=-beta,
            beta=-beta + 1,
            ply=ply + 1,
            transposition_table=transposition_table,
            killer_table=killer_table,
        )
        null_score = -null_score
        if null_score >= beta:
            return null_score, null_nodes + 1

    preferred_move = None if cached_entry is None else cached_entry.best_move
    legal_moves = ordered_moves(
        board,
        moves=legal_moves,
        preferred_move=preferred_move,
        killer_table=killer_table,
        ply=ply,
    )

    best_score = -SEARCH_INFINITY
    best_move: Move | None = None
    nodes_searched = 0

    for move in legal_moves:
        candidate = board.clone()
        candidate.apply_move(move)

        # Check extension: if this move gives check (the opponent is now in
        # check), search it one ply deeper. Bounded by ply to guarantee
        # termination on perpetual checks.
        extension = 1 if (ply < MAX_EXTENSION_PLY and candidate.is_in_check()) else 0

        child_score, child_nodes = _alpha_beta(
            candidate,
            depth=depth - 1 + extension,
            alpha=-beta,
            beta=-alpha,
            ply=ply + 1,
            transposition_table=transposition_table,
            killer_table=killer_table,
        )
        score = -child_score

        nodes_searched += child_nodes
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            # Record killer for quiet cutoffs only — captures and promotions
            # already sort above quiet moves via MVV-LVA / promotion bonus.
            if (
                board.piece_at(move.to_square) is None
                and not move.is_en_passant
                and move.promotion is None
            ):
                killer_table.add(ply, move)
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
    nodes_searched = 1

    # Generate legal moves once (replaces the redundant is_game_over() call,
    # which itself generated moves). No legal moves => terminal: mate if in
    # check, otherwise stalemate. This preserves exact terminal scoring.
    in_check = board.is_in_check()
    legal = board.legal_moves()
    if not legal:
        return (ply - CHECKMATE_SCORE if in_check else 0), nodes_searched

    if in_check:
        best_score = -SEARCH_INFINITY
        moves = ordered_moves(board, moves=legal)
    else:
        stand_pat = evaluate_for_side_to_move(board)
        if stand_pat >= beta:
            return beta, nodes_searched
        if stand_pat > alpha:
            alpha = stand_pat
        best_score = stand_pat
        moves = _noisy_moves(board, legal)

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
    killer_table: KillerTable | None = None,
    ply: int = 0,
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

    if killer_table is not None and killer_table.is_killer(ply, move):
        score += KILLER_BONUS

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


def _terminal_score_no_moves(board: Board, ply: int) -> int:
    """Terminal score when the side to move has NO legal moves.

    Avoids regenerating the move list (is_checkmate/is_stalemate would): the
    caller already established there are no legal moves, so it's mate if in
    check, otherwise stalemate.
    """
    if board.is_in_check():
        return ply - CHECKMATE_SCORE
    return 0


def _noisy_moves(board: Board, legal: list[Move]) -> list[Move]:
    """Filter already-generated legal moves down to captures, promotions, and en passant."""
    moves = [
        move
        for move in legal
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
