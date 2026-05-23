//! Alpha-beta search: iterative deepening, transposition table, move ordering,
//! null-move pruning, killer moves, check extensions, and quiescence.
//!
//! Port of the Python `bot/search.py`. Scores are negamax (always from the
//! side-to-move's perspective).

use crate::board::Board;
use crate::eval::evaluate;
use crate::types::{Move, MoveKind, PieceType};

pub const INF: i32 = 1_000_000;
pub const MATE: i32 = 30_000;
/// Scores at least this large are "mate in N" (used to detect forced mate).
pub const MATE_THRESHOLD: i32 = MATE - 1000;

const MAX_PLY: usize = 128;
const NULL_MIN_DEPTH: i32 = 3;
const NULL_R: i32 = 2;
const MAX_EXTENSION_PLY: usize = 40;

// Move-ordering bonuses.
const TT_BONUS: i32 = 1_000_000;
const PROMO_BONUS: i32 = 100_000;
const CAPTURE_BONUS: i32 = 50_000;
const KILLER_BONUS: i32 = 40_000;
const CASTLE_BONUS: i32 = 1_000;
const ORDER_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    best_move: Option<Move>,
    score: i32,
    depth: i32,
    bound: Bound,
}

const TT_BITS: usize = 20; // ~1M entries
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: usize = TT_SIZE - 1;

pub struct Searcher {
    tt: Vec<Option<TtEntry>>,
    killers: [[Option<Move>; 2]; MAX_PLY],
    pub nodes: u64,
}

impl Searcher {
    pub fn new() -> Searcher {
        Searcher {
            tt: vec![None; TT_SIZE],
            killers: [[None; 2]; MAX_PLY],
            nodes: 0,
        }
    }

    /// Iterative deepening. Returns the best move and its score.
    pub fn search(&mut self, board: &mut Board, max_depth: i32) -> (Option<Move>, i32) {
        self.nodes = 0;
        let mut best = (None, 0);
        for depth in 1..=max_depth {
            best = self.search_root(board, depth);
        }
        best
    }

    fn search_root(&mut self, board: &mut Board, depth: i32) -> (Option<Move>, i32) {
        let tt_move = self.tt_probe(board.hash).and_then(|e| e.best_move);
        let mut moves = board.legal_moves();
        self.order_moves(board, &mut moves, tt_move, 0);

        let mut alpha = -INF;
        let beta = INF;
        let mut best = -INF;
        let mut best_move = None;

        for mv in moves {
            let undo = board.make_move(mv);
            let ext = self.extension(board, 0);
            let score = -self.alpha_beta(board, depth - 1 + ext, -beta, -alpha, 1);
            board.unmake_move(undo);
            self.nodes += 1;

            if score > best {
                best = score;
                best_move = Some(mv);
            }
            if score > alpha {
                alpha = score;
            }
        }

        self.tt_store(board.hash, depth, best, Bound::Exact, best_move);
        (best_move, best)
    }

    fn alpha_beta(&mut self, board: &mut Board, depth: i32, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        if depth <= 0 {
            return self.quiescence(board, alpha, beta, ply);
        }

        let orig_alpha = alpha;
        let key = board.hash;
        let mut tt_move = None;
        if let Some(e) = self.tt_probe(key) {
            tt_move = e.best_move;
            if e.depth >= depth {
                match e.bound {
                    Bound::Exact => return e.score,
                    Bound::Lower if e.score >= beta => return e.score,
                    Bound::Upper if e.score <= alpha => return e.score,
                    _ => {}
                }
            }
        }

        let in_check = board.in_check(board.side);

        // Null-move pruning. Skip our turn; if the opponent still can't reach
        // beta, this node is too good — prune. Guarded against check and zugzwang.
        if depth >= NULL_MIN_DEPTH && !in_check && board.has_non_pawn_material(board.side) {
            let u = board.make_null();
            let score = -self.alpha_beta(board, depth - 1 - NULL_R, -beta, -beta + 1, ply + 1);
            board.unmake_null(u);
            if score >= beta {
                return score;
            }
        }

        let mut moves = board.legal_moves();
        if moves.is_empty() {
            // Terminal: checkmate (ply-adjusted to prefer slower losses) or stalemate.
            return if in_check { -MATE + ply as i32 } else { 0 };
        }
        self.order_moves(board, &mut moves, tt_move, ply);

        let mut best = -INF;
        let mut best_move = None;

        for mv in moves {
            let is_quiet = self.is_quiet(board, mv);
            let undo = board.make_move(mv);
            let ext = self.extension(board, ply);
            let score = -self.alpha_beta(board, depth - 1 + ext, -beta, -alpha, ply + 1);
            board.unmake_move(undo);
            self.nodes += 1;

            if score > best {
                best = score;
                best_move = Some(mv);
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                if is_quiet {
                    self.add_killer(ply, mv);
                }
                break;
            }
        }

        let bound = if best <= orig_alpha {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt_store(key, depth, best, bound, best_move);
        best
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        let in_check = board.in_check(board.side);

        let mut moves;
        let mut best;
        if in_check {
            moves = board.legal_moves();
            if moves.is_empty() {
                return -MATE + ply as i32; // checkmate
            }
            best = -INF;
            self.order_moves(board, &mut moves, None, ply);
        } else {
            let stand = evaluate(board);
            if stand >= beta {
                return stand;
            }
            if stand > alpha {
                alpha = stand;
            }
            best = stand;
            // Only noisy moves (captures / promotions / en passant).
            moves = board.legal_moves();
            moves.retain(|&mv| self.is_noisy(board, mv));
            self.order_moves(board, &mut moves, None, ply);
        }

        for mv in moves {
            let undo = board.make_move(mv);
            let score = -self.quiescence(board, -beta, -alpha, ply + 1);
            board.unmake_move(undo);
            self.nodes += 1;

            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }

    /// Extend by one ply when the move just played gives check (bounded by ply).
    #[inline]
    fn extension(&self, board: &Board, ply: usize) -> i32 {
        if ply < MAX_EXTENSION_PLY && board.in_check(board.side) {
            1
        } else {
            0
        }
    }

    #[inline]
    fn is_noisy(&self, board: &Board, mv: Move) -> bool {
        mv.promo.is_some() || mv.kind == MoveKind::EnPassant || board.piece_at(mv.to).is_some()
    }

    #[inline]
    fn is_quiet(&self, board: &Board, mv: Move) -> bool {
        !self.is_noisy(board, mv)
    }

    fn order_moves(&self, board: &Board, moves: &mut [Move], tt_move: Option<Move>, ply: usize) {
        moves.sort_by_key(|&mv| std::cmp::Reverse(self.move_score(board, mv, tt_move, ply)));
    }

    fn move_score(&self, board: &Board, mv: Move, tt_move: Option<Move>, ply: usize) -> i32 {
        let mut score = 0;
        if Some(mv) == tt_move {
            score += TT_BONUS;
        }
        if let Some(p) = mv.promo {
            score += PROMO_BONUS + ORDER_VALUES[p.index()];
        }
        let attacker = board
            .piece_at(mv.from)
            .map(|(_, pt)| ORDER_VALUES[pt.index()])
            .unwrap_or(0);
        if mv.kind == MoveKind::EnPassant {
            score += CAPTURE_BONUS + 10 * ORDER_VALUES[PieceType::Pawn.index()] - attacker;
        } else if let Some((_, victim)) = board.piece_at(mv.to) {
            score += CAPTURE_BONUS + 10 * ORDER_VALUES[victim.index()] - attacker;
        }
        if matches!(mv.kind, MoveKind::CastleKing | MoveKind::CastleQueen) {
            score += CASTLE_BONUS;
        }
        if ply < MAX_PLY && self.is_killer(ply, mv) {
            score += KILLER_BONUS;
        }
        score
    }

    #[inline]
    fn is_killer(&self, ply: usize, mv: Move) -> bool {
        self.killers[ply][0] == Some(mv) || self.killers[ply][1] == Some(mv)
    }

    #[inline]
    fn add_killer(&mut self, ply: usize, mv: Move) {
        if ply >= MAX_PLY || self.killers[ply][0] == Some(mv) {
            return;
        }
        self.killers[ply][1] = self.killers[ply][0];
        self.killers[ply][0] = Some(mv);
    }

    #[inline]
    fn tt_probe(&self, key: u64) -> Option<TtEntry> {
        let e = self.tt[(key as usize) & TT_MASK];
        match e {
            Some(entry) if entry.key == key => Some(entry),
            _ => None,
        }
    }

    #[inline]
    fn tt_store(&mut self, key: u64, depth: i32, score: i32, bound: Bound, best_move: Option<Move>) {
        let idx = (key as usize) & TT_MASK;
        // Depth-preferred replacement.
        if let Some(existing) = self.tt[idx] {
            if existing.key == key && existing.depth > depth {
                return;
            }
        }
        self.tt[idx] = Some(TtEntry { key, best_move, score, depth, bound });
    }
}

impl Default for Searcher {
    fn default() -> Self {
        Searcher::new()
    }
}
