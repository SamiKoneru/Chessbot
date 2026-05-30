//! Alpha-beta search: iterative deepening, transposition table, move ordering,
//! null-move pruning, killer moves, check extensions, and quiescence.
//!
//! Port of the Python `bot/search.py`. Scores are negamax (always from the
//! side-to-move's perspective).

use crate::board::{Board, Undo};
use crate::nnue::{AccPair, Nnue};
use crate::types::{Color, Move, MoveKind, PieceType};
use std::time::Instant;

pub const INF: i32 = 1_000_000;
pub const MATE: i32 = 30_000;
/// Scores at least this large are "mate in N" (used to detect forced mate).
pub const MATE_THRESHOLD: i32 = MATE - 1000;

const MAX_PLY: usize = 128;
const NULL_MIN_DEPTH: i32 = 3;
const NULL_R: i32 = 2;
const MAX_EXTENSION_PLY: usize = 40;
// Late Move Reductions: reduce late, quiet moves at depth >= LMR_MIN_DEPTH once
// past the first LMR_MIN_MOVE moves. A re-search at full depth catches any move
// the reduction undervalued, so LMR never loses a genuinely good move.
const LMR_MIN_DEPTH: i32 = 3;
const LMR_MIN_MOVE: usize = 3;

// Move-ordering bonuses.
const TT_BONUS: i32 = 1_000_000;
const PROMO_BONUS: i32 = 100_000;
const CAPTURE_BONUS: i32 = 50_000;
const KILLER_BONUS: i32 = 40_000;
const CASTLE_BONUS: i32 = 1_000;
const ORDER_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];
// History bonus is capped below KILLER_BONUS so quiet history can't outrank
// killers or captures — it only orders quiet moves among themselves.
const HISTORY_MAX: i32 = 30_000;

// Aspiration windows: from this depth on, search a narrow window around the
// previous iteration's score and widen only on a fail.
const ASPIRATION_MIN_DEPTH: i32 = 5;
const ASPIRATION_DELTA: i32 = 30;

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
    /// Quiet-move success counts indexed by [from][to]; used for move ordering.
    history: Box<[[i32; 64]; 64]>,
    nnue: Option<Nnue>,
    /// Per-perspective accumulators for the current search path (only used with
    /// the NNUE). Top is the current position; pushed on make, popped on unmake.
    acc_stack: Vec<AccPair>,
    stop_time: Option<Instant>,
    stopped: bool,
    pub nodes: u64,
}

impl Searcher {
    pub fn new() -> Searcher {
        Searcher {
            tt: vec![None; TT_SIZE],
            killers: [[None; 2]; MAX_PLY],
            history: Box::new([[0; 64]; 64]),
            nnue: None,
            acc_stack: Vec::with_capacity(MAX_PLY + 8),
            stop_time: None,
            stopped: false,
            nodes: 0,
        }
    }

    /// Use the NNUE for leaf evaluation instead of the material baseline.
    pub fn set_nnue(&mut self, nnue: Nnue) {
        self.nnue = Some(nnue);
    }

    pub fn has_nnue(&self) -> bool {
        self.nnue.is_some()
    }

    /// Reset transposition table and killers (called on `ucinewgame`).
    pub fn clear(&mut self) {
        for e in self.tt.iter_mut() {
            *e = None;
        }
        self.killers = [[None; 2]; MAX_PLY];
    }

    #[inline]
    fn check_stop(&mut self) {
        if let Some(t) = self.stop_time {
            if Instant::now() >= t {
                self.stopped = true;
            }
        }
    }

    /// Time-aware iterative deepening for UCI. Prints `info` lines per completed
    /// depth and returns the best move from the last fully-searched depth.
    /// `deadline = None` means no time limit (search to `max_depth`).
    pub fn search_uci(&mut self, board: &mut Board, max_depth: i32, deadline: Option<Instant>) -> Option<Move> {
        self.nodes = 0;
        self.stopped = false;
        self.stop_time = deadline;
        self.init_acc(board);
        *self.history = [[0; 64]; 64];
        let start = Instant::now();
        let mut best_move = None;
        let mut prev = 0;

        for depth in 1..=max_depth {
            let (mv, score) = self.aspiration_search(board, depth, prev);
            if self.stopped {
                // This depth was aborted mid-way; keep the previous depth's move
                // (but make sure we always have *something* to return).
                if best_move.is_none() {
                    best_move = mv;
                }
                break;
            }
            best_move = mv;
            prev = score;

            let elapsed = start.elapsed();
            let nps = (self.nodes as f64 / elapsed.as_secs_f64().max(1e-9)) as u64;
            let pv = self.extract_pv(board, depth as usize);
            let pv_str: Vec<String> = pv.iter().map(|m| m.to_uci()).collect();
            println!(
                "info depth {depth} score {} nodes {} nps {nps} time {} pv {}",
                format_score(score),
                self.nodes,
                elapsed.as_millis(),
                pv_str.join(" ")
            );

            if score.abs() >= MATE_THRESHOLD {
                break; // forced mate found
            }
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }
        }
        best_move
    }

    /// Reconstruct the principal variation by walking best moves through the TT.
    fn extract_pv(&self, board: &mut Board, max_len: usize) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut undos = Vec::new();
        for _ in 0..max_len {
            let mv = match self.tt_probe(board.hash).and_then(|e| e.best_move) {
                Some(m) if board.legal_moves().contains(&m) => m,
                _ => break,
            };
            pv.push(mv);
            undos.push(board.make_move(mv));
        }
        while let Some(u) = undos.pop() {
            board.unmake_move(u);
        }
        pv
    }

    #[inline]
    fn eval(&self, board: &Board) -> i32 {
        match &self.nnue {
            // Use the maintained accumulator (fast); the dense head only.
            Some(n) => n.eval_from_acc(self.acc_stack.last().unwrap(), board.side),
            None => crate::eval::evaluate(board),
        }
    }

    /// Initialize the accumulator stack from the root position (NNUE only).
    fn init_acc(&mut self, board: &Board) {
        self.acc_stack.clear();
        if self.nnue.is_some() {
            let acc = self.nnue.as_ref().unwrap().refresh(board);
            self.acc_stack.push(acc);
        }
    }

    /// make_move that also incrementally updates the accumulator (NNUE only).
    fn make(&mut self, board: &mut Board, mv: Move) -> Undo {
        if self.nnue.is_none() {
            return board.make_move(mv);
        }
        // Read pre-move piece info (the captured square differs for en passant).
        let mover = board.piece_at(mv.from).expect("no piece on from-square");
        let captured = if mv.kind == MoveKind::EnPassant {
            let csq = if mover.0 == Color::White { mv.to - 8 } else { mv.to + 8 };
            Some((mover.0.opposite(), PieceType::Pawn, csq))
        } else {
            board.piece_at(mv.to).map(|(c, pt)| (c, pt, mv.to))
        };
        let undo = board.make_move(mv);
        let new = {
            let nnue = self.nnue.as_ref().unwrap();
            let prev = self.acc_stack.last().unwrap();
            nnue.apply_move(board, mv, prev, mover, captured)
        };
        self.acc_stack.push(new);
        undo
    }

    #[inline]
    fn unmake(&mut self, board: &mut Board, undo: Undo) {
        board.unmake_move(undo);
        if self.nnue.is_some() {
            self.acc_stack.pop();
        }
    }

    /// Iterative deepening. Returns the best move and its score.
    pub fn search(&mut self, board: &mut Board, max_depth: i32) -> (Option<Move>, i32) {
        self.nodes = 0;
        self.stopped = false;
        self.stop_time = None;
        self.init_acc(board);
        *self.history = [[0; 64]; 64];
        let mut best = (None, 0);
        let mut prev = 0;
        for depth in 1..=max_depth {
            best = self.aspiration_search(board, depth, prev);
            prev = best.1;
        }
        best
    }

    /// One iterative-deepening iteration with an aspiration window: search a
    /// narrow window around `prev` (the last depth's score) and widen on a fail.
    fn aspiration_search(&mut self, board: &mut Board, depth: i32, prev: i32) -> (Option<Move>, i32) {
        if depth < ASPIRATION_MIN_DEPTH {
            return self.search_root(board, depth, -INF, INF);
        }
        let mut delta = ASPIRATION_DELTA;
        let mut alpha = (prev - delta).max(-INF);
        let mut beta = (prev + delta).min(INF);
        loop {
            let (mv, score) = self.search_root(board, depth, alpha, beta);
            if self.stopped {
                return (mv, score);
            }
            if score <= alpha {
                delta *= 2;
                alpha = if delta > 800 { -INF } else { prev - delta };
            } else if score >= beta {
                delta *= 2;
                beta = if delta > 800 { INF } else { prev + delta };
            } else {
                return (mv, score);
            }
        }
    }

    fn search_root(&mut self, board: &mut Board, depth: i32, mut alpha: i32, beta: i32) -> (Option<Move>, i32) {
        let orig_alpha = alpha;
        let tt_move = self.tt_probe(board.hash).and_then(|e| e.best_move);
        let mut moves = board.legal_moves();
        self.order_moves(board, &mut moves, tt_move, 0);

        let mut best = -INF;
        let mut best_move = None;

        for mv in moves {
            let undo = self.make(board, mv);
            let ext = self.extension(board, 0);
            let score = -self.alpha_beta(board, depth - 1 + ext, -beta, -alpha, 1);
            self.unmake(board, undo);
            self.nodes += 1;

            if self.stopped {
                break;
            }
            if score > best {
                best = score;
                best_move = Some(mv);
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break; // fail-high (aspiration window exceeded)
            }
        }

        let bound = if best <= orig_alpha {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt_store(board.hash, depth, best, bound, best_move);
        (best_move, best)
    }

    fn alpha_beta(&mut self, board: &mut Board, depth: i32, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        if self.stopped {
            return 0;
        }
        // Hard recursion cap: extensions can keep depth from falling, so bound ply
        // directly to guarantee termination (and never overflow the stack).
        if ply >= MAX_PLY - 1 {
            return self.eval(board);
        }
        if (self.nodes & 4095) == 0 {
            self.check_stop();
            if self.stopped {
                return 0;
            }
        }
        if depth <= 0 {
            return self.quiescence(board, alpha, beta, ply, 0);
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

        for (move_index, mv) in moves.into_iter().enumerate() {
            let is_quiet = self.is_quiet(board, mv);
            let undo = self.make(board, mv);
            let ext = self.extension(board, ply);
            let new_depth = depth - 1 + ext;

            let score;
            if move_index == 0 {
                // Principal-variation move: full window at full depth.
                score = -self.alpha_beta(board, new_depth, -beta, -alpha, ply + 1);
            } else {
                // Late Move Reduction for late, quiet, non-checking moves.
                let mut reduction = 0;
                if depth >= LMR_MIN_DEPTH
                    && move_index >= LMR_MIN_MOVE
                    && is_quiet
                    && ext == 0
                    && !in_check
                {
                    reduction = if move_index >= 6 { 2 } else { 1 };
                }
                let reduced = (new_depth - reduction).max(0);
                // Null-window scout (possibly reduced).
                let mut s = -self.alpha_beta(board, reduced, -alpha - 1, -alpha, ply + 1);
                // A reduced scout that beat alpha may have undervalued the move:
                // re-search at full depth (still null window).
                if reduction > 0 && s > alpha {
                    s = -self.alpha_beta(board, new_depth, -alpha - 1, -alpha, ply + 1);
                }
                // PVS: if it lands inside the window, re-search with the full window.
                if s > alpha && s < beta {
                    s = -self.alpha_beta(board, new_depth, -beta, -alpha, ply + 1);
                }
                score = s;
            }

            self.unmake(board, undo);
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
                    self.add_history(mv.from, mv.to, depth);
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

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32, ply: usize, qs_depth: i32) -> i32 {
        if self.stopped {
            return 0;
        }
        // Hard recursion cap: a long sequence of checks (which the in-check branch
        // searches without reducing depth) must not recurse without bound.
        if ply >= MAX_PLY - 1 {
            return self.eval(board);
        }
        let in_check = board.in_check(board.side);

        let mut moves;
        let mut best;
        if in_check {
            // Must consider all evasions — no stand-pat, no pruning.
            moves = board.legal_moves();
            if moves.is_empty() {
                return -MATE + ply as i32; // checkmate
            }
            best = -INF;
            self.order_moves(board, &mut moves, None, ply);
        } else {
            let stand = self.eval(board);
            if stand >= beta {
                return stand;
            }
            if stand > alpha {
                alpha = stand;
            }
            best = stand;
            moves = self.qsearch_moves(board, qs_depth);
            self.order_moves(board, &mut moves, None, ply);
        }

        for mv in moves {
            let undo = self.make(board, mv);
            let score = -self.quiescence(board, -beta, -alpha, ply + 1, qs_depth + 1);
            self.unmake(board, undo);
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

    /// Moves searched in quiescence (when not in check): promotions, captures
    /// that don't lose material (SEE >= 0), and — only at the qsearch horizon —
    /// non-losing quiet checks, so short forcing tactics aren't truncated.
    fn qsearch_moves(&self, board: &mut Board, qs_depth: i32) -> Vec<Move> {
        let legal = board.legal_moves();
        let mut out = Vec::with_capacity(legal.len());
        for mv in legal {
            if mv.promo.is_some() {
                out.push(mv);
            } else if mv.kind == MoveKind::EnPassant || board.piece_at(mv.to).is_some() {
                if board.see(mv) >= 0 {
                    out.push(mv);
                }
            } else if qs_depth == 0 && self.gives_check(board, mv) && board.see(mv) >= 0 {
                out.push(mv);
            }
        }
        out
    }

    /// Does `mv` give check? Uses the board's own make/unmake (not the search
    /// wrappers), so it never disturbs the NNUE accumulator stack.
    fn gives_check(&self, board: &mut Board, mv: Move) -> bool {
        let undo = board.make_move(mv);
        let chk = board.in_check(board.side);
        board.unmake_move(undo);
        chk
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
        // Quiet-move history (capped below the killer/capture tiers).
        let is_capture = mv.kind == MoveKind::EnPassant || board.piece_at(mv.to).is_some();
        if mv.promo.is_none() && !is_capture {
            score += self.history[mv.from as usize][mv.to as usize].min(HISTORY_MAX);
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
    fn add_history(&mut self, from: u8, to: u8, depth: i32) {
        self.history[from as usize][to as usize] += depth * depth;
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

/// Format a score for a UCI `info` line: `cp N`, or `mate N` for forced mates.
pub fn format_score(score: i32) -> String {
    if score >= MATE_THRESHOLD {
        format!("mate {}", (MATE - score + 1) / 2)
    } else if score <= -MATE_THRESHOLD {
        format!("mate {}", -((MATE + score + 1) / 2))
    } else {
        format!("cp {score}")
    }
}
