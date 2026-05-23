//! Perft: count leaf nodes of the move tree to a fixed depth.
//!
//! This is the correctness oracle for move generation — the counts must match
//! the known reference values (and the Python engine's perft).

use crate::board::Board;

pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = board.legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut nodes = 0;
    for mv in moves {
        let undo = board.make_move(mv);
        nodes += perft(board, depth - 1);
        board.unmake_move(undo);
    }
    nodes
}

/// Per-root-move breakdown (handy for debugging mismatches against a reference).
pub fn perft_divide(board: &mut Board, depth: u32) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    for mv in board.legal_moves() {
        let undo = board.make_move(mv);
        let n = if depth <= 1 { 1 } else { perft(board, depth - 1) };
        board.unmake_move(undo);
        out.push((mv.to_uci(), n));
    }
    out.sort();
    out
}
