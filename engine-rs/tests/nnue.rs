//! Validates the incremental accumulator against a from-scratch recompute.
//!
//! Uses dummy weights (the incremental == refresh identity holds for any
//! weights) and walks Kiwipete exhaustively to depth 3, so every move type
//! — captures, en passant, castling, promotions, king moves — is exercised.
//! If the incremental update ever drifted from the true accumulator, the NNUE
//! eval would silently corrupt during search; this is the guard against that.

use chessbot_engine::board::Board;
use chessbot_engine::nnue::{AccPair, Nnue, HIDDEN};
use chessbot_engine::types::{Color, Move, MoveKind, PieceType};

fn captured_of(board: &Board, mv: Move) -> Option<(Color, PieceType, u8)> {
    let mover_color = board.piece_at(mv.from).unwrap().0;
    if mv.kind == MoveKind::EnPassant {
        let csq = if mover_color == Color::White { mv.to - 8 } else { mv.to + 8 };
        Some((mover_color.opposite(), PieceType::Pawn, csq))
    } else {
        board.piece_at(mv.to).map(|(c, pt)| (c, pt, mv.to))
    }
}

fn validate(nnue: &Nnue, board: &mut Board, acc: &AccPair, depth: u32) {
    let fresh = nnue.refresh(board);
    for p in 0..2 {
        for i in 0..HIDDEN {
            assert!(
                (acc[p][i] - fresh[p][i]).abs() < 1e-3,
                "accumulator drift at perspective {p}, unit {i}: incremental {} vs refresh {}",
                acc[p][i],
                fresh[p][i]
            );
        }
    }
    if depth == 0 {
        return;
    }
    for mv in board.legal_moves() {
        let mover = board.piece_at(mv.from).unwrap();
        let captured = captured_of(board, mv);
        let undo = board.make_move(mv);
        let next = nnue.apply_move(board, mv, acc, mover, captured);
        validate(nnue, board, &next, depth - 1);
        board.unmake_move(undo);
    }
}

#[test]
fn incremental_accumulator_matches_refresh() {
    let nnue = Nnue::dummy();
    let mut b =
        Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    let acc = nnue.refresh(&b);
    validate(&nnue, &mut b, &acc, 3);
}
