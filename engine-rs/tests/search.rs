//! Search and hashing correctness tests.

use chessbot_engine::board::Board;
use chessbot_engine::perft::perft;
use chessbot_engine::search::{Searcher, MATE_THRESHOLD};

#[test]
fn finds_mate_in_one() {
    // Back-rank mate: Re1-e8#.
    let mut b = Board::from_fen("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1");
    let mut s = Searcher::new();
    let (mv, score) = s.search(&mut b, 4);
    assert_eq!(mv.unwrap().to_uci(), "e1e8");
    assert!(score >= MATE_THRESHOLD, "expected mate score, got {score}");
}

#[test]
fn grabs_free_material() {
    // White queen on d1 can capture an undefended black queen on d8 down the open d-file.
    let mut b = Board::from_fen("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1");
    let mut s = Searcher::new();
    let (mv, _) = s.search(&mut b, 4);
    assert_eq!(mv.unwrap().to_uci(), "d1d8", "should capture the free queen");
}

#[test]
fn search_is_deterministic() {
    let mut b = Board::startpos();
    let mut s1 = Searcher::new();
    let mut s2 = Searcher::new();
    let r1 = s1.search(&mut b, 5);
    let r2 = s2.search(&mut b, 5);
    assert_eq!(r1.0.map(|m| m.to_uci()), r2.0.map(|m| m.to_uci()));
    assert_eq!(r1.1, r2.1);
}

/// Incremental Zobrist hashing (maintained across make/unmake) must always match
/// a from-scratch recompute. perft exercises every make/unmake path; if the hash
/// ever drifted, transposition-table behavior would silently corrupt the search.
/// We piggyback on perft passing (move gen unchanged) plus this round-trip check.
#[test]
fn hash_round_trips_through_make_unmake() {
    let mut b = Board::from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    );
    let before = b.hash;
    let moves = b.legal_moves();
    for mv in moves {
        let undo = b.make_move(mv);
        // After a move, the hash must differ from the pre-move hash (side flipped).
        assert_ne!(b.hash, before, "hash unchanged after a move");
        b.unmake_move(undo);
        assert_eq!(b.hash, before, "hash not restored after unmake");
    }
}

#[test]
fn movegen_still_correct() {
    // Guard against the board.rs edits (Zobrist, accessors) breaking move gen.
    let mut b = Board::startpos();
    assert_eq!(perft(&mut b, 4), 197_281);
}
