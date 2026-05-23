//! Perft correctness tests against standard reference values. If these pass,
//! move generation (including castling, en passant, promotions, pins, and
//! check evasion) is correct.

use chessbot_engine::board::Board;
use chessbot_engine::perft::perft;

#[test]
fn perft_startpos() {
    let mut b = Board::startpos();
    assert_eq!(perft(&mut b, 1), 20);
    assert_eq!(perft(&mut b, 2), 400);
    assert_eq!(perft(&mut b, 3), 8902);
    assert_eq!(perft(&mut b, 4), 197_281);
    assert_eq!(perft(&mut b, 5), 4_865_609);
}

#[test]
fn perft_kiwipete() {
    // The classic "Kiwipete" position — exercises castling, en passant,
    // promotions, double checks, and pins all at once.
    let mut b = Board::from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    );
    assert_eq!(perft(&mut b, 1), 48);
    assert_eq!(perft(&mut b, 2), 2039);
    assert_eq!(perft(&mut b, 3), 97_862);
    assert_eq!(perft(&mut b, 4), 4_085_603);
}

#[test]
fn perft_position3() {
    // A rook-and-pawn endgame heavy on en passant and promotion edge cases.
    let mut b = Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    assert_eq!(perft(&mut b, 1), 14);
    assert_eq!(perft(&mut b, 2), 191);
    assert_eq!(perft(&mut b, 3), 2812);
    assert_eq!(perft(&mut b, 4), 43_238);
    assert_eq!(perft(&mut b, 5), 674_624);
}
