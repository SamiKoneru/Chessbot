//! Static Exchange Evaluation correctness tests.

use chessbot_engine::board::Board;

fn see_of(fen: &str, uci: &str) -> i32 {
    let mut b = Board::from_fen(fen);
    let mv = b
        .legal_moves()
        .into_iter()
        .find(|m| m.to_uci() == uci)
        .unwrap_or_else(|| panic!("move {uci} not legal in {fen}"));
    b.see(mv)
}

#[test]
fn pawn_takes_undefended_pawn() {
    // exd5, the d5 pawn is undefended: win a pawn.
    assert_eq!(see_of("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"), 100);
}

#[test]
fn pawn_takes_defended_pawn() {
    // exd5, recaptured by c6 pawn: win a pawn, lose a pawn -> 0.
    assert_eq!(see_of("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"), 0);
}

#[test]
fn knight_takes_pawn_defended_by_pawn() {
    // Nxd5, recaptured by c6 pawn: win pawn (100), lose knight (320) -> -220.
    assert_eq!(see_of("4k3/8/2p5/3p4/8/4N3/8/4K3 w - - 0 1", "e3d5"), -220);
}

#[test]
fn pawn_takes_queen() {
    // exd5 capturing a queen, undefended: +900.
    assert_eq!(see_of("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", "e4d5"), 900);
}
