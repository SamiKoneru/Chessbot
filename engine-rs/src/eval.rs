//! Material evaluation (placeholder until the NNUE port lands).
//!
//! Returns centipawns from the side-to-move's perspective, matching the Python
//! engine's default evaluator. The search treats this as a swappable leaf eval;
//! the NNUE will plug in here later.

use crate::board::Board;
use crate::types::{Color, PieceType};

// Pawn, Knight, Bishop, Rook, Queen, King.
const VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];

pub fn evaluate(board: &Board) -> i32 {
    let mut score = 0;
    for pt in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ] {
        let v = VALUES[pt.index()];
        score += v * board.count(Color::White, pt) as i32;
        score -= v * board.count(Color::Black, pt) as i32;
    }
    // White-relative -> side-to-move-relative.
    if board.side == Color::White {
        score
    } else {
        -score
    }
}
