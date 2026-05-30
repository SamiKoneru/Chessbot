//! Chessbot engine core (Rust).
//!
//! Port of the Python `bot/` engine. Square indexing is Little-Endian Rank-File
//! (LERF): a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63. This matches the
//! 0..63 convention used by the Python NNUE feature extraction, so trained
//! weights will transfer cleanly when we add evaluation.

pub mod types;
pub mod zobrist;
pub mod board;
pub mod eval;
pub mod nnue;
pub mod search;
pub mod perft;
pub mod uci;
