//! NNUE evaluation with an incrementally-updated accumulator.
//!
//! Weights are exported by `scripts/export_nnue.py` and match the Python model
//! exactly (HalfKP features, shared feature transformer, clipped-ReLU dense head,
//! output × 400 = centipawns).
//!
//! The per-perspective accumulators are maintained across make/unmake by the
//! search: a normal move toggles a few feature rows (add/subtract); a king move
//! refreshes that side's accumulator from scratch (because every HalfKP feature
//! is keyed on the own-king square). This is the standard "refresh on king move"
//! strategy and is what makes NNUE fast.

use crate::board::Board;
use crate::types::{Color, Move, MoveKind, PieceType};
use std::io;

pub const HIDDEN: usize = 256;
const NUM_PIECE_KINDS: usize = 10; // 5 piece types (excl. king) × {own, opp}
const STRIDE_KING: usize = 64 * NUM_PIECE_KINDS; // 640
const FEATURES_PER_PERSPECTIVE: usize = 40_960;
const HEAD: usize = 32; // both hidden dense layers are 32 wide
const EVAL_SCALE: f32 = 400.0;

/// Per-perspective accumulators: index 0 = white perspective, 1 = black.
pub type AccPair = [[f32; HIDDEN]; 2];

pub struct Nnue {
    ft_weight: Vec<f32>, // [FEATURES_PER_PERSPECTIVE * HIDDEN], row = feature
    ft_bias: Vec<f32>,   // [HIDDEN]
    fc1_w: Vec<f32>,     // [HEAD * (2*HIDDEN)]
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>, // [HEAD * HEAD]
    fc2_b: Vec<f32>,
    out_w: Vec<f32>, // [HEAD]
    out_b: f32,
}

/// HalfKP feature index for a piece, from one perspective. LERF squares (a1=0…h8=63);
/// black's perspective is the vertical mirror (sq ^ 56). Matches the Python features.
#[inline]
fn feature_idx(king_raw: u8, piece_sq: u8, pc: Color, pt: PieceType, persp: Color) -> usize {
    let (king, psq) = if persp == Color::White {
        (king_raw as usize, piece_sq as usize)
    } else {
        ((king_raw ^ 56) as usize, (piece_sq ^ 56) as usize)
    };
    let kind = pt.index() * 2 + if pc == persp { 0 } else { 1 };
    king * STRIDE_KING + psq * NUM_PIECE_KINDS + kind
}

fn read_f32s(data: &[u8], pos: &mut usize, count: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        let b = [data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]];
        v.push(f32::from_le_bytes(b));
        *pos += 4;
    }
    v
}

fn linear(w: &[f32], b: &[f32], x: &[f32], out_n: usize, in_n: usize, clamp: bool) -> Vec<f32> {
    let mut y = vec![0f32; out_n];
    for o in 0..out_n {
        let mut s = b[o];
        let base = o * in_n;
        for i in 0..in_n {
            s += w[base + i] * x[i];
        }
        y[o] = if clamp { s.clamp(0.0, 1.0) } else { s };
    }
    y
}

impl Nnue {
    pub fn load(path: &str) -> io::Result<Nnue> {
        let data = std::fs::read(path)?;
        if data.len() < 8 || &data[0..4] != b"NNU1" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad NNUE magic"));
        }
        let hidden = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if hidden != HIDDEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("NNUE hidden={hidden}, this build supports only {HIDDEN}"),
            ));
        }
        let mut p = 8usize;
        let ft_weight = read_f32s(&data, &mut p, FEATURES_PER_PERSPECTIVE * HIDDEN);
        let ft_bias = read_f32s(&data, &mut p, HIDDEN);
        let fc1_w = read_f32s(&data, &mut p, HEAD * (2 * HIDDEN));
        let fc1_b = read_f32s(&data, &mut p, HEAD);
        let fc2_w = read_f32s(&data, &mut p, HEAD * HEAD);
        let fc2_b = read_f32s(&data, &mut p, HEAD);
        let out_w = read_f32s(&data, &mut p, HEAD);
        let out_b = read_f32s(&data, &mut p, 1)[0];
        Ok(Nnue { ft_weight, ft_bias, fc1_w, fc1_b, fc2_w, fc2_b, out_w, out_b })
    }

    /// Deterministic small-weight net for validation tests (no weights file needed;
    /// incremental == refresh holds for *any* weights).
    pub fn dummy() -> Nnue {
        let f = |j: usize, m: usize| ((j % m) as f32 - (m as f32) / 2.0) * 0.0009;
        Nnue {
            ft_weight: (0..FEATURES_PER_PERSPECTIVE * HIDDEN).map(|j| f(j, 31)).collect(),
            ft_bias: (0..HIDDEN).map(|j| f(j, 7)).collect(),
            fc1_w: (0..HEAD * 2 * HIDDEN).map(|j| f(j, 13)).collect(),
            fc1_b: (0..HEAD).map(|j| f(j, 5)).collect(),
            fc2_w: (0..HEAD * HEAD).map(|j| f(j, 11)).collect(),
            fc2_b: (0..HEAD).map(|j| f(j, 3)).collect(),
            out_w: (0..HEAD).map(|j| f(j, 9)).collect(),
            out_b: 0.01,
        }
    }

    fn refresh_one(&self, board: &Board, persp: Color) -> [f32; HIDDEN] {
        let mut acc = [0f32; HIDDEN];
        acc.copy_from_slice(&self.ft_bias);
        let king = board.king_square(persp);
        for sq in 0..64u8 {
            if let Some((c, pt)) = board.piece_at(sq) {
                if pt == PieceType::King {
                    continue;
                }
                let base = feature_idx(king, sq, c, pt, persp) * HIDDEN;
                let row = &self.ft_weight[base..base + HIDDEN];
                for i in 0..HIDDEN {
                    acc[i] += row[i];
                }
            }
        }
        acc
    }

    /// Recompute both accumulators from scratch (used at the search root).
    pub fn refresh(&self, board: &Board) -> AccPair {
        [
            self.refresh_one(board, Color::White),
            self.refresh_one(board, Color::Black),
        ]
    }

    #[inline]
    fn toggle_one(&self, acc: &mut [f32; HIDDEN], persp: Color, king: u8, pc: Color, pt: PieceType, sq: u8, add: bool) {
        let base = feature_idx(king, sq, pc, pt, persp) * HIDDEN;
        let row = &self.ft_weight[base..base + HIDDEN];
        if add {
            for i in 0..HIDDEN {
                acc[i] += row[i];
            }
        } else {
            for i in 0..HIDDEN {
                acc[i] -= row[i];
            }
        }
    }

    /// Incrementally produce the accumulators for the position *after* `mv`, given
    /// the accumulators before it. `mover`/`captured` are read from the pre-move
    /// board by the caller (the captured square differs for en passant).
    pub fn apply_move(
        &self,
        board_after: &Board,
        mv: Move,
        prev: &AccPair,
        mover: (Color, PieceType),
        captured: Option<(Color, PieceType, u8)>,
    ) -> AccPair {
        let (mover_color, moved_pt) = mover;
        let wk = board_after.king_square(Color::White);
        let bk = board_after.king_square(Color::Black);
        let mut acc = *prev;
        let is_castle = matches!(mv.kind, MoveKind::CastleKing | MoveKind::CastleQueen);

        if moved_pt == PieceType::King || is_castle {
            // Own king moved → every feature of the mover's perspective changes; refresh it.
            acc[mover_color.index()] = self.refresh_one(board_after, mover_color);
            // The other perspective only needs capture / castling-rook deltas (the
            // moved king is not a HalfKP feature, and the other king is unchanged).
            let other = mover_color.opposite();
            let other_king = if other == Color::White { wk } else { bk };
            let acc_other = &mut acc[other.index()];
            if let Some((cc, cpt, csq)) = captured {
                self.toggle_one(acc_other, other, other_king, cc, cpt, csq, false);
            }
            if is_castle {
                let (rf, rt) = castle_rook(mv);
                self.toggle_one(acc_other, other, other_king, mover_color, PieceType::Rook, rf, false);
                self.toggle_one(acc_other, other, other_king, mover_color, PieceType::Rook, rt, true);
            }
        } else {
            // Non-king move: both kings unchanged, so toggle both perspectives.
            self.toggle_one(&mut acc[0], Color::White, wk, mover_color, moved_pt, mv.from, false);
            self.toggle_one(&mut acc[1], Color::Black, bk, mover_color, moved_pt, mv.from, false);
            if let Some((cc, cpt, csq)) = captured {
                self.toggle_one(&mut acc[0], Color::White, wk, cc, cpt, csq, false);
                self.toggle_one(&mut acc[1], Color::Black, bk, cc, cpt, csq, false);
            }
            let placed = mv.promo.unwrap_or(moved_pt);
            self.toggle_one(&mut acc[0], Color::White, wk, mover_color, placed, mv.to, true);
            self.toggle_one(&mut acc[1], Color::Black, bk, mover_color, placed, mv.to, true);
        }
        acc
    }

    /// Run the dense head on a maintained accumulator pair. Centipawns, stm-relative.
    pub fn eval_from_acc(&self, acc: &AccPair, stm: Color) -> i32 {
        let other = stm.opposite();
        let mut x = [0f32; 2 * HIDDEN];
        for i in 0..HIDDEN {
            x[i] = acc[stm.index()][i].clamp(0.0, 1.0);
            x[HIDDEN + i] = acc[other.index()][i].clamp(0.0, 1.0);
        }
        let y1 = linear(&self.fc1_w, &self.fc1_b, &x, HEAD, 2 * HIDDEN, true);
        let y2 = linear(&self.fc2_w, &self.fc2_b, &y1, HEAD, HEAD, true);
        let mut out = self.out_b;
        for i in 0..HEAD {
            out += self.out_w[i] * y2[i];
        }
        (out * EVAL_SCALE).round() as i32
    }

    /// Standalone from-scratch evaluation (refresh + head). Reference / convenience.
    pub fn evaluate(&self, board: &Board) -> i32 {
        let acc = self.refresh(board);
        self.eval_from_acc(&acc, board.side)
    }
}

/// Rook from/to squares for a castling move, keyed by the king's destination.
#[inline]
fn castle_rook(mv: Move) -> (u8, u8) {
    match mv.to {
        6 => (7, 5),    // white kingside:  h1->f1
        2 => (0, 3),    // white queenside: a1->d1
        62 => (63, 61), // black kingside:  h8->f8
        58 => (56, 59), // black queenside: a8->d8
        _ => unreachable!("non-castling move passed to castle_rook"),
    }
}
