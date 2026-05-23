//! NNUE evaluation: loads weights exported by `scripts/export_nnue.py` and runs
//! the forward pass. Matches the Python `bot/nnue` model exactly (HalfKP features,
//! shared feature transformer, clipped-ReLU dense head, output × 400 = centipawns).
//!
//! This first version recomputes the accumulator from scratch each call (correct,
//! but the slow part). Incremental accumulator updates + int8 quantization are the
//! planned speed follow-up.

use crate::board::Board;
use crate::types::{Color, PieceType};
use std::io;

const NUM_PIECE_KINDS: u32 = 10; // 5 piece types (excl. king) × {own, opp}
const STRIDE_KING: u32 = 64 * NUM_PIECE_KINDS; // 640
const FEATURES_PER_PERSPECTIVE: usize = 40_960;
const HEAD: usize = 32; // both hidden dense layers are 32 wide
const EVAL_SCALE: f32 = 400.0;

pub struct Nnue {
    hidden: usize,
    ft_weight: Vec<f32>, // [FEATURES_PER_PERSPECTIVE * hidden], row = feature
    ft_bias: Vec<f32>,   // [hidden]
    fc1_w: Vec<f32>,     // [HEAD * (2*hidden)]  (out, in)
    fc1_b: Vec<f32>,     // [HEAD]
    fc2_w: Vec<f32>,     // [HEAD * HEAD]
    fc2_b: Vec<f32>,     // [HEAD]
    out_w: Vec<f32>,     // [HEAD]
    out_b: f32,
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

impl Nnue {
    pub fn load(path: &str) -> io::Result<Nnue> {
        let data = std::fs::read(path)?;
        if data.len() < 8 || &data[0..4] != b"NNU1" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad NNUE magic"));
        }
        let hidden = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let mut p = 8usize;

        let ft_weight = read_f32s(&data, &mut p, FEATURES_PER_PERSPECTIVE * hidden);
        let ft_bias = read_f32s(&data, &mut p, hidden);
        let fc1_w = read_f32s(&data, &mut p, HEAD * (2 * hidden));
        let fc1_b = read_f32s(&data, &mut p, HEAD);
        let fc2_w = read_f32s(&data, &mut p, HEAD * HEAD);
        let fc2_b = read_f32s(&data, &mut p, HEAD);
        let out_w = read_f32s(&data, &mut p, HEAD);
        let out_b = read_f32s(&data, &mut p, 1)[0];

        Ok(Nnue { hidden, ft_weight, ft_bias, fc1_w, fc1_b, fc2_w, fc2_b, out_w, out_b })
    }

    /// HalfKP active features for one perspective. Square indexing is LERF
    /// (a1=0…h8=63), matching the Python feature extraction directly; black's
    /// perspective is the vertical mirror (sq ^ 56).
    fn add_features(&self, board: &Board, perspective: Color, feats: &mut Vec<u32>) {
        feats.clear();
        let king = board.king_square(perspective);
        let king_sq = if perspective == Color::White { king as u32 } else { (king ^ 56) as u32 };
        for sq in 0..64u8 {
            if let Some((color, pt)) = board.piece_at(sq) {
                if pt == PieceType::King {
                    continue;
                }
                let psq = if perspective == Color::White { sq as u32 } else { (sq ^ 56) as u32 };
                let base = pt.index() as u32; // pawn..queen = 0..4
                let kind = base * 2 + if color == perspective { 0 } else { 1 };
                feats.push(king_sq * STRIDE_KING + psq * NUM_PIECE_KINDS + kind);
            }
        }
    }

    fn accumulate(&self, feats: &[u32]) -> Vec<f32> {
        let h = self.hidden;
        let mut acc = self.ft_bias.clone();
        for &f in feats {
            let base = f as usize * h;
            let row = &self.ft_weight[base..base + h];
            for i in 0..h {
                acc[i] += row[i];
            }
        }
        acc
    }

    /// Evaluation in centipawns, from the side-to-move's perspective.
    pub fn evaluate(&self, board: &Board) -> i32 {
        let h = self.hidden;
        let stm = board.side;
        let other = stm.opposite();

        let mut feats = Vec::with_capacity(32);
        self.add_features(board, stm, &mut feats);
        let acc_stm = self.accumulate(&feats);
        self.add_features(board, other, &mut feats);
        let acc_other = self.accumulate(&feats);

        // Input layer: clipped-ReLU of [stm | other] accumulators.
        let mut x = vec![0f32; 2 * h];
        for i in 0..h {
            x[i] = acc_stm[i].clamp(0.0, 1.0);
            x[h + i] = acc_other[i].clamp(0.0, 1.0);
        }

        let y1 = linear(&self.fc1_w, &self.fc1_b, &x, HEAD, 2 * h, true);
        let y2 = linear(&self.fc2_w, &self.fc2_b, &y1, HEAD, HEAD, true);

        let mut out = self.out_b;
        for i in 0..HEAD {
            out += self.out_w[i] * y2[i];
        }
        (out * EVAL_SCALE).round() as i32
    }
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
