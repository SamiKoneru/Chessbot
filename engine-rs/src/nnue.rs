//! NNUE inference with int8/int16 quantization.
//!
//! Scheme (Stockfish-style):
//!   feature-transformer weights: int16, scale FT_SCALE (127)
//!   feature-transformer accumulator: int16, updated incrementally
//!   clipped-ReLU output of accumulator: int8 in [0, FT_SCALE]
//!   dense weights: int8, scale W_SCALE (64)
//!   dense biases: int32, scale FT_SCALE * W_SCALE
//!   dense accumulator: int32; requantize by `/ W_SCALE` and clamp to [0, FT_SCALE]
//!   final output: int32 at scale FT_SCALE * W_SCALE -> centipawns via
//!     `out * EVAL_SCALE / (FT_SCALE * W_SCALE)`
//!
//! Binary format magic: "NNU2" (the float32 "NNU1" format is no longer supported —
//! re-export the .pt with `scripts/export_nnue.py`).

use crate::board::Board;
use crate::types::{Color, Move, MoveKind, PieceType};
use std::io;

pub const HIDDEN: usize = 256;
const NUM_PIECE_KINDS: usize = 10;
const STRIDE_KING: usize = 64 * NUM_PIECE_KINDS;
const FEATURES_PER_PERSPECTIVE: usize = 40_960;
const HEAD: usize = 32;
const FT_SCALE: i32 = 127;
const W_SCALE: i32 = 64;
const EVAL_SCALE: i32 = 400;

/// Per-perspective accumulators (int16): index 0 = white perspective, 1 = black.
pub type AccPair = [[i16; HIDDEN]; 2];

pub struct Nnue {
    ft_weight: Vec<i16>,    // [FEATURES_PER_PERSPECTIVE * HIDDEN]
    ft_bias: [i16; HIDDEN],
    fc1_w: Vec<i8>,         // [HEAD * (2 * HIDDEN)]
    fc1_b: [i32; HEAD],
    fc2_w: Vec<i8>,         // [HEAD * HEAD]
    fc2_b: [i32; HEAD],
    out_w: [i8; HEAD],
    out_b: i32,
}

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

fn read_i16s(data: &[u8], pos: &mut usize, count: usize) -> Vec<i16> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(i16::from_le_bytes([data[*pos], data[*pos + 1]]));
        *pos += 2;
    }
    v
}
fn read_i8s(data: &[u8], pos: &mut usize, count: usize) -> Vec<i8> {
    let v: Vec<i8> = data[*pos..*pos + count].iter().map(|&b| b as i8).collect();
    *pos += count;
    v
}
fn read_i32s(data: &[u8], pos: &mut usize, count: usize) -> Vec<i32> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(i32::from_le_bytes([
            data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3],
        ]));
        *pos += 4;
    }
    v
}

impl Nnue {
    pub fn load(path: &str) -> io::Result<Nnue> {
        let data = std::fs::read(path)?;
        if data.len() < 8 || &data[0..4] != b"NNU2" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad NNUE magic; expected int8-quantized weights (NNU2). \
                 Re-export the .pt with `python scripts/export_nnue.py`.",
            ));
        }
        let hidden = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if hidden != HIDDEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("NNUE hidden={hidden}, this build supports only {HIDDEN}"),
            ));
        }
        let mut p = 8usize;
        let ft_weight = read_i16s(&data, &mut p, FEATURES_PER_PERSPECTIVE * HIDDEN);
        let ft_bias_v = read_i16s(&data, &mut p, HIDDEN);
        let fc1_w = read_i8s(&data, &mut p, HEAD * 2 * HIDDEN);
        let fc1_b_v = read_i32s(&data, &mut p, HEAD);
        let fc2_w = read_i8s(&data, &mut p, HEAD * HEAD);
        let fc2_b_v = read_i32s(&data, &mut p, HEAD);
        let out_w_v = read_i8s(&data, &mut p, HEAD);
        let out_b = read_i32s(&data, &mut p, 1)[0];

        let mut ft_bias = [0i16; HIDDEN];
        ft_bias.copy_from_slice(&ft_bias_v);
        let mut fc1_b = [0i32; HEAD];
        fc1_b.copy_from_slice(&fc1_b_v);
        let mut fc2_b = [0i32; HEAD];
        fc2_b.copy_from_slice(&fc2_b_v);
        let mut out_w = [0i8; HEAD];
        out_w.copy_from_slice(&out_w_v);

        Ok(Nnue {
            ft_weight,
            ft_bias,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            out_w,
            out_b,
        })
    }

    /// Deterministic small-weight net for validation tests. incremental == refresh
    /// holds for *any* weights, so we use a simple bounded pattern.
    pub fn dummy() -> Nnue {
        let g16 = |j: usize, m: i32| ((j as i32 % m) - m / 2) as i16;
        let g8 = |j: usize, m: i32| ((j as i32 % m) - m / 2) as i8;
        let ft_weight: Vec<i16> =
            (0..FEATURES_PER_PERSPECTIVE * HIDDEN).map(|j| g16(j, 7)).collect();
        let mut ft_bias = [0i16; HIDDEN];
        for j in 0..HIDDEN {
            ft_bias[j] = g16(j, 5);
        }
        let fc1_w: Vec<i8> = (0..HEAD * 2 * HIDDEN).map(|j| g8(j, 9)).collect();
        let mut fc1_b = [0i32; HEAD];
        for j in 0..HEAD {
            fc1_b[j] = (j as i32 % 11 - 5) * 10;
        }
        let fc2_w: Vec<i8> = (0..HEAD * HEAD).map(|j| g8(j, 5)).collect();
        let mut fc2_b = [0i32; HEAD];
        for j in 0..HEAD {
            fc2_b[j] = (j as i32 % 7 - 3) * 10;
        }
        let mut out_w = [0i8; HEAD];
        for j in 0..HEAD {
            out_w[j] = (j as i32 % 5 - 2) as i8;
        }
        Nnue {
            ft_weight,
            ft_bias,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            out_w,
            out_b: 100,
        }
    }

    fn refresh_one(&self, board: &Board, persp: Color) -> [i16; HIDDEN] {
        let mut acc = self.ft_bias;
        let king = board.king_square(persp);
        for sq in 0..64u8 {
            if let Some((c, pt)) = board.piece_at(sq) {
                if pt == PieceType::King {
                    continue;
                }
                let base = feature_idx(king, sq, c, pt, persp) * HIDDEN;
                let row = &self.ft_weight[base..base + HIDDEN];
                for i in 0..HIDDEN {
                    acc[i] = acc[i].saturating_add(row[i]);
                }
            }
        }
        acc
    }

    pub fn refresh(&self, board: &Board) -> AccPair {
        [
            self.refresh_one(board, Color::White),
            self.refresh_one(board, Color::Black),
        ]
    }

    #[inline]
    fn toggle_one(
        &self,
        acc: &mut [i16; HIDDEN],
        persp: Color,
        king: u8,
        pc: Color,
        pt: PieceType,
        sq: u8,
        add: bool,
    ) {
        let base = feature_idx(king, sq, pc, pt, persp) * HIDDEN;
        let row = &self.ft_weight[base..base + HIDDEN];
        if add {
            for i in 0..HIDDEN {
                acc[i] = acc[i].saturating_add(row[i]);
            }
        } else {
            for i in 0..HIDDEN {
                acc[i] = acc[i].saturating_sub(row[i]);
            }
        }
    }

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
            acc[mover_color.index()] = self.refresh_one(board_after, mover_color);
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

    pub fn eval_from_acc(&self, acc: &AccPair, stm: Color) -> i32 {
        let other = stm.opposite();
        // Clipped ReLU on the int16 accumulator -> int8 in [0, FT_SCALE].
        let mut x = [0i8; 2 * HIDDEN];
        for i in 0..HIDDEN {
            x[i] = acc[stm.index()][i].clamp(0, FT_SCALE as i16) as i8;
            x[HIDDEN + i] = acc[other.index()][i].clamp(0, FT_SCALE as i16) as i8;
        }
        // fc1: int8 * int8 -> int32; bias is at scale FT_SCALE * W_SCALE.
        let mut y1 = [0i8; HEAD];
        for o in 0..HEAD {
            let mut s = self.fc1_b[o];
            let base = o * 2 * HIDDEN;
            for i in 0..2 * HIDDEN {
                s += self.fc1_w[base + i] as i32 * x[i] as i32;
            }
            // Divide by W_SCALE to bring back to FT_SCALE scale; clamp -> i8 input for next layer.
            y1[o] = (s / W_SCALE).clamp(0, FT_SCALE) as i8;
        }
        // fc2: same pattern.
        let mut y2 = [0i8; HEAD];
        for o in 0..HEAD {
            let mut s = self.fc2_b[o];
            let base = o * HEAD;
            for i in 0..HEAD {
                s += self.fc2_w[base + i] as i32 * y1[i] as i32;
            }
            y2[o] = (s / W_SCALE).clamp(0, FT_SCALE) as i8;
        }
        // Final output (no CReLU). At scale FT_SCALE * W_SCALE; convert to centipawns.
        let mut out_i32 = self.out_b;
        for i in 0..HEAD {
            out_i32 += self.out_w[i] as i32 * y2[i] as i32;
        }
        (out_i32 * EVAL_SCALE) / (FT_SCALE * W_SCALE)
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        let acc = self.refresh(board);
        self.eval_from_acc(&acc, board.side)
    }
}

#[inline]
fn castle_rook(mv: Move) -> (u8, u8) {
    match mv.to {
        6 => (7, 5),
        2 => (0, 3),
        62 => (63, 61),
        58 => (56, 59),
        _ => unreachable!("non-castling move passed to castle_rook"),
    }
}
