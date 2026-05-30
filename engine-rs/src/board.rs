//! Bitboard board representation, FEN, attacks, move generation, make/unmake.

use crate::types::*;
use crate::zobrist::ZOBRIST;

pub type Bitboard = u64;

const FILE_A: Bitboard = 0x0101_0101_0101_0101;
const FILE_B: Bitboard = FILE_A << 1;
const FILE_G: Bitboard = FILE_A << 6;
const FILE_H: Bitboard = FILE_A << 7;

// Castling-rights bitmask bits.
const WK: u8 = 1;
const WQ: u8 = 2;
const BK: u8 = 4;
const BQ: u8 = 8;

// Piece values used by Static Exchange Evaluation (pawn..king).
const SEE_VALUE: [i32; 6] = [100, 320, 330, 500, 900, 10_000];

#[inline]
fn bit(sq: u8) -> Bitboard {
    1u64 << sq
}

/// Iterate set-bit indices (squares) of a bitboard, lowest first.
struct Bits(Bitboard);
impl Iterator for Bits {
    type Item = u8;
    #[inline]
    fn next(&mut self) -> Option<u8> {
        if self.0 == 0 {
            None
        } else {
            let sq = self.0.trailing_zeros() as u8;
            self.0 &= self.0 - 1;
            Some(sq)
        }
    }
}
#[inline]
fn bits(bb: Bitboard) -> Bits {
    Bits(bb)
}

const ORTHO: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIAG: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

#[inline]
fn knight_attacks(sq: u8) -> Bitboard {
    let b = bit(sq);
    ((b << 17) & !FILE_A)
        | ((b << 15) & !FILE_H)
        | ((b << 10) & !(FILE_A | FILE_B))
        | ((b << 6) & !(FILE_G | FILE_H))
        | ((b >> 17) & !FILE_H)
        | ((b >> 15) & !FILE_A)
        | ((b >> 10) & !(FILE_G | FILE_H))
        | ((b >> 6) & !(FILE_A | FILE_B))
}

#[inline]
fn king_attacks(sq: u8) -> Bitboard {
    let b = bit(sq);
    let east = (b << 1) & !FILE_A;
    let west = (b >> 1) & !FILE_H;
    let mut a = east | west | (b << 8) | (b >> 8);
    a |= (east << 8) | (east >> 8) | (west << 8) | (west >> 8);
    a
}

#[inline]
fn pawn_attacks(color: Color, sq: u8) -> Bitboard {
    let b = bit(sq);
    match color {
        Color::White => ((b << 9) & !FILE_A) | ((b << 7) & !FILE_H),
        Color::Black => ((b >> 7) & !FILE_A) | ((b >> 9) & !FILE_H),
    }
}

/// Sliding attacks by ray-walking (correct, not magic — speed comes later).
fn sliding_attacks(sq: u8, occ: Bitboard, dirs: &[(i8, i8)]) -> Bitboard {
    let mut a = 0u64;
    let r0 = (sq / 8) as i8;
    let f0 = (sq % 8) as i8;
    for &(dr, df) in dirs {
        let mut r = r0 + dr;
        let mut f = f0 + df;
        while (0..8).contains(&r) && (0..8).contains(&f) {
            let s = (r * 8 + f) as u8;
            a |= bit(s);
            if occ & bit(s) != 0 {
                break;
            }
            r += dr;
            f += df;
        }
    }
    a
}

#[inline]
fn castle_mask(sq: u8) -> u8 {
    match sq {
        4 => !(WK | WQ),  // e1
        0 => !WQ,         // a1
        7 => !WK,         // h1
        60 => !(BK | BQ), // e8
        56 => !BQ,        // a8
        63 => !BK,        // h8
        _ => 0xFF,
    }
}

/// Information needed to reverse a move.
#[derive(Clone, Copy)]
pub struct Undo {
    mv: Move,
    captured: Option<PieceType>,
    castling: u8,
    ep: Option<Square>,
    halfmove: u16,
    hash: u64,
}

/// Information needed to reverse a null move.
#[derive(Clone, Copy)]
pub struct NullUndo {
    ep: Option<Square>,
    halfmove: u16,
    hash: u64,
}

#[derive(Clone)]
pub struct Board {
    pieces: [Option<(Color, PieceType)>; 64],
    by_color: [Bitboard; 2],
    by_type: [Bitboard; 6],
    pub side: Color,
    castling: u8,
    ep: Option<Square>,
    pub halfmove: u16,
    pub fullmove: u16,
    pub hash: u64,
}

impl Board {
    fn empty() -> Board {
        Board {
            pieces: [None; 64],
            by_color: [0; 2],
            by_type: [0; 6],
            side: Color::White,
            castling: 0,
            ep: None,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
        }
    }

    pub fn startpos() -> Board {
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    }

    pub fn from_fen(fen: &str) -> Board {
        let mut b = Board::empty();
        let parts: Vec<&str> = fen.split_whitespace().collect();

        let mut rank: i32 = 7; // FEN lists rank 8 first -> our rank index 7
        let mut file: i32 = 0;
        for c in parts[0].chars() {
            match c {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => file += c.to_digit(10).unwrap() as i32,
                _ => {
                    let (color, pt) = char_to_piece(c);
                    b.add_piece((rank * 8 + file) as u8, color, pt);
                    file += 1;
                }
            }
        }

        b.side = if parts.get(1) == Some(&"b") {
            Color::Black
        } else {
            Color::White
        };

        if let Some(cstr) = parts.get(2) {
            for c in cstr.chars() {
                match c {
                    'K' => b.castling |= WK,
                    'Q' => b.castling |= WQ,
                    'k' => b.castling |= BK,
                    'q' => b.castling |= BQ,
                    _ => {}
                }
            }
        }

        b.ep = match parts.get(3) {
            Some(&"-") | None => None,
            Some(s) => Some(str_to_sq(s)),
        };
        b.halfmove = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        b.fullmove = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
        b.hash = b.recompute_hash();
        b
    }

    /// Full Zobrist recompute (used at FEN load; make/unmake maintain it incrementally).
    fn recompute_hash(&self) -> u64 {
        let mut h = 0u64;
        for sq in 0..64 {
            if let Some((c, pt)) = self.pieces[sq] {
                h ^= ZOBRIST.pieces[c.index()][pt.index()][sq];
            }
        }
        h ^= ZOBRIST.castling[self.castling as usize];
        if let Some(ep) = self.ep {
            h ^= ZOBRIST.ep_file[file_of(ep) as usize];
        }
        if self.side == Color::Black {
            h ^= ZOBRIST.side;
        }
        h
    }

    #[inline]
    fn add_piece(&mut self, sq: u8, color: Color, pt: PieceType) {
        self.pieces[sq as usize] = Some((color, pt));
        let b = bit(sq);
        self.by_color[color.index()] |= b;
        self.by_type[pt.index()] |= b;
        self.hash ^= ZOBRIST.pieces[color.index()][pt.index()][sq as usize];
    }

    #[inline]
    fn remove_piece(&mut self, sq: u8) {
        if let Some((color, pt)) = self.pieces[sq as usize] {
            let b = !bit(sq);
            self.by_color[color.index()] &= b;
            self.by_type[pt.index()] &= b;
            self.pieces[sq as usize] = None;
            self.hash ^= ZOBRIST.pieces[color.index()][pt.index()][sq as usize];
        }
    }

    /// Relocate a piece between two squares; `to` must be empty.
    #[inline]
    fn move_piece(&mut self, from: u8, to: u8) {
        let (c, pt) = self.pieces[from as usize].unwrap();
        self.remove_piece(from);
        self.add_piece(to, c, pt);
    }

    #[inline]
    fn occupied(&self) -> Bitboard {
        self.by_color[0] | self.by_color[1]
    }
    #[inline]
    fn pieces_of(&self, color: Color, pt: PieceType) -> Bitboard {
        self.by_color[color.index()] & self.by_type[pt.index()]
    }
    #[inline]
    pub fn king_square(&self, color: Color) -> u8 {
        self.pieces_of(color, PieceType::King).trailing_zeros() as u8
    }

    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<(Color, PieceType)> {
        self.pieces[sq as usize]
    }

    #[inline]
    pub fn count(&self, color: Color, pt: PieceType) -> u32 {
        self.pieces_of(color, pt).count_ones()
    }

    /// True if `color` has any piece other than pawns and the king (zugzwang guard).
    #[inline]
    pub fn has_non_pawn_material(&self, color: Color) -> bool {
        let c = self.by_color[color.index()];
        let pawns_kings = self.by_type[PieceType::Pawn.index()] | self.by_type[PieceType::King.index()];
        c & !pawns_kings != 0
    }

    /// Is `sq` attacked by any piece of `by`?
    pub fn is_square_attacked(&self, sq: u8, by: Color) -> bool {
        // Pawns: a `by` pawn attacks `sq` from where the opposite-color pawn
        // standing on `sq` would attack — hence pawn_attacks(by.opposite(), sq).
        if pawn_attacks(by.opposite(), sq) & self.pieces_of(by, PieceType::Pawn) != 0 {
            return true;
        }
        if knight_attacks(sq) & self.pieces_of(by, PieceType::Knight) != 0 {
            return true;
        }
        if king_attacks(sq) & self.pieces_of(by, PieceType::King) != 0 {
            return true;
        }
        let occ = self.occupied();
        let diag = self.pieces_of(by, PieceType::Bishop) | self.pieces_of(by, PieceType::Queen);
        if sliding_attacks(sq, occ, &DIAG) & diag != 0 {
            return true;
        }
        let ortho = self.pieces_of(by, PieceType::Rook) | self.pieces_of(by, PieceType::Queen);
        if sliding_attacks(sq, occ, &ORTHO) & ortho != 0 {
            return true;
        }
        false
    }

    pub fn in_check(&self, color: Color) -> bool {
        self.is_square_attacked(self.king_square(color), color.opposite())
    }

    /// Static Exchange Evaluation: the material the moving side nets from the
    /// capture sequence on `mv.to`, assuming both sides recapture with their
    /// least valuable piece. Negative means the capture loses material.
    /// Used by quiescence to prune losing captures and gate sacrificial checks.
    pub fn see(&self, mv: Move) -> i32 {
        let mover = match self.pieces[mv.from as usize] {
            Some((c, _)) => c,
            None => return 0,
        };
        let (victim_value, ep_sq) = if mv.kind == MoveKind::EnPassant {
            let ep = match mover {
                Color::White => mv.to - 8,
                Color::Black => mv.to + 8,
            };
            (SEE_VALUE[PieceType::Pawn.index()], Some(ep))
        } else {
            match self.pieces[mv.to as usize] {
                Some((_, pt)) => (SEE_VALUE[pt.index()], None),
                None => (0, None), // quiet move: SEE measures whether the dest is safe
            }
        };
        let attacker_value = SEE_VALUE[self.pieces[mv.from as usize].unwrap().1.index()];
        let mut occ = self.occupied() & !bit(mv.from);
        if let Some(ep) = ep_sq {
            occ &= !bit(ep);
        }
        victim_value - self.see_after(mv.to, attacker_value, mover.opposite(), occ)
    }

    /// Best material the `side` can win from the square `to`, given the piece
    /// currently sitting there is worth `on_square_value`. Recursive swap-off;
    /// each side may decline to capture (hence `.max(0)`). X-rays are handled by
    /// recomputing slider attackers against the shrinking occupancy.
    fn see_after(&self, to: u8, on_square_value: i32, side: Color, occ: Bitboard) -> i32 {
        match self.least_valuable_attacker(to, side, occ) {
            None => 0,
            Some((sq, pt)) => {
                let gain = on_square_value
                    - self.see_after(to, SEE_VALUE[pt.index()], side.opposite(), occ & !bit(sq));
                gain.max(0)
            }
        }
    }

    /// Least valuable piece of `side` that attacks `to` under occupancy `occ`.
    fn least_valuable_attacker(&self, to: u8, side: Color, occ: Bitboard) -> Option<(u8, PieceType)> {
        let mine = self.by_color[side.index()] & occ;
        let pawns = pawn_attacks(side.opposite(), to) & mine & self.by_type[PieceType::Pawn.index()];
        if pawns != 0 {
            return Some((pawns.trailing_zeros() as u8, PieceType::Pawn));
        }
        let knights = knight_attacks(to) & mine & self.by_type[PieceType::Knight.index()];
        if knights != 0 {
            return Some((knights.trailing_zeros() as u8, PieceType::Knight));
        }
        let diag = sliding_attacks(to, occ, &DIAG);
        let bishops = diag & mine & self.by_type[PieceType::Bishop.index()];
        if bishops != 0 {
            return Some((bishops.trailing_zeros() as u8, PieceType::Bishop));
        }
        let ortho = sliding_attacks(to, occ, &ORTHO);
        let rooks = ortho & mine & self.by_type[PieceType::Rook.index()];
        if rooks != 0 {
            return Some((rooks.trailing_zeros() as u8, PieceType::Rook));
        }
        let queens = (diag | ortho) & mine & self.by_type[PieceType::Queen.index()];
        if queens != 0 {
            return Some((queens.trailing_zeros() as u8, PieceType::Queen));
        }
        let king = king_attacks(to) & mine & self.by_type[PieceType::King.index()];
        if king != 0 {
            return Some((king.trailing_zeros() as u8, PieceType::King));
        }
        None
    }

    pub fn pseudo_legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::with_capacity(48);
        let us = self.side;
        let them = us.opposite();
        let own = self.by_color[us.index()];
        let enemy = self.by_color[them.index()];
        let occ = own | enemy;

        // Pawns
        let (forward, start_rank, promo_rank): (i32, u8, u8) = match us {
            Color::White => (8, 1, 7),
            Color::Black => (-8, 6, 0),
        };
        for from in bits(self.pieces_of(us, PieceType::Pawn)) {
            let from_i = from as i32;
            let one = from_i + forward;
            if (0..64).contains(&one) && self.pieces[one as usize].is_none() {
                let to = one as u8;
                self.add_pawn_move(&mut moves, from, to, rank_of(to) == promo_rank);
                if rank_of(from) == start_rank {
                    let two = (from_i + 2 * forward) as u8;
                    if self.pieces[two as usize].is_none() {
                        moves.push(Move { from, to: two, promo: None, kind: MoveKind::DoublePush });
                    }
                }
            }
            let atk = pawn_attacks(us, from);
            for to in bits(atk & enemy) {
                self.add_pawn_move(&mut moves, from, to, rank_of(to) == promo_rank);
            }
            if let Some(ep) = self.ep {
                if atk & bit(ep) != 0 {
                    moves.push(Move { from, to: ep, promo: None, kind: MoveKind::EnPassant });
                }
            }
        }

        // Knights
        for from in bits(self.pieces_of(us, PieceType::Knight)) {
            for to in bits(knight_attacks(from) & !own) {
                moves.push(Move { from, to, promo: None, kind: MoveKind::Normal });
            }
        }
        // Bishops / Rooks / Queens
        for from in bits(self.pieces_of(us, PieceType::Bishop)) {
            for to in bits(sliding_attacks(from, occ, &DIAG) & !own) {
                moves.push(Move { from, to, promo: None, kind: MoveKind::Normal });
            }
        }
        for from in bits(self.pieces_of(us, PieceType::Rook)) {
            for to in bits(sliding_attacks(from, occ, &ORTHO) & !own) {
                moves.push(Move { from, to, promo: None, kind: MoveKind::Normal });
            }
        }
        for from in bits(self.pieces_of(us, PieceType::Queen)) {
            let t = sliding_attacks(from, occ, &DIAG) | sliding_attacks(from, occ, &ORTHO);
            for to in bits(t & !own) {
                moves.push(Move { from, to, promo: None, kind: MoveKind::Normal });
            }
        }
        // King (non-castling)
        let ksq = self.king_square(us);
        for to in bits(king_attacks(ksq) & !own) {
            moves.push(Move { from: ksq, to, promo: None, kind: MoveKind::Normal });
        }
        self.add_castles(&mut moves, us, occ);

        moves
    }

    fn add_pawn_move(&self, moves: &mut Vec<Move>, from: u8, to: u8, promo: bool) {
        if promo {
            for p in [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight] {
                moves.push(Move { from, to, promo: Some(p), kind: MoveKind::Normal });
            }
        } else {
            moves.push(Move { from, to, promo: None, kind: MoveKind::Normal });
        }
    }

    fn add_castles(&self, moves: &mut Vec<Move>, us: Color, occ: Bitboard) {
        let them = us.opposite();
        let (rights_k, rights_q, e, f, g, d, c, b_sq) = match us {
            //                    K    Q   e   f   g   d   c   b
            Color::White => (WK, WQ, 4u8, 5u8, 6u8, 3u8, 2u8, 1u8),
            Color::Black => (BK, BQ, 60, 61, 62, 59, 58, 57),
        };
        // Can't castle out of check.
        if self.is_square_attacked(e, them) {
            return;
        }
        // Kingside: f,g empty; e,f,g not attacked.
        if self.castling & rights_k != 0
            && occ & (bit(f) | bit(g)) == 0
            && !self.is_square_attacked(f, them)
            && !self.is_square_attacked(g, them)
        {
            moves.push(Move { from: e, to: g, promo: None, kind: MoveKind::CastleKing });
        }
        // Queenside: b,c,d empty; e,d,c not attacked.
        if self.castling & rights_q != 0
            && occ & (bit(b_sq) | bit(c) | bit(d)) == 0
            && !self.is_square_attacked(d, them)
            && !self.is_square_attacked(c, them)
        {
            moves.push(Move { from: e, to: c, promo: None, kind: MoveKind::CastleQueen });
        }
    }

    pub fn legal_moves(&mut self) -> Vec<Move> {
        let us = self.side;
        let pseudo = self.pseudo_legal_moves();
        let mut legal = Vec::with_capacity(pseudo.len());
        for mv in pseudo {
            let undo = self.make_move(mv);
            if !self.is_square_attacked(self.king_square(us), us.opposite()) {
                legal.push(mv);
            }
            self.unmake_move(undo);
        }
        legal
    }

    pub fn make_move(&mut self, mv: Move) -> Undo {
        let (color, pt) = self.pieces[mv.from as usize].unwrap();
        let opp = color.opposite();
        let mut undo = Undo {
            mv,
            captured: None,
            castling: self.castling,
            ep: self.ep,
            halfmove: self.halfmove,
            hash: self.hash,
        };

        // Remove old castling / en-passant contributions from the hash; the new
        // ones are XORed back in after those fields are updated below. (Piece
        // moves update the hash automatically via add_piece/remove_piece.)
        self.hash ^= ZOBRIST.castling[self.castling as usize];
        if let Some(ep) = self.ep {
            self.hash ^= ZOBRIST.ep_file[file_of(ep) as usize];
        }

        let captured_sq = if mv.kind == MoveKind::EnPassant {
            match color {
                Color::White => mv.to - 8,
                Color::Black => mv.to + 8,
            }
        } else {
            mv.to
        };
        if let Some((cc, cpt)) = self.pieces[captured_sq as usize] {
            if cc == opp {
                undo.captured = Some(cpt);
                self.remove_piece(captured_sq);
            }
        }

        self.remove_piece(mv.from);
        self.add_piece(mv.to, color, mv.promo.unwrap_or(pt));

        match mv.kind {
            MoveKind::CastleKing => match color {
                Color::White => self.move_piece(7, 5),
                Color::Black => self.move_piece(63, 61),
            },
            MoveKind::CastleQueen => match color {
                Color::White => self.move_piece(0, 3),
                Color::Black => self.move_piece(56, 59),
            },
            _ => {}
        }

        self.castling &= castle_mask(mv.from) & castle_mask(mv.to);

        self.ep = if mv.kind == MoveKind::DoublePush {
            Some(match color {
                Color::White => mv.from + 8,
                Color::Black => mv.from - 8,
            })
        } else {
            None
        };

        // Re-add the (updated) castling / ep contributions and flip side-to-move.
        self.hash ^= ZOBRIST.castling[self.castling as usize];
        if let Some(ep) = self.ep {
            self.hash ^= ZOBRIST.ep_file[file_of(ep) as usize];
        }
        self.hash ^= ZOBRIST.side;

        if pt == PieceType::Pawn || undo.captured.is_some() {
            self.halfmove = 0;
        } else {
            self.halfmove += 1;
        }
        if color == Color::Black {
            self.fullmove += 1;
        }
        self.side = opp;
        undo
    }

    /// Make a "null move": pass the turn (no piece moves). Used by null-move pruning.
    pub fn make_null(&mut self) -> NullUndo {
        let u = NullUndo { ep: self.ep, halfmove: self.halfmove, hash: self.hash };
        if let Some(ep) = self.ep {
            self.hash ^= ZOBRIST.ep_file[file_of(ep) as usize];
        }
        self.ep = None;
        self.hash ^= ZOBRIST.side;
        self.side = self.side.opposite();
        self.halfmove += 1;
        u
    }

    pub fn unmake_null(&mut self, u: NullUndo) {
        self.side = self.side.opposite();
        self.ep = u.ep;
        self.halfmove = u.halfmove;
        self.hash = u.hash;
    }

    pub fn unmake_move(&mut self, undo: Undo) {
        let mv = undo.mv;
        self.side = self.side.opposite(); // restore mover
        let color = self.side;
        let opp = color.opposite();
        if color == Color::Black {
            self.fullmove -= 1;
        }

        match mv.kind {
            MoveKind::CastleKing => match color {
                Color::White => self.move_piece(5, 7),
                Color::Black => self.move_piece(61, 63),
            },
            MoveKind::CastleQueen => match color {
                Color::White => self.move_piece(3, 0),
                Color::Black => self.move_piece(59, 56),
            },
            _ => {}
        }

        let (_, placed_pt) = self.pieces[mv.to as usize].unwrap();
        self.remove_piece(mv.to);
        let original = if mv.promo.is_some() { PieceType::Pawn } else { placed_pt };
        self.add_piece(mv.from, color, original);

        if let Some(cpt) = undo.captured {
            let cap_sq = if mv.kind == MoveKind::EnPassant {
                match color {
                    Color::White => mv.to - 8,
                    Color::Black => mv.to + 8,
                }
            } else {
                mv.to
            };
            self.add_piece(cap_sq, opp, cpt);
        }

        self.castling = undo.castling;
        self.ep = undo.ep;
        self.halfmove = undo.halfmove;
        // Piece restores above touched the hash via add/remove; the saved
        // pre-move hash is authoritative, so just restore it.
        self.hash = undo.hash;
    }
}

fn char_to_piece(c: char) -> (Color, PieceType) {
    let color = if c.is_ascii_uppercase() { Color::White } else { Color::Black };
    let pt = match c.to_ascii_lowercase() {
        'p' => PieceType::Pawn,
        'n' => PieceType::Knight,
        'b' => PieceType::Bishop,
        'r' => PieceType::Rook,
        'q' => PieceType::Queen,
        'k' => PieceType::King,
        _ => panic!("bad FEN piece char: {c}"),
    };
    (color, pt)
}
