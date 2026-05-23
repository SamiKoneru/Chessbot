//! Core value types: colors, pieces, squares, and moves.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    White,
    Black,
}

impl Color {
    #[inline]
    pub fn opposite(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceType {
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Board square, 0..63, LERF (a1 = 0, h8 = 63).
pub type Square = u8;

#[inline]
pub fn file_of(sq: Square) -> u8 {
    sq & 7
}
#[inline]
pub fn rank_of(sq: Square) -> u8 {
    sq >> 3
}
#[inline]
pub fn make_square(file: u8, rank: u8) -> Square {
    rank * 8 + file
}

pub fn sq_to_str(sq: Square) -> String {
    let f = (b'a' + file_of(sq)) as char;
    let r = (b'1' + rank_of(sq)) as char;
    format!("{f}{r}")
}

pub fn str_to_sq(s: &str) -> Square {
    let bytes = s.as_bytes();
    let file = bytes[0] - b'a';
    let rank = bytes[1] - b'1';
    make_square(file, rank)
}

/// How a move behaves beyond a plain from→to relocation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoveKind {
    Normal,
    DoublePush,
    EnPassant,
    CastleKing,
    CastleQueen,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promo: Option<PieceType>,
    pub kind: MoveKind,
}

impl Move {
    pub fn to_uci(self) -> String {
        let mut s = sq_to_str(self.from);
        s.push_str(&sq_to_str(self.to));
        if let Some(p) = self.promo {
            s.push(match p {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                _ => 'q',
            });
        }
        s
    }
}
