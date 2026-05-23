//! Zobrist hashing keys, generated deterministically at compile time.
//!
//! The board maintains its hash incrementally across make/unmake (XORing these
//! keys), so the transposition table gets a cheap, stable position id.

pub struct Zobrist {
    pub pieces: [[[u64; 64]; 6]; 2], // [color][piece_type][square]
    pub side: u64,                   // XORed in when it's Black to move
    pub castling: [u64; 16],         // indexed by the 4-bit castling-rights mask
    pub ep_file: [u64; 8],           // indexed by en-passant file
}

/// splitmix64 step: returns (value, next_state). Pure so it works in const fn.
const fn next(state: u64) -> (u64, u64) {
    let s = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31), s)
}

const fn generate() -> Zobrist {
    let mut st: u64 = 0x9D2C_5680_1F3A_77E1;
    let mut pieces = [[[0u64; 64]; 6]; 2];
    let mut c = 0;
    while c < 2 {
        let mut p = 0;
        while p < 6 {
            let mut sq = 0;
            while sq < 64 {
                let (v, ns) = next(st);
                st = ns;
                pieces[c][p][sq] = v;
                sq += 1;
            }
            p += 1;
        }
        c += 1;
    }
    let (side, ns) = next(st);
    st = ns;
    let mut castling = [0u64; 16];
    let mut i = 0;
    while i < 16 {
        let (v, ns) = next(st);
        st = ns;
        castling[i] = v;
        i += 1;
    }
    let mut ep_file = [0u64; 8];
    let mut i = 0;
    while i < 8 {
        let (v, ns) = next(st);
        st = ns;
        ep_file[i] = v;
        i += 1;
    }
    Zobrist { pieces, side, castling, ep_file }
}

pub static ZOBRIST: Zobrist = generate();
