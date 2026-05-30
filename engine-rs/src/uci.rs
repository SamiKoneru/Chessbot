//! UCI (Universal Chess Interface) protocol loop.
//!
//! Supports the commands needed to play games under time control: `uci`,
//! `isready`, `ucinewgame`, `position`, `go` (depth / movetime / wtime+btime),
//! `setoption name NnuePath`, and `quit`. The search is synchronous, so true
//! `go infinite` + async `stop` (analysis mode) is approximated with a long
//! time budget rather than an interruptible thread — fine for engine-vs-engine
//! play, which drives the engine with time controls.

use crate::board::Board;
use crate::nnue::Nnue;
use crate::search::Searcher;
use crate::types::{str_to_sq, Color, Move, PieceType};
use std::io::{self, BufRead, Write};
use std::time::{Duration, Instant};

const NAME: &str = "Chessbot 0.1";
const AUTHOR: &str = "SamiKoneru";
const MAX_DEPTH: i32 = 64;

pub fn uci_loop(initial_nnue: Option<String>) {
    let mut board = Board::startpos();
    let mut searcher = Searcher::new();

    if let Some(path) = initial_nnue {
        load_nnue(&mut searcher, &path);
    }

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        let mut tokens = line.split_whitespace();
        match tokens.next() {
            Some("uci") => {
                println!("id name {NAME}");
                println!("id author {AUTHOR}");
                println!("option name NnuePath type string default <empty>");
                println!("uciok");
            }
            Some("isready") => println!("readyok"),
            Some("ucinewgame") => {
                searcher.clear();
                board = Board::startpos();
            }
            Some("setoption") => handle_setoption(line, &mut searcher),
            Some("position") => {
                let rest = line.strip_prefix("position").unwrap_or("").trim();
                board = parse_position(rest);
            }
            Some("go") => {
                let go = parse_go(line);
                let (max_depth, deadline) = allocate(&go, board.side);
                let best = searcher.search_uci(&mut board, max_depth, deadline);
                let best_str = best.map(|m| m.to_uci()).unwrap_or_else(|| "0000".into());
                println!("bestmove {best_str}");
            }
            Some("stop") => {} // synchronous search; nothing in flight to interrupt
            Some("quit") => break,
            _ => {}
        }
        let _ = io::stdout().flush();
    }
}

fn load_nnue(searcher: &mut Searcher, path: &str) {
    match Nnue::load(path) {
        Ok(n) => {
            searcher.set_nnue(n);
            println!("info string loaded NNUE from {path}");
        }
        Err(e) => println!("info string failed to load NNUE {path}: {e}"),
    }
}

fn handle_setoption(line: &str, searcher: &mut Searcher) {
    // setoption name <Name> value <Value>
    if let (Some(n), Some(v)) = (line.find("name "), line.find("value ")) {
        let name = line[n + 5..v].trim();
        let value = line[v + 6..].trim();
        if name.eq_ignore_ascii_case("NnuePath") && !value.is_empty() {
            load_nnue(searcher, value);
        }
    }
}

fn parse_position(rest: &str) -> Board {
    let mut it = rest.split_whitespace().peekable();
    let mut board = match it.next() {
        Some("fen") => {
            let mut fen_parts = Vec::new();
            while let Some(&t) = it.peek() {
                if t == "moves" {
                    break;
                }
                fen_parts.push(t);
                it.next();
            }
            Board::from_fen(&fen_parts.join(" "))
        }
        _ => Board::startpos(), // "startpos" or anything unexpected
    };

    if it.peek() == Some(&"moves") {
        it.next();
        for mv_str in it {
            if let Some(mv) = find_move(&mut board, mv_str) {
                board.make_move(mv);
            }
        }
    }
    board
}

/// Resolve a UCI move string (e.g. "e2e4", "e7e8q", "e1g1") to the matching legal
/// move, which carries the correct kind (castle / en passant / promotion).
fn find_move(board: &mut Board, uci: &str) -> Option<Move> {
    if uci.len() < 4 {
        return None;
    }
    let from = str_to_sq(&uci[0..2]);
    let to = str_to_sq(&uci[2..4]);
    let promo = uci.as_bytes().get(4).map(|&c| match c {
        b'n' => PieceType::Knight,
        b'b' => PieceType::Bishop,
        b'r' => PieceType::Rook,
        _ => PieceType::Queen,
    });
    board
        .legal_moves()
        .into_iter()
        .find(|m| m.from == from && m.to == to && m.promo == promo)
}

#[derive(Default)]
struct GoParams {
    depth: Option<i32>,
    movetime: Option<u64>,
    wtime: Option<u64>,
    btime: Option<u64>,
    winc: Option<u64>,
    binc: Option<u64>,
    infinite: bool,
}

fn parse_go(line: &str) -> GoParams {
    let mut go = GoParams::default();
    let toks: Vec<&str> = line.split_whitespace().collect();
    for i in 0..toks.len() {
        let val = toks.get(i + 1);
        match toks[i] {
            "depth" => go.depth = val.and_then(|s| s.parse().ok()),
            "movetime" => go.movetime = val.and_then(|s| s.parse().ok()),
            "wtime" => go.wtime = val.and_then(|s| s.parse().ok()),
            "btime" => go.btime = val.and_then(|s| s.parse().ok()),
            "winc" => go.winc = val.and_then(|s| s.parse().ok()),
            "binc" => go.binc = val.and_then(|s| s.parse().ok()),
            "infinite" => go.infinite = true,
            _ => {}
        }
    }
    go
}

/// Decide the search depth limit and (optional) wall-clock deadline.
fn allocate(go: &GoParams, side: Color) -> (i32, Option<Instant>) {
    let max_depth = go.depth.unwrap_or(MAX_DEPTH);

    if let Some(mt) = go.movetime {
        return (max_depth, Some(Instant::now() + Duration::from_millis(mt)));
    }

    let (time, inc) = match side {
        Color::White => (go.wtime, go.winc),
        Color::Black => (go.btime, go.binc),
    };
    if let Some(t) = time {
        // Simple budget: ~1/25 of the clock plus most of the increment, leaving
        // a small safety margin so we never flag.
        let budget = (t / 25 + inc.unwrap_or(0) * 3 / 4)
            .min(t.saturating_sub(50))
            .max(10);
        return (max_depth, Some(Instant::now() + Duration::from_millis(budget)));
    }

    if go.infinite {
        // No async stop in this synchronous engine — cap with a long budget.
        return (max_depth, Some(Instant::now() + Duration::from_secs(30)));
    }

    // Bare `go depth N` → fixed depth; bare `go` → a default 3s think.
    if go.depth.is_some() {
        (max_depth, None)
    } else {
        (MAX_DEPTH, Some(Instant::now() + Duration::from_millis(3000)))
    }
}
