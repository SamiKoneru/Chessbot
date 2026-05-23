//! Temporary entry point: perft benchmark + a search demo, until the UCI loop lands.

use chessbot_engine::board::Board;
use chessbot_engine::perft::perft;
use chessbot_engine::search::Searcher;
use std::time::Instant;

fn main() {
    let mut b = Board::startpos();
    println!("perft from start position:");
    for depth in 1..=5 {
        let t = Instant::now();
        let nodes = perft(&mut b, depth);
        let secs = t.elapsed().as_secs_f64();
        println!(
            "  depth {depth}: {nodes:>10} nodes  {secs:.2}s  ({:.0} nps)",
            nodes as f64 / secs.max(1e-9)
        );
    }

    println!("\nsearch demo (material eval):");
    let positions = [
        ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("mate in 1", "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"),
        ("italian", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ];
    for (name, fen) in positions {
        let mut board = Board::from_fen(fen);
        let mut s = Searcher::new();
        let t = Instant::now();
        let (mv, score) = s.search(&mut board, 6);
        let secs = t.elapsed().as_secs_f64();
        println!(
            "  {name:<10} best={:<6} score={score:>7}  depth 6  {:>9} nodes  {secs:.2}s  ({:.0} nps)",
            mv.map(|m| m.to_uci()).unwrap_or_else(|| "none".into()),
            s.nodes,
            s.nodes as f64 / secs.max(1e-9)
        );
    }
}
