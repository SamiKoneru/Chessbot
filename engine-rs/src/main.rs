//! Temporary entry point: run perft as a smoke test until the UCI loop lands.

use chessbot_engine::board::Board;
use chessbot_engine::perft::perft;
use std::time::Instant;

fn main() {
    let mut b = Board::startpos();
    println!("perft from start position:");
    for depth in 1..=5 {
        let t = Instant::now();
        let nodes = perft(&mut b, depth);
        let secs = t.elapsed().as_secs_f64();
        println!(
            "  depth {depth}: {nodes:>10} nodes  {:.2}s  ({:.0} nps)",
            secs,
            nodes as f64 / secs.max(1e-9)
        );
    }
}
