//! Temporary entry point: perft + search demo, and (with a weights path arg) an
//! NNUE eval dump for cross-checking against the Python evaluator. UCI loop next.

use chessbot_engine::board::Board;
use chessbot_engine::nnue::Nnue;
use chessbot_engine::perft::perft;
use chessbot_engine::search::Searcher;
use std::time::Instant;

const DEMO_FENS: &[(&str, &str)] = &[
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("white +Q", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("italian", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("midgame", "r1bq1rk1/pp2bppp/2n2n2/2pp4/3P4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 7"),
];

fn main() {
    let weights = std::env::args().nth(1);

    let mut b = Board::startpos();
    println!("perft from start position:");
    for depth in 1..=5 {
        let t = Instant::now();
        let nodes = perft(&mut b, depth);
        let secs = t.elapsed().as_secs_f64();
        println!("  depth {depth}: {nodes:>10} nodes  {secs:.2}s  ({:.0} nps)", nodes as f64 / secs.max(1e-9));
    }

    if let Some(path) = weights {
        // NNUE mode: dump evals (for comparison with Python) and run a search.
        let nnue = Nnue::load(&path).expect("failed to load NNUE weights");
        println!("\nNNUE eval (stm-relative cp) — compare with Python:");
        for (name, fen) in DEMO_FENS {
            let board = Board::from_fen(fen);
            println!("  {name:<10} {}", nnue.evaluate(&board));
        }

        println!("\nNNUE search demo:");
        for (name, fen) in DEMO_FENS {
            let mut board = Board::from_fen(fen);
            let mut s = Searcher::new();
            s.set_nnue(Nnue::load(&path).unwrap());
            let t = Instant::now();
            let (mv, score) = s.search(&mut board, 6);
            let secs = t.elapsed().as_secs_f64();
            println!(
                "  {name:<10} best={:<6} score={score:>6}  depth 6  {:>9} nodes  {secs:.2}s  ({:.0} nps)",
                mv.map(|m| m.to_uci()).unwrap_or_else(|| "none".into()),
                s.nodes,
                s.nodes as f64 / secs.max(1e-9)
            );
        }
    } else {
        println!("\nsearch demo (material eval — pass a weights path to use NNUE):");
        for (name, fen) in DEMO_FENS {
            let mut board = Board::from_fen(fen);
            let mut s = Searcher::new();
            let t = Instant::now();
            let (mv, score) = s.search(&mut board, 6);
            let secs = t.elapsed().as_secs_f64();
            println!(
                "  {name:<10} best={:<6} score={score:>6}  {:>8} nodes  {secs:.2}s",
                mv.map(|m| m.to_uci()).unwrap_or_else(|| "none".into()),
                s.nodes,
            );
        }
    }
}
