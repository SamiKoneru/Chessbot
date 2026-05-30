//! Entry point. Default mode is UCI (what chess GUIs expect). A `bench`
//! subcommand runs the perft + search demo for quick local checks.
//!
//!   chessbot-engine                 # UCI loop (load NNUE via setoption NnuePath or NNUE_PATH env)
//!   chessbot-engine <nnue.bin>      # UCI loop, preloading the given NNUE weights
//!   chessbot-engine bench           # perft + material-eval search demo
//!   chessbot-engine bench <nnue.bin># perft + NNUE search demo

use chessbot_engine::board::Board;
use chessbot_engine::nnue::Nnue;
use chessbot_engine::perft::perft;
use chessbot_engine::search::Searcher;
use chessbot_engine::uci;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("bench") {
        bench(args.get(2).map(|s| s.as_str()));
    } else {
        // UCI mode. Optional first arg or NNUE_PATH env var preloads weights.
        let nnue_path = args.get(1).cloned().or_else(|| std::env::var("NNUE_PATH").ok());
        uci::uci_loop(nnue_path);
    }
}

fn bench(nnue_path: Option<&str>) {
    let mut b = Board::startpos();
    println!("perft from start position:");
    for depth in 1..=5 {
        let t = Instant::now();
        let nodes = perft(&mut b, depth);
        let secs = t.elapsed().as_secs_f64();
        println!("  depth {depth}: {nodes:>10} nodes  {secs:.2}s  ({:.0} nps)", nodes as f64 / secs.max(1e-9));
    }

    let fens = [
        ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("white +Q", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("italian", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
        ("midgame", "r1bq1rk1/pp2bppp/2n2n2/2pp4/3P4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 7"),
    ];

    // Direct leaf-eval print (cross-check against the Python NNUE evaluator).
    if let Some(p) = nnue_path {
        let nnue = Nnue::load(p).expect("failed to load NNUE");
        println!("\nNNUE leaf eval (stm-relative cp) — compare with the Python evaluator:");
        for (name, fen) in fens {
            let board = Board::from_fen(fen);
            println!("  {name:<10} {}", nnue.evaluate(&board));
        }
    }

    let label = if nnue_path.is_some() { "NNUE" } else { "material" };
    println!("\nsearch demo ({label} eval), depth 6:");
    for (name, fen) in fens {
        let mut board = Board::from_fen(fen);
        let mut s = Searcher::new();
        if let Some(p) = nnue_path {
            s.set_nnue(Nnue::load(p).expect("failed to load NNUE"));
        }
        let t = Instant::now();
        let (mv, score) = s.search(&mut board, 6);
        let secs = t.elapsed().as_secs_f64();
        println!(
            "  {name:<10} best={:<6} score={score:>6}  {:>9} nodes  {secs:.2}s",
            mv.map(|m| m.to_uci()).unwrap_or_else(|| "none".into()),
            s.nodes,
        );
    }
}
