//! Price a sample option and show the finite-difference result beside the
//! closed-form Black–Scholes value, for both European and American exercise.

use tile_fd_pricer::black_scholes::{self, Params};
use tile_fd_pricer::fd::{self, FdConfig, Scheme};
use tile_fd_pricer::{Exercise, OptionType};

fn main() {
    let p = Params {
        spot: 100.0,
        strike: 100.0,
        rate: 0.05,
        dividend: 0.0,
        vol: 0.20,
        t: 1.0,
        kind: OptionType::Put,
    };

    let cfg = FdConfig {
        n_space: 800,
        n_time: 800,
        num_std: 8.0,
        scheme: Scheme::CrankNicolson,
    };

    println!(
        "Contract: {:?}  S={} K={} r={} q={} σ={} T={}",
        p.kind, p.spot, p.strike, p.rate, p.dividend, p.vol, p.t
    );
    println!(
        "Grid: {} space × {} time nodes ({:?})\n",
        cfg.n_space, cfg.n_time, cfg.scheme
    );

    let exact = black_scholes::price(&p);
    let euro = fd::solve(&p, Exercise::European, &cfg);
    let amer = fd::solve(&p, Exercise::American, &cfg);

    println!("{:<22}{:>12}{:>12}{:>12}", "", "price", "delta", "gamma");
    println!(
        "{:<22}{:>12.6}{:>12.6}{:>12.6}",
        "Black–Scholes (exact)", exact.price, exact.delta, exact.gamma
    );
    println!(
        "{:<22}{:>12.6}{:>12.6}{:>12.6}",
        "FD European", euro.price, euro.delta, euro.gamma
    );
    println!(
        "{:<22}{:>12.6}{:>12}{:>12}",
        "FD American", amer.price, "", ""
    );
    println!();
    println!(
        "European price error vs closed form : {:.2e}",
        (euro.price - exact.price).abs()
    );
    println!(
        "American early-exercise premium      : {:.6}",
        amer.price - euro.price
    );
}
