//! Throughput ladder for batched option pricing.
//!
//! Prices a heterogeneous book three ways — scalar reference, AVX2 (4×f64),
//! AVX2 + threads — and reports options/second and speedups. The SIMD result
//! is verified bit-identical to the scalar result before timing, so the
//! speedups are honest: same work, same answer, faster.
//!
//! Run: `cargo run --release --example bench`

use std::time::Instant;
use tile_fd_pricer::batch::{price_batch, price_batch_parallel, price_one, stable_steps};
use tile_fd_pricer::black_scholes::Params;
use tile_fd_pricer::OptionType;

fn book(count: usize) -> Vec<Params> {
    (0..count)
        .map(|i| {
            let f = i as f64 / count as f64;
            Params {
                spot: 100.0,
                strike: 60.0 + 80.0 * f,
                rate: 0.05,
                dividend: 0.0,
                vol: 0.15 + 0.30 * ((i % 11) as f64 / 10.0),
                t: 1.0,
                kind: if i % 2 == 0 { OptionType::Call } else { OptionType::Put },
            }
        })
        .collect()
}

fn time<F: FnOnce() -> Vec<f64>>(f: F) -> (Vec<f64>, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64())
}

fn main() {
    let count = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let n = 256;
    let num_std = 6.0;
    let opts = book(count);
    let steps = stable_steps(&opts, n, num_std);

    let nodes = (count * (n - 1) * steps) as f64; // interior stencil updates

    println!("Book: {count} options  •  grid {n} nodes × {steps} steps  •  {:.2e} stencil updates\n", nodes);

    let (scalar, t_scalar) = time(|| opts.iter().map(|p| price_one(p, n, num_std, steps)).collect());
    let (simd, t_simd) = time(|| price_batch(&opts, n, num_std, steps));
    let (par, t_par) = time(|| price_batch_parallel(&opts, n, num_std, steps));

    // Honesty check: SIMD and parallel must equal the scalar reference exactly.
    assert_eq!(simd, scalar, "SIMD result diverged from scalar");
    assert_eq!(par, scalar, "parallel result diverged from scalar");

    let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    let row = |name: &str, t: f64| {
        println!(
            "{:<22}{:>9.3} s{:>16.0} opt/s{:>14.2} Gupd/s{:>10.2}×",
            name,
            t,
            count as f64 / t,
            nodes / t / 1e9,
            t_scalar / t
        );
    };
    println!("{:<22}{:>11}{:>21}{:>16}{:>11}", "", "time", "throughput", "stencil", "speedup");
    row("scalar reference", t_scalar);
    row("AVX2 (4×f64)", t_simd);
    row(&format!("AVX2 + {threads} threads"), t_par);

    println!("\nResults verified bit-identical across all three paths.");
    println!("Sample: option[0] = {:.6}", scalar[0]);
}
