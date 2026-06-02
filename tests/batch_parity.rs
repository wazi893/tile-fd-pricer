//! The batched/SIMD pricer must agree with both the scalar reference and the
//! closed-form Black–Scholes price. The headline guarantee is **bit-identity**:
//! the AVX2 path reproduces the scalar reference to the last ULP, because both
//! accumulate the stencil in the same fused-multiply-add order.

use tile_fd_pricer::batch::{self, price_batch, price_batch_parallel, price_one, stable_steps};
use tile_fd_pricer::black_scholes::{self, Params};
use tile_fd_pricer::OptionType;

/// A heterogeneous option chain: varied strike and volatility, one expiry.
fn book(count: usize) -> Vec<Params> {
    (0..count)
        .map(|i| {
            let f = i as f64 / count as f64;
            Params {
                spot: 100.0,
                strike: 60.0 + 80.0 * f, // 60 … 140
                rate: 0.05,
                dividend: 0.0,
                vol: 0.15 + 0.30 * ((i % 7) as f64 / 6.0), // 0.15 … 0.45
                t: 1.0,
                kind: if i % 2 == 0 {
                    OptionType::Call
                } else {
                    OptionType::Put
                },
            }
        })
        .collect()
}

#[test]
fn simd_is_bit_identical_to_scalar() {
    let opts = book(37); // deliberately not a multiple of the SIMD width
    let n = 256;
    let num_std = 6.0;
    let steps = stable_steps(&opts, n, num_std);

    let simd = price_batch(&opts, n, num_std, steps);
    let scalar: Vec<f64> = opts
        .iter()
        .map(|p| price_one(p, n, num_std, steps))
        .collect();

    for (i, (a, b)) in simd.iter().zip(&scalar).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "option {i}: simd {a} != scalar {b} (Δ={:e})",
            (a - b).abs()
        );
    }
}

#[test]
fn soa_scalar_is_bit_identical_to_scalar() {
    // The attribution baseline must match the naive reference exactly, so the
    // benchmark ladder compares like with like.
    let opts = book(37);
    let n = 256;
    let num_std = 6.0;
    let steps = stable_steps(&opts, n, num_std);

    let soa = batch::price_batch_scalar_soa(&opts, n, num_std, steps);
    let scalar: Vec<f64> = opts
        .iter()
        .map(|p| price_one(p, n, num_std, steps))
        .collect();

    for (i, (a, b)) in soa.iter().zip(&scalar).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "option {i}: soa {a} != scalar {b}");
    }
}

#[test]
fn parallel_matches_single_threaded() {
    let opts = book(200);
    let n = 200;
    let (num_std, steps) = (6.0, 400);
    let single = price_batch(&opts, n, num_std, steps);
    let many = price_batch_parallel(&opts, n, num_std, steps);
    assert_eq!(single, many, "threading changed the result");
}

#[test]
fn batch_matches_closed_form() {
    // Same contracts, priced analytically — the batch must track Black–Scholes.
    let opts = book(64);
    let n = 400;
    let num_std = 8.0;
    let steps = stable_steps(&opts, n, num_std);
    let fd = price_batch(&opts, n, num_std, steps);

    for (i, p) in opts.iter().enumerate() {
        let exact = black_scholes::price(p).price;
        assert!(
            (fd[i] - exact).abs() < 1e-2,
            "option {i} ({:?} K={}): FD {} vs exact {}",
            p.kind,
            p.strike,
            fd[i],
            exact
        );
    }
}

#[test]
fn lanes_width_is_four() {
    assert_eq!(batch::LANES, 4);
}
