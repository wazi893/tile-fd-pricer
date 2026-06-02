//! Batched option pricing — the throughput path.
//!
//! A single option is microseconds of work; the real problem a pricing desk
//! has is valuing a *book* (an option chain, a vol surface) of thousands of
//! contracts. The natural vectorisation is therefore **across options**: pack
//! `W` options into the SIMD lanes and march them through the same explicit
//! stencil in lockstep. Each lane carries its own strike/vol/grid, so the
//! per-lane coefficients `(pd, pm, pu)` are vectors, but the kernel is still
//! the one stencil line — now retiring `W` options per instruction.
//!
//! Three levels form the performance ladder (see `examples/bench.rs`):
//! 1. scalar reference ([`price_one`] in a loop),
//! 2. AVX2 (4×f64) batched kernel ([`price_batch`]),
//! 3. AVX2 + threads ([`price_batch_parallel`]).
//!
//! The AVX2 path is **bit-identical** to the scalar reference: both accumulate
//! the stencil in the same fused-multiply-add order, so vectorisation buys
//! throughput without perturbing a single ULP — the same determinism contract
//! the engine holds across its backends.

use crate::black_scholes::Params;
use crate::OptionType;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD width for f64 on AVX2.
pub const LANES: usize = 4;

/// Per-option discretisation constants, precomputed once and held constant
/// across every time step (the log-space PDE has constant coefficients).
#[derive(Clone, Copy)]
struct Lane {
    pd: f64,
    pm: f64,
    pu: f64,
    dtau: f64,
    s0: f64, // underlying at the low boundary
    sn: f64, // underlying at the high boundary
    strike: f64,
    rate: f64,
    div: f64,
    kind: OptionType,
}

fn lane(p: &Params, n: usize, num_std: f64, steps: usize) -> Lane {
    let center = p.spot.ln();
    let half = (num_std * p.vol * p.t.sqrt() + (p.rate - p.dividend).abs() * p.t).max(0.1);
    let dx = 2.0 * half / n as f64;
    let dtau = p.t / steps as f64;
    let a = 0.5 * p.vol * p.vol;
    let b = p.rate - p.dividend - 0.5 * p.vol * p.vol;
    let alpha = a / (dx * dx);
    let beta = b / (2.0 * dx);
    Lane {
        pd: dtau * (alpha - beta),
        pm: 1.0 - dtau * (2.0 * alpha + p.rate),
        pu: dtau * (alpha + beta),
        dtau,
        s0: (center - half).exp(),
        sn: (center + half).exp(),
        strike: p.strike,
        rate: p.rate,
        div: p.dividend,
        kind: p.kind,
    }
}

impl Lane {
    /// Underlying price at node `j` of this option's grid.
    fn s_at(&self, j: usize, n: usize) -> f64 {
        // log-uniform: s0 * exp(j*dx), and dx = ln(sn/s0)/n.
        let dx = (self.sn / self.s0).ln() / n as f64;
        self.s0 * (j as f64 * dx).exp()
    }
    fn payoff(&self, s: f64) -> f64 {
        match self.kind {
            OptionType::Call => (s - self.strike).max(0.0),
            OptionType::Put => (self.strike - s).max(0.0),
        }
    }
    /// Dirichlet boundary `(low, high)` at time-to-maturity `tau`.
    fn boundary(&self, tau: f64) -> (f64, f64) {
        let dr = (-self.rate * tau).exp();
        let dq = (-self.div * tau).exp();
        match self.kind {
            OptionType::Call => (0.0, self.sn * dq - self.strike * dr),
            OptionType::Put => ((self.strike * dr - self.s0 * dq).max(0.0), 0.0),
        }
    }
}

/// Minimum number of explicit time steps for stability across the whole batch.
///
/// Explicit Euler is stable only when `dτ·(2a/dx² + r) ≤ 1`; the binding
/// constraint is the option with the finest grid relative to its volatility.
pub fn stable_steps(opts: &[Params], n: usize, num_std: f64) -> usize {
    let mut steps = 1usize;
    for p in opts {
        let half = (num_std * p.vol * p.t.sqrt() + (p.rate - p.dividend).abs() * p.t).max(0.1);
        let dx = 2.0 * half / n as f64;
        let a = 0.5 * p.vol * p.vol;
        let dt_max = 1.0 / (2.0 * a / (dx * dx) + p.rate.max(0.0));
        steps = steps.max((p.t / (0.9 * dt_max)).ceil() as usize);
    }
    steps.max(1)
}

/// Scalar reference pricer for a single European option (the naive baseline,
/// and the bit-exact oracle the SIMD path is checked against).
pub fn price_one(p: &Params, n: usize, num_std: f64, steps: usize) -> f64 {
    let lc = lane(p, n, num_std, steps);
    let mut v = vec![0.0f64; n + 1];
    let mut w = vec![0.0f64; n + 1];
    for (j, slot) in v.iter_mut().enumerate() {
        *slot = lc.payoff(lc.s_at(j, n));
    }
    for step in 1..=steps {
        let tau = step as f64 * lc.dtau;
        for j in 1..n {
            // pm*c + pd*l + pu*r, fused in this exact order to match AVX2.
            let acc = lc.pm * v[j];
            let acc = lc.pd.mul_add(v[j - 1], acc);
            w[j] = lc.pu.mul_add(v[j + 1], acc);
        }
        let (lo, hi) = lc.boundary(tau);
        w[0] = lo;
        w[n] = hi;
        std::mem::swap(&mut v, &mut w);
    }
    v[n / 2]
}

/// Structure-of-arrays *scalar* pricer for one quad — identical memory layout
/// and FMA order to the AVX2 kernel, but with a plain scalar loop over the four
/// lanes instead of vector instructions.
///
/// Its only purpose is benchmark *attribution*: comparing `price_one` (naive,
/// per-option allocation, array-of-structs) against this isolates the
/// data-layout win, and comparing this against the AVX2 kernel isolates the
/// pure-SIMD win. Because the FMA order matches, it is bit-identical to both.
#[allow(clippy::needless_range_loop)] // lane index also addresses the SoA buffer
fn price_quad_scalar(opts: &[Params; LANES], n: usize, num_std: f64, steps: usize) -> [f64; LANES] {
    let lc: [Lane; LANES] = std::array::from_fn(|i| lane(&opts[i], n, num_std, steps));
    // Hoist coefficients into locals (the AVX2 kernel holds them in registers),
    // so the comparison reflects SIMD, not coefficient reloads.
    let pd: [f64; LANES] = std::array::from_fn(|l| lc[l].pd);
    let pm: [f64; LANES] = std::array::from_fn(|l| lc[l].pm);
    let pu: [f64; LANES] = std::array::from_fn(|l| lc[l].pu);
    let stride = n + 1;
    let mut src = vec![0.0f64; stride * LANES];
    let mut dst = vec![0.0f64; stride * LANES];
    for j in 0..=n {
        for l in 0..LANES {
            src[j * LANES + l] = lc[l].payoff(lc[l].s_at(j, n));
        }
    }
    for step in 1..=steps {
        for j in 1..n {
            let base = j * LANES;
            for l in 0..LANES {
                // SAFETY: for j in 1..n, base-LANES .. base+LANES+l all lie in
                // 0..stride*LANES. Unchecked to match the AVX2 kernel's safety
                // profile, so the benchmark isolates SIMD, not bounds checks.
                unsafe {
                    let c = *src.get_unchecked(base + l);
                    let left = *src.get_unchecked(base - LANES + l);
                    let right = *src.get_unchecked(base + LANES + l);
                    let acc = pm[l] * c;
                    let acc = pd[l].mul_add(left, acc);
                    *dst.get_unchecked_mut(base + l) = pu[l].mul_add(right, acc);
                }
            }
        }
        for (l, lane) in lc.iter().enumerate() {
            let tau = step as f64 * lane.dtau;
            let (lo, hi) = lane.boundary(tau);
            dst[l] = lo;
            dst[n * LANES + l] = hi;
        }
        std::mem::swap(&mut src, &mut dst);
    }
    let mid = (n / 2) * LANES;
    std::array::from_fn(|l| src[mid + l])
}

/// Batch pricer using the SoA-scalar quad kernel (no SIMD). See
/// [`price_quad_scalar`] — used for benchmark attribution.
pub fn price_batch_scalar_soa(opts: &[Params], n: usize, num_std: f64, steps: usize) -> Vec<f64> {
    let steps = if steps == 0 {
        stable_steps(opts, n, num_std)
    } else {
        steps
    };
    let mut out = vec![0.0f64; opts.len()];
    for (c, chunk) in opts.chunks(LANES).enumerate() {
        let mut quad = [chunk[0]; LANES];
        quad[..chunk.len()].copy_from_slice(chunk);
        let res = price_quad_scalar(&quad, n, num_std, steps);
        out[c * LANES..c * LANES + chunk.len()].copy_from_slice(&res[..chunk.len()]);
    }
    out
}

/// Price a batch of European options. Uses the AVX2 kernel when the CPU
/// supports it, otherwise falls back to the scalar reference. `steps` of 0
/// auto-selects the stable minimum.
pub fn price_batch(opts: &[Params], n: usize, num_std: f64, steps: usize) -> Vec<f64> {
    let steps = if steps == 0 {
        stable_steps(opts, n, num_std)
    } else {
        steps
    };
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return price_batch_avx2(opts, n, num_std, steps);
        }
    }
    opts.iter()
        .map(|p| price_one(p, n, num_std, steps))
        .collect()
}

/// AVX2 dispatcher: chunk the book into groups of four and price each quad.
#[cfg(target_arch = "x86_64")]
fn price_batch_avx2(opts: &[Params], n: usize, num_std: f64, steps: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; opts.len()];
    for (c, chunk) in opts.chunks(LANES).enumerate() {
        let mut quad = [chunk[0]; LANES];
        quad[..chunk.len()].copy_from_slice(chunk);
        // SAFETY: guarded by is_x86_feature_detected in price_batch.
        let res = unsafe { price_quad_avx2(&quad, n, num_std, steps) };
        out[c * LANES..c * LANES + chunk.len()].copy_from_slice(&res[..chunk.len()]);
    }
    out
}

/// Price exactly four options at once with one AVX2 lane each.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn price_quad_avx2(
    opts: &[Params; LANES],
    n: usize,
    num_std: f64,
    steps: usize,
) -> [f64; LANES] {
    let lc: [Lane; LANES] = std::array::from_fn(|i| lane(&opts[i], n, num_std, steps));

    let pdv = _mm256_set_pd(lc[3].pd, lc[2].pd, lc[1].pd, lc[0].pd);
    let pmv = _mm256_set_pd(lc[3].pm, lc[2].pm, lc[1].pm, lc[0].pm);
    let puv = _mm256_set_pd(lc[3].pu, lc[2].pu, lc[1].pu, lc[0].pu);

    // Node-major layout: V[j*LANES + lane]. Two buffers, ping-ponged.
    let stride = n + 1;
    let mut a = vec![0.0f64; stride * LANES];
    let mut b = vec![0.0f64; stride * LANES];
    for j in 0..=n {
        for l in 0..LANES {
            a[j * LANES + l] = lc[l].payoff(lc[l].s_at(j, n));
        }
    }

    let mut src = a.as_mut_ptr();
    let mut dst = b.as_mut_ptr();
    for step in 1..=steps {
        for j in 1..n {
            let base = j * LANES;
            let lvec = _mm256_loadu_pd(src.add(base - LANES));
            let cvec = _mm256_loadu_pd(src.add(base));
            let rvec = _mm256_loadu_pd(src.add(base + LANES));
            let mut acc = _mm256_mul_pd(pmv, cvec);
            acc = _mm256_fmadd_pd(pdv, lvec, acc);
            acc = _mm256_fmadd_pd(puv, rvec, acc);
            _mm256_storeu_pd(dst.add(base), acc);
        }
        // Per-lane Dirichlet boundaries (scalar; 8 stores per step is noise).
        for (l, lane) in lc.iter().enumerate() {
            let tau = step as f64 * lane.dtau;
            let (lo, hi) = lane.boundary(tau);
            *dst.add(l) = lo;
            *dst.add(n * LANES + l) = hi;
        }
        std::mem::swap(&mut src, &mut dst);
    }

    let mid = (n / 2) * LANES;
    std::array::from_fn(|l| *src.add(mid + l))
}

/// AVX2 batch pricing spread across worker threads (level 3 of the ladder).
pub fn price_batch_parallel(opts: &[Params], n: usize, num_std: f64, steps: usize) -> Vec<f64> {
    let steps = if steps == 0 {
        stable_steps(opts, n, num_std)
    } else {
        steps
    };
    let threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    let total = opts.len();
    if threads <= 1 || total < LANES * 4 {
        return price_batch(opts, n, num_std, steps);
    }

    // Split into LANES-aligned contiguous slices, one per thread.
    let mut out = vec![0.0f64; total];
    let quads = total.div_ceil(LANES);
    let quads_per = quads.div_ceil(threads);
    let chunk = quads_per * LANES;

    std::thread::scope(|scope| {
        let mut opt_rest = opts;
        let mut out_rest = out.as_mut_slice();
        while !opt_rest.is_empty() {
            let take = chunk.min(opt_rest.len());
            let (o_head, o_tail) = opt_rest.split_at(take);
            let (r_head, r_tail) = out_rest.split_at_mut(take);
            scope.spawn(move || {
                let res = price_batch(o_head, n, num_std, steps);
                r_head.copy_from_slice(&res);
            });
            opt_rest = o_tail;
            out_rest = r_tail;
        }
    });
    out
}
