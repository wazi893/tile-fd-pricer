//! Closed-form Black–Scholes–Merton prices and Greeks.
//!
//! This module is the *correctness oracle* for the finite-difference solver in
//! [`crate::fd`]. European options have an exact analytic price, so the FD grid
//! can be validated against it to machine-relevant precision and — more
//! importantly — checked for the correct discretization *convergence order*.

use crate::{Exercise, OptionType};

/// Standard normal cumulative distribution function `N(x)`.
///
/// Abramowitz & Stegun 26.2.17 — the formula used in Hull's *Options, Futures,
/// and Other Derivatives*. Maximum absolute error < 7.5e-8, which is far below
/// the discretization error of any practical FD grid, so it is an effective
/// ground truth for validation.
pub fn norm_cdf(x: f64) -> f64 {
    if x < 0.0 {
        return 1.0 - norm_cdf(-x);
    }
    const B1: f64 = 0.319381530;
    const B2: f64 = -0.356563782;
    const B3: f64 = 1.781477937;
    const B4: f64 = -1.821255978;
    const B5: f64 = 1.330274429;
    const P: f64 = 0.2316419;

    let t = 1.0 / (1.0 + P * x);
    let poly = t * (B1 + t * (B2 + t * (B3 + t * (B4 + t * B5))));
    1.0 - norm_pdf(x) * poly
}

/// Standard normal probability density function `φ(x)`.
pub fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1/sqrt(2π)
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Market and contract parameters shared by the analytic and FD pricers.
///
/// All rates are continuously compounded and annualised; `t` is time to
/// maturity in years.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Spot price of the underlying, `S`.
    pub spot: f64,
    /// Strike price, `K`.
    pub strike: f64,
    /// Risk-free rate, `r`.
    pub rate: f64,
    /// Continuous dividend yield, `q`.
    pub dividend: f64,
    /// Volatility, `σ`.
    pub vol: f64,
    /// Time to maturity in years, `T`.
    pub t: f64,
    /// Call or put.
    pub kind: OptionType,
}

impl Params {
    /// Intrinsic payoff at a given underlying price.
    pub fn payoff(&self, s: f64) -> f64 {
        match self.kind {
            OptionType::Call => (s - self.strike).max(0.0),
            OptionType::Put => (self.strike - s).max(0.0),
        }
    }
}

/// Result of an analytic price: value plus the first/second-order Greeks that
/// the FD grid will be cross-checked against.
#[derive(Clone, Copy, Debug)]
pub struct Analytic {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
}

/// Exact Black–Scholes–Merton price and Greeks for a European option.
///
/// Only valid for [`Exercise::European`]; American options have no closed form
/// and must be priced on the grid.
pub fn price(p: &Params) -> Analytic {
    let Params {
        spot: s,
        strike: k,
        rate: r,
        dividend: q,
        vol,
        t,
        kind,
    } = *p;

    // Degenerate guards: zero time or zero vol collapse to discounted intrinsic.
    if t <= 0.0 || vol <= 0.0 {
        let fwd = s * (-q * t).exp();
        let disc_k = k * (-r * t).exp();
        let price = match kind {
            OptionType::Call => (fwd - disc_k).max(0.0),
            OptionType::Put => (disc_k - fwd).max(0.0),
        };
        return Analytic {
            price,
            delta: 0.0,
            gamma: 0.0,
        };
    }

    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;
    let disc_r = (-r * t).exp();
    let disc_q = (-q * t).exp();

    let (price, delta) = match kind {
        OptionType::Call => (
            s * disc_q * norm_cdf(d1) - k * disc_r * norm_cdf(d2),
            disc_q * norm_cdf(d1),
        ),
        OptionType::Put => (
            k * disc_r * norm_cdf(-d2) - s * disc_q * norm_cdf(-d1),
            -disc_q * norm_cdf(-d1),
        ),
    };
    // Gamma is identical for calls and puts.
    let gamma = disc_q * norm_pdf(d1) / (s * vol * sqrt_t);

    Analytic {
        price,
        delta,
        gamma,
    }
}

/// Convenience guard: analytic pricing is only meaningful for European options.
pub fn is_analytic(exercise: Exercise) -> bool {
    matches!(exercise, Exercise::European)
}

/// Black–Scholes vega, `∂price/∂σ`.
pub fn vega(p: &Params) -> f64 {
    if p.t <= 0.0 || p.vol <= 0.0 {
        return 0.0;
    }
    let sqrt_t = p.t.sqrt();
    let d1 = ((p.spot / p.strike).ln() + (p.rate - p.dividend + 0.5 * p.vol * p.vol) * p.t)
        / (p.vol * sqrt_t);
    p.spot * (-p.dividend * p.t).exp() * norm_pdf(d1) * sqrt_t
}

/// Invert Black–Scholes for the implied volatility that reproduces `target`.
///
/// Newton's method seeded at 20% vol, falling back to bisection if vega
/// collapses. Returns `None` if the target is outside the no-arbitrage bounds.
pub fn implied_vol(p: &Params, target: f64) -> Option<f64> {
    let mut vol = 0.2;
    for _ in 0..100 {
        let trial = Params { vol, ..*p };
        let diff = price(&trial).price - target;
        if diff.abs() < 1e-8 {
            return Some(vol);
        }
        let v = vega(&trial);
        if v < 1e-10 {
            break;
        }
        vol -= diff / v;
        if !(1e-4..=5.0).contains(&vol) {
            break;
        }
    }
    // Bisection fallback on [1e-4, 5].
    let (mut lo, mut hi) = (1e-4, 5.0);
    let f = |s: f64| price(&Params { vol: s, ..*p }).price - target;
    if f(lo) * f(hi) > 0.0 {
        return None;
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(0.5 * (lo + hi))
}
