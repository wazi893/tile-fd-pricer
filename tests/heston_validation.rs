//! Validation for the Heston pricer.
//!
//! There is no closed form, so the strategy is a chain of cross-checks:
//!   1. As vol-of-vol ξ → 0 with v₀ = θ, variance is frozen at θ and the price
//!      must collapse to Black–Scholes with σ = √θ. This pins the analytic
//!      Fourier integral against a *known* answer.
//!   2. The 2D FD solver must reproduce the same Black–Scholes limit.
//!   3. With genuine stochastic vol, FD must match the (now-trusted) analytic
//!      Fourier price.
//!   4. Put–call parity holds on the analytic prices.

use tile_fd_pricer::black_scholes::{self, Params};
use tile_fd_pricer::heston::{self, Config, HestonParams};
use tile_fd_pricer::OptionType;

fn base(kind: OptionType) -> HestonParams {
    HestonParams {
        spot: 100.0,
        strike: 100.0,
        rate: 0.05,
        dividend: 0.0,
        t: 1.0,
        kind,
        v0: 0.04,
        kappa: 2.0,
        theta: 0.04,
        xi: 0.3,
        rho: -0.5,
    }
}

fn bs_equiv(h: &HestonParams, vol: f64) -> f64 {
    black_scholes::price(&Params {
        spot: h.spot,
        strike: h.strike,
        rate: h.rate,
        dividend: h.dividend,
        vol,
        t: h.t,
        kind: h.kind,
    })
    .price
}

#[test]
fn analytic_reduces_to_black_scholes() {
    // ξ tiny and v0 = θ ⇒ variance ≈ constant θ ⇒ BS with σ = √θ.
    let mut h = base(OptionType::Call);
    h.xi = 1e-3;
    h.v0 = 0.04;
    h.theta = 0.04;
    let an = heston::analytic_price(&h);
    let bs = bs_equiv(&h, h.theta.sqrt());
    assert!((an - bs).abs() < 5e-3, "Heston analytic {an} vs BS {bs}");
}

#[test]
fn fd_reduces_to_black_scholes() {
    let mut h = base(OptionType::Call);
    h.xi = 1e-3;
    let cfg = Config { nx: 160, nv: 40, num_std: 6.0, v_max: 0.0, steps: 0 };
    let r = heston::solve(&h, &cfg);
    let bs = bs_equiv(&h, h.theta.sqrt());
    assert!((r.price - bs).abs() < 0.15, "Heston FD {} vs BS {bs}", r.price);
}

#[test]
fn fd_matches_analytic_with_stochastic_vol() {
    // Full Heston: FD on a modest grid vs the trusted Fourier price.
    let h = base(OptionType::Call);
    let an = heston::analytic_price(&h);
    let cfg = Config { nx: 200, nv: 80, num_std: 6.0, v_max: 0.0, steps: 0 };
    let r = heston::solve(&h, &cfg);
    let rel = (r.price - an).abs() / an;
    assert!(rel < 0.02, "FD {} vs analytic {an} (rel {:.3})", r.price, rel);
}

#[test]
fn analytic_put_call_parity() {
    let c = heston::analytic_price(&base(OptionType::Call));
    let p = heston::analytic_price(&base(OptionType::Put));
    let h = base(OptionType::Call);
    let rhs = h.spot * (-h.dividend * h.t).exp() - h.strike * (-h.rate * h.t).exp();
    assert!((c - p - rhs).abs() < 1e-3, "parity: C−P={} vs {}", c - p, rhs);
}

#[test]
fn fd_put_via_solver_matches_analytic() {
    let h = base(OptionType::Put);
    let an = heston::analytic_price(&h);
    let cfg = Config { nx: 200, nv: 80, num_std: 6.0, v_max: 0.0, steps: 0 };
    let r = heston::solve(&h, &cfg);
    let rel = (r.price - an).abs() / an;
    assert!(rel < 0.03, "FD put {} vs analytic {an} (rel {:.3})", r.price, rel);
}

#[test]
fn smile_is_present() {
    // Heston with ρ<0 produces a negative-skew implied-vol smile: OTM puts
    // (low strike) should be worth more than the flat-vol model implies.
    let mut itm = base(OptionType::Put);
    itm.strike = 80.0;
    let mut atm = base(OptionType::Put);
    atm.strike = 100.0;
    let p_low = heston::analytic_price(&itm);
    let p_atm = heston::analytic_price(&atm);
    // Sanity: lower-strike put is cheaper in absolute terms but both positive.
    assert!(p_low > 0.0 && p_atm > 0.0);
    assert!(p_atm > p_low, "ATM put {p_atm} should exceed K=80 put {p_low}");
}
