//! Correctness suite. European options are checked against the closed-form
//! Black–Scholes price; the headline test is the *convergence order* — the
//! discretization error must fall roughly quadratically as the grid refines,
//! which is the honest proof that the scheme is implemented correctly.

use tile_fd_pricer::black_scholes::{self, Params};
use tile_fd_pricer::fd::{self, FdConfig, Scheme};
use tile_fd_pricer::{Exercise, OptionType};

fn atm_call() -> Params {
    Params {
        spot: 100.0,
        strike: 100.0,
        rate: 0.05,
        dividend: 0.0,
        vol: 0.2,
        t: 1.0,
        kind: OptionType::Call,
    }
}

fn atm_put() -> Params {
    Params {
        kind: OptionType::Put,
        ..atm_call()
    }
}

#[test]
fn analytic_greeks_match_finite_difference_bumps() {
    // The closed-form Greeks must equal central-difference bumps of the price.
    // This validates vega/theta/rho without any hardcoded reference values.
    for p in [atm_call(), atm_put()] {
        let g = black_scholes::price(&p);
        let bump = |f: &dyn Fn(f64) -> Params, h: f64| {
            (black_scholes::price(&f(h)).price - black_scholes::price(&f(-h)).price) / (2.0 * h)
        };
        let vega_fd = bump(
            &|h| Params {
                vol: p.vol + h,
                ..p
            },
            1e-4,
        );
        let rho_fd = bump(
            &|h| Params {
                rate: p.rate + h,
                ..p
            },
            1e-5,
        );
        // theta is ∂/∂(calendar time) = −∂/∂(time-to-maturity).
        let theta_fd = -bump(&|h| Params { t: p.t + h, ..p }, 1e-5);

        assert!(
            (g.vega - vega_fd).abs() < 1e-2,
            "{:?} vega {} vs {}",
            p.kind,
            g.vega,
            vega_fd
        );
        assert!(
            (g.rho - rho_fd).abs() < 1e-2,
            "{:?} rho {} vs {}",
            p.kind,
            g.rho,
            rho_fd
        );
        assert!(
            (g.theta - theta_fd).abs() < 1e-2,
            "{:?} theta {} vs {}",
            p.kind,
            g.theta,
            theta_fd
        );
    }
}

#[test]
fn down_and_out_barrier_sanity() {
    // Validated by limits and no-arbitrage relations (no hardcoded barrier
    // formula): a knock-out is worth less than the vanilla option, and as the
    // barrier moves far below spot it must converge to the vanilla price.
    let p = atm_call();
    let vanilla = black_scholes::price(&p).price;

    let near = fd::price_down_and_out(&p, 95.0, 400, 0); // barrier just below spot
    let far = fd::price_down_and_out(&p, 20.0, 400, 0); // barrier far away

    // A barrier just below spot knocks out easily ⇒ worth well under vanilla.
    assert!(
        near > 0.0 && near < 0.8 * vanilla,
        "near-barrier {near} vs vanilla {vanilla}"
    );
    // A far barrier almost never knocks out ⇒ converges to the vanilla price
    // (to within the explicit scheme's discretisation error).
    assert!(
        (far - vanilla).abs() < 0.05,
        "far barrier {far} vs vanilla {vanilla}"
    );
    // Monotonic: a lower barrier is less likely to knock out, so worth more.
    assert!(
        far > near,
        "lower barrier {far} should exceed higher barrier {near}"
    );
}

#[test]
fn european_call_matches_closed_form() {
    let p = atm_call();
    let cfg = FdConfig {
        n_space: 500,
        n_time: 500,
        num_std: 8.0,
        scheme: Scheme::CrankNicolson,
    };
    let fd = fd::solve(&p, Exercise::European, &cfg);
    let exact = black_scholes::price(&p);
    assert!(
        (fd.price - exact.price).abs() < 5e-3,
        "FD {} vs exact {}",
        fd.price,
        exact.price
    );
}

#[test]
fn european_put_matches_closed_form() {
    let p = atm_put();
    let cfg = FdConfig {
        n_space: 500,
        n_time: 500,
        num_std: 8.0,
        scheme: Scheme::CrankNicolson,
    };
    let fd = fd::solve(&p, Exercise::European, &cfg);
    let exact = black_scholes::price(&p);
    assert!(
        (fd.price - exact.price).abs() < 5e-3,
        "FD {} vs exact {}",
        fd.price,
        exact.price
    );
}

#[test]
fn greeks_match_closed_form() {
    let p = atm_call();
    let cfg = FdConfig {
        n_space: 800,
        n_time: 800,
        num_std: 8.0,
        scheme: Scheme::CrankNicolson,
    };
    let fd = fd::solve(&p, Exercise::European, &cfg);
    let exact = black_scholes::price(&p);
    assert!(
        (fd.delta - exact.delta).abs() < 5e-3,
        "delta {} vs {}",
        fd.delta,
        exact.delta
    );
    assert!(
        (fd.gamma - exact.gamma).abs() < 5e-4,
        "gamma {} vs {}",
        fd.gamma,
        exact.gamma
    );
}

#[test]
fn put_call_parity_on_grid() {
    // C − P = S·e^{−qT} − K·e^{−rT}, computed entirely from FD prices.
    let c = atm_call();
    let p = atm_put();
    let cfg = FdConfig {
        n_space: 500,
        n_time: 500,
        num_std: 8.0,
        scheme: Scheme::CrankNicolson,
    };
    let call = fd::solve(&c, Exercise::European, &cfg).price;
    let put = fd::solve(&p, Exercise::European, &cfg).price;
    let lhs = call - put;
    let rhs = c.spot * (-c.dividend * c.t).exp() - c.strike * (-c.rate * c.t).exp();
    assert!((lhs - rhs).abs() < 5e-3, "parity: {} vs {}", lhs, rhs);
}

#[test]
fn crank_nicolson_converges_quadratically() {
    let p = atm_call();
    let exact = black_scholes::price(&p).price;
    let resolutions = [64usize, 128, 256, 512];
    let mut errors = Vec::new();
    for &nr in &resolutions {
        let cfg = FdConfig {
            n_space: nr,
            n_time: nr,
            num_std: 8.0,
            scheme: Scheme::CrankNicolson,
        };
        let fd = fd::solve(&p, Exercise::European, &cfg);
        errors.push((fd.price - exact).abs());
    }
    // Each doubling of resolution must cut the error by at least ~2x; a clean
    // second-order scheme approaches 4x. This rejects a merely-stable-but-wrong
    // discretisation.
    for w in errors.windows(2) {
        let ratio = w[0] / w[1];
        assert!(ratio > 2.0, "convergence ratio {ratio} too low: {errors:?}");
    }
    assert!(
        *errors.last().unwrap() < 2e-3,
        "finest error too large: {errors:?}"
    );
}

#[test]
fn explicit_and_crank_nicolson_agree() {
    let p = atm_call();
    let ex = fd::solve(
        &p,
        Exercise::European,
        &FdConfig {
            n_space: 400,
            n_time: 400,
            num_std: 8.0,
            scheme: Scheme::Explicit,
        },
    );
    let cn = fd::solve(
        &p,
        Exercise::European,
        &FdConfig {
            n_space: 400,
            n_time: 400,
            num_std: 8.0,
            scheme: Scheme::CrankNicolson,
        },
    );
    assert!(
        (ex.price - cn.price).abs() < 5e-3,
        "explicit {} vs CN {}",
        ex.price,
        cn.price
    );
}

#[test]
fn american_put_has_early_exercise_premium() {
    let p = atm_put();
    let cfg = FdConfig {
        n_space: 500,
        n_time: 500,
        num_std: 8.0,
        scheme: Scheme::Explicit,
    };
    let euro = fd::solve(&p, Exercise::European, &cfg).price;
    let amer = fd::solve(&p, Exercise::American, &cfg).price;
    let intrinsic = p.payoff(p.spot);
    // An American put is worth at least its European twin and never less than
    // immediate exercise.
    assert!(amer >= euro - 1e-6, "american {amer} < european {euro}");
    assert!(
        amer >= intrinsic - 1e-6,
        "american {amer} < intrinsic {intrinsic}"
    );
    // With positive rates the premium should be strictly positive.
    assert!(
        amer > euro + 1e-3,
        "no early-exercise premium: {amer} vs {euro}"
    );
}

#[test]
fn american_call_no_dividends_equals_european() {
    // Classic result: it is never optimal to exercise an American call early
    // on a non-dividend-paying stock, so the two prices coincide.
    let p = atm_call();
    let cfg = FdConfig {
        n_space: 500,
        n_time: 500,
        num_std: 8.0,
        scheme: Scheme::Explicit,
    };
    let euro = fd::solve(&p, Exercise::European, &cfg).price;
    let amer = fd::solve(&p, Exercise::American, &cfg).price;
    assert!(
        (amer - euro).abs() < 5e-3,
        "american call {amer} vs european {euro}"
    );
}

#[test]
fn pricing_is_deterministic() {
    let p = atm_call();
    let cfg = FdConfig::default();
    let a = fd::solve(&p, Exercise::European, &cfg);
    let b = fd::solve(&p, Exercise::European, &cfg);
    // Bit-identical, not merely close — the property cross-backend parity rests on.
    assert_eq!(a.values, b.values);
    assert_eq!(a.price.to_bits(), b.price.to_bits());
}
