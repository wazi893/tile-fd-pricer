//! Heston stochastic-volatility model: a 2D finite-difference pricer plus a
//! semi-analytic (Fourier) oracle to validate it against.
//!
//! Under Heston the spot `S` and its instantaneous variance `v` evolve as
//! ```text
//! dS = (r−q)S dt + √v S dW₁
//! dv = κ(θ − v) dt + ξ√v dW₂ ,   dW₁·dW₂ = ρ dt
//! ```
//! The option value `U(S, v, τ)` satisfies a 2D convection–diffusion PDE with a
//! correlation **cross term** `ρξv·∂²U/∂S∂v`. Discretised in `x = ln S` and
//! `v`, each time step is a **nine-point stencil** — the five axis neighbours
//! plus four diagonal corners — i.e. exactly the `output = f(neighbours)`
//! pattern of the cellular engine, now in two dimensions.
//!
//! Heston has no closed form, so [`analytic_price`] implements the
//! characteristic-function integral (trap-free formulation) as the reference,
//! and the FD solver [`solve`] is checked against it and against the
//! Black–Scholes limit (vol-of-vol `ξ → 0`).

use crate::OptionType;
use std::f64::consts::PI;

/// Heston market + contract parameters.
#[derive(Clone, Copy, Debug)]
pub struct HestonParams {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub t: f64,
    pub kind: OptionType,
    /// Initial variance `v₀` (note: variance, not vol — `√v₀` is the spot vol).
    pub v0: f64,
    /// Mean-reversion speed `κ`.
    pub kappa: f64,
    /// Long-run variance `θ`.
    pub theta: f64,
    /// Vol-of-vol `ξ`.
    pub xi: f64,
    /// Spot/vol correlation `ρ`.
    pub rho: f64,
}

impl HestonParams {
    fn payoff(&self, s: f64) -> f64 {
        match self.kind {
            OptionType::Call => (s - self.strike).max(0.0),
            OptionType::Put => (self.strike - s).max(0.0),
        }
    }
}

// ----------------------------------------------------------------------------
// Minimal complex arithmetic (keeps the crate dependency-free).
// ----------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    fn new(re: f64, im: f64) -> C {
        C { re, im }
    }
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: C) -> C {
        C::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn scale(self, k: f64) -> C {
        C::new(self.re * k, self.im * k)
    }
    fn div(self, o: C) -> C {
        let d = o.re * o.re + o.im * o.im;
        C::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
    fn exp(self) -> C {
        let e = self.re.exp();
        C::new(e * self.im.cos(), e * self.im.sin())
    }
    fn ln(self) -> C {
        C::new(
            (self.re * self.re + self.im * self.im).sqrt().ln(),
            self.im.atan2(self.re),
        )
    }
    /// Principal square root via `exp(½·ln z)`.
    fn sqrt(self) -> C {
        self.ln().scale(0.5).exp()
    }
}

/// Heston probability `Pⱼ` integrand, `Re[ e^{−iφ lnK} fⱼ(φ) / (iφ) ]`.
///
/// Uses the "little trap"-free root choice (Albrecher et al. 2007), which keeps
/// the complex logarithm on its principal branch and so integrates cleanly.
fn integrand(p: &HestonParams, phi: f64, j: u8) -> f64 {
    let phi_i = C::new(0.0, phi); // iφ
    let (u, b) = if j == 1 {
        (0.5, p.kappa - p.rho * p.xi)
    } else {
        (-0.5, p.kappa)
    };
    let rho_xi_phi_i = C::new(0.0, p.rho * p.xi * phi); // ρξ·iφ
    let bc = C::new(b, 0.0);

    // d = sqrt((ρξφi − b)² − ξ²(2u·φi − φ²))
    let term1 = rho_xi_phi_i.sub(bc);
    let term1_sq = term1.mul(term1);
    let two_u_phi_i = C::new(0.0, 2.0 * u * phi);
    let inner = two_u_phi_i.sub(C::new(phi * phi, 0.0)).scale(p.xi * p.xi);
    let d = term1_sq.sub(inner).sqrt();

    // trap-free: g = (b − ρξφi − d)/(b − ρξφi + d)
    let bmr = bc.sub(rho_xi_phi_i);
    let g = bmr.sub(d).div(bmr.add(d));

    let edt = d.scale(-p.t).exp(); // e^{−dτ}
    let a = p.kappa * p.theta;

    // C(φ,τ) = (r−q)φi τ + (a/ξ²)[ (b−ρξφi−d)τ − 2 ln((1−g e^{−dτ})/(1−g)) ]
    let one = C::new(1.0, 0.0);
    let log_arg = one.sub(g.mul(edt)).div(one.sub(g));
    let cc = phi_i.scale((p.rate - p.dividend) * p.t).add(
        bmr.sub(d)
            .scale(p.t)
            .sub(log_arg.ln().scale(2.0))
            .scale(a / (p.xi * p.xi)),
    );

    // D(φ,τ) = (b−ρξφi−d)/ξ² · (1 − e^{−dτ})/(1 − g e^{−dτ})
    let dd = bmr
        .sub(d)
        .scale(1.0 / (p.xi * p.xi))
        .mul(one.sub(edt).div(one.sub(g.mul(edt))));

    // fⱼ = exp(C + D·v₀ + iφ·lnS₀)
    let f = cc.add(dd.scale(p.v0)).add(phi_i.scale(p.spot.ln())).exp();

    // Re[ e^{−iφ lnK} f / (iφ) ]
    let num = C::new(0.0, -phi * p.strike.ln()).exp().mul(f);
    num.div(phi_i).re
}

/// Semi-analytic Heston price via Simpson integration of the Fourier integrals.
/// This is the reference the FD grid is validated against.
pub fn analytic_price(p: &HestonParams) -> f64 {
    // Simpson over [φ₀, φ_max]; the integrand decays exponentially in φ.
    let (phi0, phi_max, n) = (1e-6, 200.0, 4000usize);
    let h = (phi_max - phi0) / n as f64;
    let prob = |j: u8| -> f64 {
        let mut s = integrand(p, phi0, j) + integrand(p, phi_max, j);
        for k in 1..n {
            let phi = phi0 + k as f64 * h;
            s += integrand(p, phi, j) * if k % 2 == 1 { 4.0 } else { 2.0 };
        }
        0.5 + s * h / 3.0 / PI
    };
    let p1 = prob(1);
    let p2 = prob(2);

    let call = p.spot * (-p.dividend * p.t).exp() * p1 - p.strike * (-p.rate * p.t).exp() * p2;
    match p.kind {
        OptionType::Call => call,
        // put–call parity
        OptionType::Put => {
            call - p.spot * (-p.dividend * p.t).exp() + p.strike * (-p.rate * p.t).exp()
        }
    }
}

// ----------------------------------------------------------------------------
// 2D finite-difference solver (explicit, nine-point stencil in (x = ln S, v)).
// ----------------------------------------------------------------------------

/// Solver configuration for the 2D grid.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Space intervals in `x = ln S` (rounded to even; spot pinned to centre).
    pub nx: usize,
    /// Variance intervals in `v` (`0 … v_max`).
    pub nv: usize,
    /// Half-width of the `x` grid in vol standard deviations.
    pub num_std: f64,
    /// Top of the variance grid; 0 auto-selects a multiple of `max(v₀, θ)`.
    pub v_max: f64,
    /// Time steps; 0 auto-selects the explicit-stability minimum.
    pub steps: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            nx: 120,
            nv: 60,
            num_std: 5.0,
            v_max: 0.0,
            steps: 0,
        }
    }
}

/// Result of a 2D solve.
pub struct Result {
    pub price: f64,
    /// Value surface `U[i*(nv+1) + j]` over `(x, v)` today (`τ = T`).
    pub values: Vec<f64>,
    pub x: Vec<f64>,
    pub s: Vec<f64>,
    pub v: Vec<f64>,
    pub nx: usize,
    pub nv: usize,
    pub time_steps: usize,
}

/// Price a Heston option by 2D explicit finite differences.
pub fn solve(p: &HestonParams, cfg: &Config) -> Result {
    let nx = if cfg.nx.is_multiple_of(2) {
        cfg.nx
    } else {
        cfg.nx + 1
    };
    let nv = cfg.nv;
    let v_max = if cfg.v_max > 0.0 {
        cfg.v_max
    } else {
        (p.v0.max(p.theta) * 6.0).max(0.2)
    };

    // x grid centred on ln(spot); spot sits on node nx/2.
    let center = p.spot.ln();
    let vref = p.v0.max(p.theta).max(1e-4);
    let half =
        (cfg.num_std * vref.sqrt() * p.t.sqrt()).max(0.2) + (p.rate - p.dividend).abs() * p.t;
    let dx = 2.0 * half / nx as f64;
    let x: Vec<f64> = (0..=nx).map(|i| center - half + i as f64 * dx).collect();
    let s: Vec<f64> = x.iter().map(|&xi| xi.exp()).collect();
    let dv = v_max / nv as f64;
    let v: Vec<f64> = (0..=nv).map(|j| j as f64 * dv).collect();
    let stride = nv + 1;

    // Explicit stability bound over the grid (dominated by max variance).
    let steps = if cfg.steps > 0 {
        cfg.steps
    } else {
        let diff = v_max / (dx * dx)
            + p.xi * p.xi * v_max / (dv * dv)
            + (p.rho * p.xi * v_max).abs() / (dx * dv)
            + p.kappa
            + p.rate.max(0.0);
        ((p.t * diff / 0.4).ceil() as usize).max(1)
    };
    let dtau = p.t / steps as f64;

    // Terminal payoff (independent of v).
    let mut u = vec![0.0f64; (nx + 1) * stride];
    for i in 0..=nx {
        let pay = p.payoff(s[i]);
        for j in 0..=nv {
            u[i * stride + j] = pay;
        }
    }
    let mut un = u.clone();

    for step in 1..=steps {
        let tau = step as f64 * dtau;
        for i in 1..nx {
            for j in 1..nv {
                let vj = v[j];
                let c = u[i * stride + j];
                let xp = u[(i + 1) * stride + j];
                let xm = u[(i - 1) * stride + j];
                let vp = u[i * stride + j + 1];
                let vm = u[i * stride + j - 1];
                // diagonal corners for the cross derivative
                let pp = u[(i + 1) * stride + j + 1];
                let pm = u[(i + 1) * stride + j - 1];
                let mp = u[(i - 1) * stride + j + 1];
                let mm = u[(i - 1) * stride + j - 1];

                let uxx = (xp - 2.0 * c + xm) / (dx * dx);
                let uvv = (vp - 2.0 * c + vm) / (dv * dv);
                let uxv = (pp - pm - mp + mm) / (4.0 * dx * dv);
                let ux = (xp - xm) / (2.0 * dx);
                let uv = (vp - vm) / (2.0 * dv);

                let lu = 0.5 * vj * uxx
                    + p.rho * p.xi * vj * uxv
                    + 0.5 * p.xi * p.xi * vj * uvv
                    + (p.rate - p.dividend - 0.5 * vj) * ux
                    + p.kappa * (p.theta - vj) * uv
                    - p.rate * c;
                un[i * stride + j] = c + dtau * lu;
            }

            // v = 0 boundary: diffusion vanishes, degenerate convection PDE.
            let c0 = u[i * stride];
            let ux0 = (u[(i + 1) * stride] - u[(i - 1) * stride]) / (2.0 * dx);
            let uv0 = (u[i * stride + 1] - c0) / dv; // one-sided (characteristics enter)
            let lu0 = (p.rate - p.dividend) * ux0 + p.kappa * p.theta * uv0 - p.rate * c0;
            un[i * stride] = c0 + dtau * lu0;
        }

        // x boundaries (Dirichlet), all variance levels.
        let disc_r = (-p.rate * tau).exp();
        let disc_q = (-p.dividend * tau).exp();
        for j in 0..=nv {
            let (lo, hi) = match p.kind {
                OptionType::Call => (0.0, s[nx] * disc_q - p.strike * disc_r),
                OptionType::Put => ((p.strike * disc_r - s[0] * disc_q).max(0.0), 0.0),
            };
            un[j] = lo;
            un[nx * stride + j] = hi;
        }
        // v = v_max boundary: ∂U/∂v = 0 (Neumann).
        for i in 0..=nx {
            un[i * stride + nv] = un[i * stride + nv - 1];
        }

        std::mem::swap(&mut u, &mut un);
    }

    // Price at (spot, v0): exact on the spot node, linear in v.
    let i0 = nx / 2;
    let jf = (p.v0 / dv).clamp(0.0, (nv - 1) as f64);
    let j0 = jf.floor() as usize;
    let frac = jf - j0 as f64;
    let price = u[i0 * stride + j0] * (1.0 - frac) + u[i0 * stride + j0 + 1] * frac;

    Result {
        price,
        values: u,
        x,
        s,
        v,
        nx,
        nv,
        time_steps: steps,
    }
}

/// Variable-coefficient tridiagonal solve over the inclusive range `[lo, hi]`
/// (Thomas algorithm). Solution overwrites `rhs`; `c` is scratch.
fn thomas_var(
    sub: &[f64],
    di: &[f64],
    sup: &[f64],
    rhs: &mut [f64],
    c: &mut [f64],
    lo: usize,
    hi: usize,
) {
    c[lo] = sup[lo] / di[lo];
    rhs[lo] /= di[lo];
    for k in (lo + 1)..=hi {
        let m = di[k] - sub[k] * c[k - 1];
        c[k] = sup[k] / m;
        rhs[k] = (rhs[k] - sub[k] * rhs[k - 1]) / m;
    }
    for k in (lo..hi).rev() {
        rhs[k] -= c[k] * rhs[k + 1];
    }
}

/// Price a Heston option with the **Douglas ADI** scheme (θ = ½).
///
/// Unlike the explicit [`solve`], this is unconditionally stable, so it needs
/// ~100 time steps where the explicit scheme needs tens of thousands. The
/// correlation cross term is handled explicitly in a predictor; the two axial
/// operators are then corrected implicitly via tridiagonal solves along `x` and
/// along `v`. Boundaries match [`solve`]: Dirichlet in `x`, a degenerate
/// convection update at `v = 0`, and Neumann at `v = v_max`.
pub fn solve_adi(p: &HestonParams, cfg: &Config) -> Result {
    let nx = if cfg.nx.is_multiple_of(2) {
        cfg.nx
    } else {
        cfg.nx + 1
    };
    let nv = cfg.nv;
    let v_max = if cfg.v_max > 0.0 {
        cfg.v_max
    } else {
        (p.v0.max(p.theta) * 6.0).max(0.2)
    };
    let center = p.spot.ln();
    let vref = p.v0.max(p.theta).max(1e-4);
    let half =
        (cfg.num_std * vref.sqrt() * p.t.sqrt()).max(0.2) + (p.rate - p.dividend).abs() * p.t;
    let dx = 2.0 * half / nx as f64;
    let x: Vec<f64> = (0..=nx).map(|i| center - half + i as f64 * dx).collect();
    let s: Vec<f64> = x.iter().map(|&xi| xi.exp()).collect();
    let dv = v_max / nv as f64;
    let v: Vec<f64> = (0..=nv).map(|j| j as f64 * dv).collect();
    let stride = nv + 1;

    let steps = if cfg.steps > 0 { cfg.steps } else { 100 };
    let dtau = p.t / steps as f64;
    let theta = 0.5;
    let (r, q, xi2) = (p.rate, p.dividend, p.xi * p.xi);
    let (dx2, dv2) = (dx * dx, dv * dv);

    // Per-row (v-dependent) operator coefficients; constant in i.
    let a1_lo = |j: usize| 0.5 * v[j] / dx2 - (r - q - 0.5 * v[j]) / (2.0 * dx);
    let a1_di = |j: usize| -v[j] / dx2 - 0.5 * r;
    let a1_up = |j: usize| 0.5 * v[j] / dx2 + (r - q - 0.5 * v[j]) / (2.0 * dx);
    let a2_lo = |j: usize| 0.5 * xi2 * v[j] / dv2 - p.kappa * (p.theta - v[j]) / (2.0 * dv);
    let a2_di = |j: usize| -xi2 * v[j] / dv2 - 0.5 * r;
    let a2_up = |j: usize| 0.5 * xi2 * v[j] / dv2 + p.kappa * (p.theta - v[j]) / (2.0 * dv);

    let mut u = vec![0.0f64; (nx + 1) * stride];
    for i in 0..=nx {
        let pay = p.payoff(s[i]);
        for j in 0..=nv {
            u[i * stride + j] = pay;
        }
    }
    let mut y = u.clone();
    let mut a1u = vec![0.0f64; (nx + 1) * stride];
    let mut a2u = vec![0.0f64; (nx + 1) * stride];
    let len = nx.max(nv) + 1;
    let (mut rhs, mut cwork) = (vec![0.0f64; len], vec![0.0f64; len]);
    let (mut sub, mut di, mut sup) = (vec![0.0f64; len], vec![0.0f64; len], vec![0.0f64; len]);

    for step in 1..=steps {
        let tau = step as f64 * dtau;
        let dr = (-r * tau).exp();
        let dq = (-q * tau).exp();
        let (xlo, xhi) = match p.kind {
            OptionType::Call => (0.0, s[nx] * dq - p.strike * dr),
            OptionType::Put => ((p.strike * dr - s[0] * dq).max(0.0), 0.0),
        };

        // Axial operators applied to the OLD field (interior only).
        for i in 1..nx {
            for j in 1..nv {
                a1u[i * stride + j] = a1_lo(j) * u[(i - 1) * stride + j]
                    + a1_di(j) * u[i * stride + j]
                    + a1_up(j) * u[(i + 1) * stride + j];
                a2u[i * stride + j] = a2_lo(j) * u[i * stride + j - 1]
                    + a2_di(j) * u[i * stride + j]
                    + a2_up(j) * u[i * stride + j + 1];
            }
        }

        // Predictor: Y0 = u + dτ·(A0 + A1 + A2)·u  (A0 = explicit cross term).
        for i in 1..nx {
            for j in 1..nv {
                let vj = v[j];
                let pp = u[(i + 1) * stride + j + 1];
                let pm = u[(i + 1) * stride + j - 1];
                let mp = u[(i - 1) * stride + j + 1];
                let mm = u[(i - 1) * stride + j - 1];
                let a0 = p.rho * p.xi * vj * (pp - pm - mp + mm) / (4.0 * dx * dv);
                let lu = a0 + a1u[i * stride + j] + a2u[i * stride + j];
                y[i * stride + j] = u[i * stride + j] + dtau * lu;
            }
            // v = 0 degenerate convection row.
            let c0 = u[i * stride];
            let ux0 = (u[(i + 1) * stride] - u[(i - 1) * stride]) / (2.0 * dx);
            let uv0 = (u[i * stride + 1] - c0) / dv;
            y[i * stride] = c0 + dtau * ((r - q) * ux0 + p.kappa * p.theta * uv0 - r * c0);
        }
        for j in 0..=nv {
            y[j] = xlo;
            y[nx * stride + j] = xhi;
        }
        for i in 0..=nx {
            y[i * stride + nv] = y[i * stride + nv - 1];
        }

        // Correction 1 (x-direction): (I − θdτ·A1) Y1 = Y0 − θdτ·A1·u, per v-row.
        for j in 1..nv {
            for i in 1..nx {
                rhs[i] = y[i * stride + j] - theta * dtau * a1u[i * stride + j];
            }
            let (sb, dg, sp) = (
                -theta * dtau * a1_lo(j),
                1.0 - theta * dtau * a1_di(j),
                -theta * dtau * a1_up(j),
            );
            for i in 1..nx {
                sub[i] = sb;
                di[i] = dg;
                sup[i] = sp;
            }
            rhs[1] -= sb * xlo;
            rhs[nx - 1] -= sp * xhi;
            thomas_var(&sub, &di, &sup, &mut rhs, &mut cwork, 1, nx - 1);
            for i in 1..nx {
                y[i * stride + j] = rhs[i];
            }
        }

        // Correction 2 (v-direction): (I − θdτ·A2) Y2 = Y1 − θdτ·A2·u, per x-column.
        // j = 0 is Dirichlet (predictor value); j = nv is Neumann (∂U/∂v = 0).
        for i in 1..nx {
            for j in 1..nv {
                rhs[j] = y[i * stride + j] - theta * dtau * a2u[i * stride + j];
                sub[j] = -theta * dtau * a2_lo(j);
                di[j] = 1.0 - theta * dtau * a2_di(j);
                sup[j] = -theta * dtau * a2_up(j);
            }
            // Dirichlet bottom: fold known v=0 value (predictor) into the j=1 row.
            rhs[1] -= sub[1] * y[i * stride];
            // Neumann top: U[nv] = U[nv-1] folds sup into the diagonal at j=nv-1.
            di[nv - 1] += sup[nv - 1];
            thomas_var(&sub, &di, &sup, &mut rhs, &mut cwork, 1, nv - 1);
            for j in 1..nv {
                y[i * stride + j] = rhs[j];
            }
            y[i * stride + nv] = y[i * stride + nv - 1];
        }

        std::mem::swap(&mut u, &mut y);
    }

    let i0 = nx / 2;
    let jf = (p.v0 / dv).clamp(0.0, (nv - 1) as f64);
    let j0 = jf.floor() as usize;
    let frac = jf - j0 as f64;
    let price = u[i0 * stride + j0] * (1.0 - frac) + u[i0 * stride + j0 + 1] * frac;

    Result {
        price,
        values: u,
        x,
        s,
        v,
        nx,
        nv,
        time_steps: steps,
    }
}
