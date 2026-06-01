//! Finite-difference option pricing on a log-space grid.
//!
//! The Black–Scholes PDE, written in `x = ln(S)` and `τ = T − t` (time to
//! maturity), has *constant* coefficients:
//!
//! ```text
//! ∂V/∂τ = a·∂²V/∂x²  +  b·∂V/∂x  −  r·V ,   a = ½σ²,  b = r − q − ½σ²
//! ```
//!
//! Discretising the spatial operator with central differences turns each time
//! step into a **three-point stencil**
//!
//! ```text
//! V'ᵢ = pd·Vᵢ₋₁ + pm·Vᵢ + pu·Vᵢ₊₁
//! ```
//!
//! i.e. every grid value is a function of its immediate neighbours — the exact
//! `output = f(left, center, right)` access pattern of the cellular-automaton
//! engine this crate's technique is borrowed from. The kernel
//! [`explicit_step`] is that line, isolated so it can be vectorised later
//! without touching the surrounding scheme logic.

use crate::black_scholes::Params;
use crate::{Exercise, OptionType};

/// A uniform grid in log-price. Node `n/2` is pinned exactly to `ln(spot)` so
/// the priced value and its Greeks are read off a node with no interpolation.
#[derive(Clone, Debug)]
pub struct Grid {
    /// Log-price coordinates `x = ln(S)`.
    pub x: Vec<f64>,
    /// Underlying prices `S = exp(x)`.
    pub s: Vec<f64>,
    /// Uniform spacing in `x`.
    pub dx: f64,
    /// Number of intervals (there are `n + 1` nodes).
    pub n: usize,
    /// Index of the node pinned to `ln(spot)`.
    pub spot_idx: usize,
}

impl Grid {
    /// Build a grid centred on `ln(spot)`, spanning `num_std` diffusion
    /// standard deviations either side. `n_space` is rounded up to even so the
    /// spot lands exactly on the centre node.
    pub fn new(p: &Params, n_space: usize, num_std: f64) -> Grid {
        let n = if n_space % 2 == 0 { n_space } else { n_space + 1 };
        let center = p.spot.ln();
        // Width driven by total diffusion plus drift over the option's life.
        let half_width = num_std * p.vol * p.t.sqrt() + (p.rate - p.dividend).abs() * p.t;
        let half_width = half_width.max(0.1); // never degenerate
        let dx = 2.0 * half_width / n as f64;

        let mut x = Vec::with_capacity(n + 1);
        let mut s = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let xi = center - half_width + i as f64 * dx;
            x.push(xi);
            s.push(xi.exp());
        }
        Grid { x, s, dx, n, spot_idx: n / 2 }
    }
}

/// Discretisation scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    /// Forward-Euler in time. Conditionally stable; this is the pure stencil.
    Explicit,
    /// Crank–Nicolson. Unconditionally stable, second-order in time.
    CrankNicolson,
}

/// Solver configuration.
#[derive(Clone, Copy, Debug)]
pub struct FdConfig {
    /// Number of space intervals (rounded up to even).
    pub n_space: usize,
    /// Requested number of time steps. For [`Scheme::Explicit`] this is raised
    /// if needed to satisfy the stability bound.
    pub n_time: usize,
    /// Half-width of the grid in diffusion standard deviations.
    pub num_std: f64,
    pub scheme: Scheme,
}

impl Default for FdConfig {
    fn default() -> Self {
        FdConfig { n_space: 400, n_time: 400, num_std: 8.0, scheme: Scheme::CrankNicolson }
    }
}

/// Outcome of a finite-difference solve.
#[derive(Clone, Debug)]
pub struct FdResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    /// Option value over the whole grid at `τ = T` (i.e. today).
    pub values: Vec<f64>,
    pub grid: Grid,
    /// Time steps actually taken (may exceed the request for stability).
    pub time_steps: usize,
}

/// The explicit stencil kernel: one forward-Euler sweep over the interior.
///
/// `dst[i] = pd·src[i-1] + pm·src[i] + pu·src[i+1]` for every interior node.
/// This is the single line that mirrors the cellular engine's neighbour
/// evaluation; everything else in this file is grid setup and boundary
/// handling around it.
#[inline]
pub fn explicit_step(src: &[f64], dst: &mut [f64], pd: f64, pm: f64, pu: f64) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    for i in 1..n - 1 {
        dst[i] = pd * src[i - 1] + pm * src[i] + pu * src[i + 1];
    }
}

/// Price an option by finite differences.
///
/// Deterministic and single-threaded: identical inputs yield bit-identical
/// outputs, which is the property the validation suite and future
/// cross-backend parity tests rely on.
pub fn solve(p: &Params, exercise: Exercise, cfg: &FdConfig) -> FdResult {
    solve_with(p, exercise, cfg, |_, _| {})
}

/// Like [`solve`], but invokes `record(tau, values)` once for the terminal
/// payoff (`τ = 0`) and after every completed time step. Used to capture the
/// value surface for visualisation without duplicating the scheme logic.
pub fn solve_with<F>(p: &Params, exercise: Exercise, cfg: &FdConfig, mut record: F) -> FdResult
where
    F: FnMut(f64, &[f64]),
{
    let grid = Grid::new(p, cfg.n_space, cfg.num_std);
    let n = grid.n;
    let dx = grid.dx;
    let a = 0.5 * p.vol * p.vol;
    let b = p.rate - p.dividend - 0.5 * p.vol * p.vol;

    // Terminal condition (τ = 0): the option's intrinsic payoff.
    let mut v: Vec<f64> = grid.s.iter().map(|&s| p.payoff(s)).collect();
    record(0.0, &v);

    let american = matches!(exercise, Exercise::American);

    let time_steps = match cfg.scheme {
        Scheme::Explicit => {
            // Stability: dτ·(2a/dx² + r) ≤ 1. Use a safety factor and bump the
            // step count until the bound holds.
            let dt_max = 1.0 / (2.0 * a / (dx * dx) + p.rate.max(0.0));
            let needed = (p.t / (0.9 * dt_max)).ceil() as usize;
            let steps = cfg.n_time.max(needed).max(1);
            let dtau = p.t / steps as f64;

            let alpha = a / (dx * dx);
            let beta = b / (2.0 * dx);
            let pu = dtau * (alpha + beta);
            let pm = 1.0 - dtau * (2.0 * alpha + p.rate);
            let pd = dtau * (alpha - beta);

            let mut next = v.clone();
            for step in 1..=steps {
                let tau = step as f64 * dtau;
                explicit_step(&v, &mut next, pd, pm, pu);
                apply_boundary(&mut next, &grid, p, tau, american);
                if american {
                    apply_early_exercise(&mut next, &grid, p);
                }
                std::mem::swap(&mut v, &mut next);
                record(tau, &v);
            }
            steps
        }
        Scheme::CrankNicolson => {
            let steps = cfg.n_time.max(1);
            let dtau = p.t / steps as f64;
            let alpha = a / (dx * dx);
            let beta = b / (2.0 * dx);

            // Constant tridiagonal coefficients of M = I − ½dτ·L.
            let m_sub = -0.5 * dtau * (alpha - beta);
            let m_diag = 1.0 + 0.5 * dtau * (2.0 * alpha + p.rate);
            let m_sup = -0.5 * dtau * (alpha + beta);
            // Coefficients of the explicit RHS operator (I + ½dτ·L).
            let r_sub = 0.5 * dtau * (alpha - beta);
            let r_diag = 1.0 - 0.5 * dtau * (2.0 * alpha + p.rate);
            let r_sup = 0.5 * dtau * (alpha + beta);

            let mut rhs = vec![0.0f64; n + 1];
            let mut scratch = vec![0.0f64; n + 1];
            for step in 1..=steps {
                let tau = step as f64 * dtau;
                // Boundary values at the new time level.
                let mut bc = v.clone();
                apply_boundary(&mut bc, &grid, p, tau, american);
                let (lo, hi) = (bc[0], bc[n]);

                // Build RHS for interior unknowns 1..n-1.
                for i in 1..n {
                    rhs[i] = r_sub * v[i - 1] + r_diag * v[i] + r_sup * v[i + 1];
                }
                rhs[1] -= m_sub * lo;
                rhs[n - 1] -= m_sup * hi;

                thomas(m_sub, m_diag, m_sup, &mut rhs, &mut scratch, 1, n - 1);

                v[0] = lo;
                v[n] = hi;
                for i in 1..n {
                    v[i] = rhs[i];
                }
                if american {
                    // Operator-splitting projection (a pragmatic American
                    // approximation; PSOR is a Phase-2 refinement).
                    apply_early_exercise(&mut v, &grid, p);
                }
                record(tau, &v);
            }
            steps
        }
    };

    let (price, delta, gamma) = read_value_and_greeks(&v, &grid);
    FdResult { price, delta, gamma, values: v, grid, time_steps }
}

/// Dirichlet boundary conditions, applied at time-to-maturity `tau`.
fn apply_boundary(v: &mut [f64], grid: &Grid, p: &Params, tau: f64, american: bool) {
    let n = grid.n;
    let disc_r = (-p.rate * tau).exp();
    let disc_q = (-p.dividend * tau).exp();
    match p.kind {
        OptionType::Call => {
            v[0] = 0.0; // S → 0
            v[n] = grid.s[n] * disc_q - p.strike * disc_r; // S → ∞: forward minus PV(K)
        }
        OptionType::Put => {
            v[n] = 0.0; // S → ∞
            v[0] = if american {
                p.payoff(grid.s[0]) // immediate exercise dominates as S → 0
            } else {
                (p.strike * disc_r - grid.s[0] * disc_q).max(0.0)
            };
        }
    }
}

/// American constraint: value can never drop below immediate exercise.
fn apply_early_exercise(v: &mut [f64], grid: &Grid, p: &Params) {
    for i in 0..=grid.n {
        let intrinsic = p.payoff(grid.s[i]);
        if v[i] < intrinsic {
            v[i] = intrinsic;
        }
    }
}

/// Thomas algorithm for a constant-coefficient tridiagonal system over the
/// inclusive index range `[lo, hi]`. The solution overwrites `rhs`.
fn thomas(sub: f64, diag: f64, sup: f64, rhs: &mut [f64], c: &mut [f64], lo: usize, hi: usize) {
    // Forward sweep.
    let mut beta = diag;
    rhs[lo] /= beta;
    c[lo] = sup / beta;
    for i in (lo + 1)..=hi {
        beta = diag - sub * c[i - 1];
        c[i] = sup / beta;
        rhs[i] = (rhs[i] - sub * rhs[i - 1]) / beta;
    }
    // Back substitution.
    for i in (lo..hi).rev() {
        rhs[i] -= c[i] * rhs[i + 1];
    }
}

/// Read the price at the spot node and compute Delta/Gamma by converting the
/// log-space derivatives back to price space.
fn read_value_and_greeks(v: &[f64], grid: &Grid) -> (f64, f64, f64) {
    let i = grid.spot_idx;
    let s = grid.s[i];
    let dx = grid.dx;
    let price = v[i];

    let dvdx = (v[i + 1] - v[i - 1]) / (2.0 * dx);
    let d2vdx2 = (v[i + 1] - 2.0 * v[i] + v[i - 1]) / (dx * dx);
    // S = e^x  ⇒  ∂V/∂S = (1/S)∂V/∂x,  ∂²V/∂S² = (∂²V/∂x² − ∂V/∂x)/S².
    let delta = dvdx / s;
    let gamma = (d2vdx2 - dvdx) / (s * s);
    (price, delta, gamma)
}
