//! # tile-fd-pricer
//!
//! A deterministic finite-difference option pricer.
//!
//! The numerical core is a three-point **stencil** — every grid value is a
//! function of its immediate neighbours — which is the same
//! `output = f(neighbors)` access pattern as the cellular-automaton engine the
//! technique is borrowed from. Finite-difference methods price American
//! (early-exercise) options and yield Greeks directly off the grid, both of
//! which Monte Carlo handles poorly.
//!
//! European options are validated against the closed-form Black–Scholes price
//! in [`black_scholes`]; see the `validation` test suite for the convergence
//! analysis.
//!
//! ```
//! use tile_fd_pricer::*;
//! let p = black_scholes::Params {
//!     spot: 100.0, strike: 100.0, rate: 0.05, dividend: 0.0,
//!     vol: 0.2, t: 1.0, kind: OptionType::Call,
//! };
//! let fd = fd::solve(&p, Exercise::European, &fd::FdConfig::default());
//! let exact = black_scholes::price(&p);
//! assert!((fd.price - exact.price).abs() < 1e-2);
//! ```

pub mod black_scholes;
pub mod fd;

/// Call or put.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

/// Exercise style. American options have no closed form and must be priced on
/// the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exercise {
    European,
    American,
}
