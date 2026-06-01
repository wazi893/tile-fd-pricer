# tile-fd-pricer

A **deterministic finite-difference option pricer** in Rust.

The numerical core is a three-point **stencil** — every grid value is computed
from its immediate neighbours:

```
V'ᵢ = pd·Vᵢ₋₁ + pm·Vᵢ + pu·Vᵢ₊₁
```

That is the same `output = f(left, center, right)` access pattern as the
cellular-automaton engine this technique is borrowed from (a register-resident
stencil kernel running at ~10¹⁴ cells/s on GPU). The PDE solver reuses the
*architecture* — neighbour-local updates, deterministic ordering, cache-friendly
sweeps — with a floating-point kernel in place of the boolean one.

## Why finite differences (and not Monte Carlo)

| | Monte Carlo | Finite differences |
|---|---|---|
| American / early exercise | needs Longstaff–Schwartz regression | one `max(V, payoff)` per step |
| Greeks (Δ, Γ) | bump-and-revalue, noisy | read directly off the grid |
| Convergence | stochastic, O(1/√N) | deterministic, O(Δx²) |
| Reproducibility | seed-dependent | **bit-identical** |

## Correctness

European options have a closed-form Black–Scholes price, used as the oracle.
The headline test is the **convergence order**: as the grid is refined the
Crank–Nicolson error falls quadratically, the signature of a correct
second-order scheme.

```
ATM put, S=K=100, r=5%, σ=20%, T=1y     price       delta       gamma
Black–Scholes (exact)                  5.573518   -0.363169    0.018762
FD European (800×800, Crank–Nicolson)  5.573132   -0.363169    0.018763   (err 3.9e-4)
FD American                            6.089204     — early-exercise premium 0.516
```

The suite also verifies put–call parity on the grid, that the American call on a
non-dividend stock equals its European twin (a known no-arbitrage result), and
that pricing is **bit-for-bit deterministic** across runs.

## Run it

```bash
cargo test --release             # validation suite (convergence, parity, determinism)
cargo run --release --example price
cargo run --release --example surface   # writes interactive option_surface.html
```

## Visualisation

`cargo run --release --example surface` emits a self-contained
`option_surface.html` (no dependencies, opens in any browser): the option value
surface `V(S, τ)` animating backward from expiry to today, the FD European and
American prices tracking the exact Black–Scholes curve, live Delta/Gamma, and a
toggle for the American early-exercise free boundary.

## Status / roadmap

- **Phase 1 (done)** — 1D Black–Scholes grid, explicit + Crank–Nicolson schemes,
  European + American, Greeks, closed-form validation.
- **Phase 3 (done)** — interactive HTML value-surface + Δ/Γ visualisation.
- Phase 2 — vectorise the stencil; scalar-vs-SIMD parity within ε; benchmark ladder.
- Phase 4 — 2D Heston stochastic vol (the genuine 5-point stencil) + optional CUDA.

## License

MIT OR Apache-2.0
