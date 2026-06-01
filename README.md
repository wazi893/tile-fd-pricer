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

| Gamma surface `Γ(S,τ)` (1D) | Heston implied-vol skew |
|---|---|
| ![gamma surface](docs/gamma_surface.png) | ![iv surface](docs/iv_surface.png) |

*Left: Gamma read straight off the grid — a ridge at the strike that explodes
toward expiry. Right: the Heston model's negative implied-vol skew (high vol at
low strikes), which constant-vol Black–Scholes cannot produce.*

See [`WRITEUP.md`](WRITEUP.md) for the full story.

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

## Performance — batched pricing

A single option is microseconds of work; a desk prices a *book* of thousands.
So the stencil is vectorised **across options**: four contracts ride the four
lanes of an AVX2 `f64` register and march the same stencil in lockstep, then
the book is split across threads. The SIMD path is verified **bit-identical**
to the scalar reference (same fused-multiply-add order) before timing — the
speedups are honest because the answer never changes.

```
Book: 8192 options • grid 256 nodes × 488 steps • 1.0e9 stencil updates
                          time        throughput        stencil    speedup
scalar reference        1.971 s      4 157 opt/s      0.52 Gupd/s     1.00×
AVX2 (4×f64)            0.121 s     67 922 opt/s      8.45 Gupd/s    16.34×
AVX2 + 32 threads       0.012 s    708 663 opt/s     88.2 Gupd/s   170.46×
```

The 16× from a 4-wide register is the SIMD width compounded with the
structure-of-arrays layout (lane-contiguous, cache-friendly) and the
elimination of per-option heap allocation — i.e. *data layout earns as much as
SIMD here*. Reproduce with `cargo run --release --example bench [count]`.

## Heston stochastic volatility (2D)

Under Heston the variance is itself stochastic, so the option value `U(S, v, τ)`
solves a 2D PDE with a correlation cross term. Discretised in `(ln S, v)`, each
step is a **nine-point stencil** — five axis neighbours plus four diagonal
corners — the same `f(neighbours)` pattern, now in two dimensions.

Heston has no closed form, so `heston::analytic_price` implements the
characteristic-function (Fourier) integral as an oracle, and the FD solver is
validated against it (<2%) and against the Black–Scholes limit (ξ → 0). The
signature result is the **implied-volatility skew** that flat Black–Scholes
cannot produce:

```
cargo run --release --example heston_surface   # writes heston_surface.html
```

The visualisation shows the IV surface (strike × maturity), smile
cross-sections, and the FD value surface `U(S, v)`. *Note:* the 2D solver uses
an explicit scheme, so the time-step count is stability-bound by the vol-of-vol
diffusion; an ADI scheme (Douglas / Craig–Sneyd) would remove that limit and is
the natural next step.

## Status / roadmap

- **Phase 1 (done)** — 1D Black–Scholes grid, explicit + Crank–Nicolson schemes,
  European + American, Greeks, closed-form validation.
- **Phase 2 (done)** — SoA batched pricer: scalar → AVX2 → AVX2 + threads,
  bit-identical parity, throughput ladder.
- **Phase 3 (done)** — interactive HTML value-surface + Δ/Γ visualisation.
- **Phase 4 (done)** — 2D Heston stochastic vol (nine-point stencil), Fourier
  oracle, implied-vol surface visualisation.
- Next — ADI scheme for Heston; CUDA port of the stencil; American + PSOR.

## License

MIT OR Apache-2.0
