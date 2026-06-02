# A cellular automaton and an options pricer are the same loop

*How I reused the architecture of a 10¹⁴-cell/second cellular-automaton engine to
build a deterministic finite-difference options pricer in Rust.*

A cellular automaton updates a grid by one rule: `cell = f(its neighbours)`. A
finite-difference solver for the Black–Scholes PDE updates a grid by the same
rule: `value = f(its neighbours)`. Same memory access pattern, same sweep order,
same appetite for SIMD and cache-blocking — only the arithmetic in the middle
changes. So I took a stencil engine I'd already built (~115 trillion
cell-updates/second on a GPU) and pointed its architecture at options pricing,
where finite differences beat Monte Carlo on the cases that actually matter:
American exercise, Greeks, and reproducibility.

- **1D Black–Scholes** — explicit + Crank–Nicolson, European + American, Greeks
  straight off the grid, validated by **convergence order** against the closed form.
- **Batched throughput** — vectorised *across options* and threaded to ~940k
  options/second, every path **bit-identical** to a scalar reference. The
  benchmark also *corrected itself*: the headline "16× from SIMD" turned out to
  be a `mul_add`→libm-call artifact, and the real win was threading (see below).
- **2D Heston stochastic vol** — a nine-point stencil, validated against a
  semi-analytic Fourier oracle, producing the **implied-volatility skew** that
  flat Black–Scholes physically cannot.

**Try it:** `cargo test` runs the validation suite, `cargo run --example bench`
prints the throughput ladder, and `cargo run --example heston_surface` writes an
interactive HTML vol surface you can open in a browser. Dependency-free Rust, 20
tests, zero warnings. [github.com/wazi893/tile-fd-pricer](https://github.com/wazi893/tile-fd-pricer).

**The honest caveat, up front:** I am *not* running option prices through the
automaton's 1-bit boolean kernel — that would be absurd. What carries over is the
*architecture* (the memory layout, the sweep order, the determinism discipline),
not the bits. That distinction is the whole point, and I come back to it below.

---

## The same loop, concretely

The automaton packs 64 cells per `u64` and advances them with a single
instruction; on GPU it uses warp shuffles for horizontal neighbours and keeps
rows register-resident. The Black–Scholes solver wants the identical shape. Write
the PDE in `x = ln S` and `τ = T − t` and it becomes a constant-coefficient
diffusion equation. Discretise the spatial derivative with central differences
and one time step is:

```
V'ᵢ = pd·Vᵢ₋₁ + pm·Vᵢ + pu·Vᵢ₊₁
```

That's it. That's the stencil. Same `f(neighbours)` shape, different arithmetic
in the middle. The boolean kernel becomes a floating-point kernel; the
*choreography* — memory layout, sweep order, determinism — carries over wholesale.

As flagged above, the boolean kernel doesn't *become* the float kernel — but the
choreography around it does. And that's the real lesson: the hard-won patterns
(SoA layout, cache-friendly sweeps, cross-backend determinism) are
domain-general. The grid doesn't care whether the cells are alive-or-dead or
priced-in-dollars.

## Why finite differences (and not Monte Carlo)

Monte Carlo is the reflexive choice for a quant portfolio project, and for plain
European options it's fine. But it's the *wrong* tool for three things this
project leans on:

| | Monte Carlo | Finite differences |
|---|---|---|
| American / early exercise | needs Longstaff–Schwartz regression | one `max(V, payoff)` per step |
| Greeks (Δ, Γ) | bump-and-revalue, noisy | read directly off the grid |
| Barrier / knock-out | discrete-monitoring bias | an exact `V = 0` boundary |
| Reproducibility | seed-dependent | **bit-identical** |

The American case is the cleanest demonstration: with finite differences,
early exercise is a single line — clamp the value to the intrinsic payoff after
each time step. The free boundary (the spot price below which you should exercise
now) falls out for free, and you can watch it move in the interactive
visualisation (`cargo run --example surface`).

And because the entire value grid is already in hand, the Greeks are just finite
differences *on that grid* — no extra simulation, no bump-and-revalue noise.
Here is the Gamma surface `Γ(S, τ) = ∂²V/∂S²`: a bright ridge pinned to the
strike that sharpens into a singularity as expiry (bottom) approaches.

![gamma surface](docs/gamma_surface.png)

## Correctness: test the convergence *rate*, not a magic number

European options have a closed-form price, so the obvious test is "FD matches
Black–Scholes to within 1e-_something_." But that's a weak test — it passes if you
hand-tune the grid, even if the scheme is subtly wrong.

The honest test is **convergence order**. A correct second-order scheme's error
must fall *quadratically* as the grid refines — halve the spacing, quarter the
error. So I refine the grid through `64 → 128 → 256 → 512` and assert the error
ratio exceeds 2× each step (a true second-order scheme approaches 4×):

```rust
for w in errors.windows(2) {
    let ratio = w[0] / w[1];
    assert!(ratio > 2.0, "convergence ratio {ratio} too low");
}
```

This rejects a merely-stable-but-wrong discretisation in a way a fixed tolerance
never would. The suite also checks put–call parity on the grid, the no-arbitrage
result that an American call on a non-dividend stock equals its European twin, and
bit-for-bit determinism across runs.

## Performance: vectorise across options, not across the grid

A single option is microseconds of work — too small to optimise meaningfully. The
real problem a pricing desk has is a **book** of thousands. So the natural
vectorisation is *across options*: pack four contracts into the four lanes of an
AVX2 `f64` register and march them through the same stencil in lockstep. Each lane
carries its own strike, vol, and grid, so the coefficients become vectors, but the
kernel is still one line:

```rust
let mut acc = _mm256_mul_pd(pmv, cvec);          // pm * center
acc = _mm256_fmadd_pd(pdv, lvec, acc);           // + pd * left
acc = _mm256_fmadd_pd(puv, rvec, acc);           // + pu * right
_mm256_storeu_pd(dst.add(j * LANES), acc);       // four options, one store
```

Then split the book across threads with `std::thread::scope`. Every path
accumulates in the **same fused-multiply-add order** as a scalar reference, so the
results are **bit-identical** — verified before timing, so any speedup is honest:
same work, same answer.

My first benchmark was a trap I walked into, and walking back out is the part
worth telling. It showed ~16× from SIMD and ~170× with threads. Both numbers were
inflated, and the reason is a Rust footgun:

> `f64::mul_add` is only a hardware `vfmadd` if the target has FMA enabled at
> compile time. Otherwise it lowers to a **libm `fma()` call** — a function call
> per operation — because the language guarantees the fused (single-rounding)
> result. My scalar baselines were paying that call; the hand-written AVX2 kernel
> used the hardware instruction. So most of the "SIMD speedup" was really a
> missing `-C target-feature=+fma`.

After adding `target-cpu=native` so every path gets hardware FMA (best-of-3,
8,192 options, 32-core box):

```
                          time        throughput        vs prev
scalar (naive, AoS)     0.122 s     66 968 opt/s          1.00×
scalar (SoA layout)     0.216 s     37 915 opt/s          0.57×
AVX2 (4×f64, SoA)       0.106 s     77 594 opt/s          2.05×
AVX2 + 32 threads       0.009 s    937 246 opt/s         12.1×
```

The honest decomposition:

- The compiler **auto-vectorises the naive contiguous loop**. My hand-written
  AVX2 intrinsics are only ~1.16× faster than letting the optimiser do it.
- My clever structure-of-arrays-*across-options* layout actually made the scalar
  path **slower** (0.57×): the stride-4 access pattern defeats the unit-stride
  auto-vectoriser. The "obvious" optimisation was a pessimisation.
- The durable win is **threading — ~12× on 32 cores**, not the intrinsics.

This is less impressive than "170× from SIMD," and that's exactly why it's worth
writing down. The lessons are the senior ones: know what your "scalar" code
compiles to; the compiler is frequently as good as hand-written SIMD once it can
see the target; profile and attribute before you hand-vectorise. The bit-identity
check is what made the correction findable — when every path must produce the same
bits, you can't quietly explain away a suspicious number.

## Heston: stochastic vol, in two dimensions

Black–Scholes assumes constant volatility, which is false — the market prices a
**volatility smile**. The Heston model fixes this by making variance itself
stochastic. The option value `U(S, v, τ)` now solves a 2D PDE with a correlation
cross-term, and discretising in `(ln S, v)` gives a **nine-point stencil**: the
five axis neighbours plus four diagonal corners for the mixed derivative. Same
`f(neighbours)` pattern, one dimension up.

Heston has no closed-form price, so to validate the solver I implemented the
**semi-analytic characteristic-function (Fourier) price** as an oracle — complex
arithmetic, the trap-free root choice, Simpson integration — and checked the FD
grid against it (<2%) and against the Black–Scholes limit as vol-of-vol → 0. The
two methods cross-validate without relying on any memorised benchmark constant.

The first cut used an explicit scheme, which is stability-bound by the vol-of-vol
diffusion — it needed ~20,000 time steps. So I added a **Douglas ADI** scheme:
the correlation cross-term is handled explicitly in a predictor, then the two
axial operators are corrected implicitly with tridiagonal solves along each axis.
It's unconditionally stable and prices in **~200 steps** at the same accuracy —
validated against the Fourier oracle (<2%) and the explicit scheme (<1%) across a
spread of correlations, vol-of-vols, maturities, and moneyness. Two independent
schemes agreeing with an independent analytic oracle is about as much confidence
as you get without a market to check against.

The payoff is the signature result: a genuine **negative equity skew** that flat
Black–Scholes cannot produce. With correlation ρ = −0.7, implied vol runs from 31%
at low strikes through 20% at the money (= √v₀) down to 15% at high strikes:

![implied vol surface](docs/iv_surface.png)

## What I'd do next (honest limitations)

- The Douglas ADI scheme is first-order in time because the cross-term is
  explicit; a **Craig–Sneyd** second corrector would restore second-order
  accuracy.
- American options under Crank–Nicolson currently use an operator-splitting
  projection; **PSOR** would solve the linear-complementarity problem properly.
- The stencil is a natural **CUDA** target — the same warp-shuffle,
  register-resident playbook the cellular engine already uses.

## What this demonstrates

Numerical correctness (convergence analysis, not vibes), hardware-aware
performance (SIMD + memory layout + threading, measured and attributed), and a
real quant model end-to-end (Heston, Fourier pricing, implied-vol inversion) —
with a determinism discipline carried over from a much larger systems project.

Code: dependency-free Rust. `cargo test` for the validation suite,
`cargo run --example bench` for the throughput ladder,
`cargo run --example surface` / `heston_surface` for the visualisations.
