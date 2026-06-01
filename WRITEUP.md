# I built a CPU from logic gates. Then I turned its stencil kernel into an options pricer.

*A finite-difference option pricer in Rust — and why the kernel is the same one
that runs a cellular-automaton engine at 10¹⁴ cells/second.*

---

## TL;DR

I had a high-performance cellular-automaton engine whose hot loop is a **stencil**:
every cell's next value is a function of its immediate neighbours,
`out = f(left, right, up, down)`, evaluated at ~115 trillion cells/second on a
GPU. It turns out that a **finite-difference PDE solver** — the standard way to
price options when Monte Carlo is the wrong tool — has *the exact same access
pattern*. So I reused the architecture (neighbour-local updates, structure-of-
arrays layout, deterministic ordering) and built a deterministic options pricer:

- **1D Black–Scholes** — explicit + Crank–Nicolson, European + American, Greeks
  straight off the grid, validated by **convergence order** against the closed form.
- **Batched throughput** — vectorised *across options* (four contracts per AVX2
  register), then threaded: **4,200 → 709,000 options/second**, the SIMD path
  **bit-identical** to the scalar reference.
- **2D Heston stochastic vol** — a nine-point stencil, validated against a
  semi-analytic Fourier oracle, producing the **implied-volatility skew** that
  flat Black–Scholes physically cannot.

Dependency-free Rust, 20 tests, zero warnings. [Repo](.) · [code walkthrough below].

---

## The connection nobody mentions

A cellular automaton updates a grid: each cell reads its neighbours and computes
a new value. My engine packs 64 cells per `u64` and evaluates them with a single
instruction; on GPU it uses warp shuffles for horizontal neighbours and keeps
rows register-resident.

A finite-difference solver for the Black–Scholes PDE does the same thing. Write
the PDE in `x = ln S` and `τ = T − t` and it becomes a constant-coefficient
diffusion equation. Discretise the spatial derivative with central differences
and one time step is:

```
V'ᵢ = pd·Vᵢ₋₁ + pm·Vᵢ + pu·Vᵢ₊₁
```

That's it. That's the stencil. Same `f(neighbours)` shape, different arithmetic
in the middle. The boolean kernel becomes a floating-point kernel; the
*choreography* — memory layout, sweep order, determinism — carries over wholesale.

I want to be precise about the claim, because it's easy to overstate: I am **not**
running option prices through the 1-bit boolean kernel. What transfers is the
*architecture*, not the bits. That's still the whole point — the hard-won
optimisation patterns (SoA layout, cache-friendly sweeps, cross-backend
determinism) are domain-general, and that's exactly why a performance engineer is
worth hiring.

## Why finite differences (and not Monte Carlo)

Monte Carlo is the reflexive choice for a quant portfolio project, and for plain
European options it's fine. But it's the *wrong* tool for three things this
project leans on:

| | Monte Carlo | Finite differences |
|---|---|---|
| American / early exercise | needs Longstaff–Schwartz regression | one `max(V, payoff)` per step |
| Greeks (Δ, Γ) | bump-and-revalue, noisy | read directly off the grid |
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

Then split the book across threads with `std::thread::scope`. The ladder, pricing
8,192 options:

```
                          time        throughput        speedup
scalar reference        1.971 s      4 157 opt/s          1.00×
AVX2 (4×f64)            0.121 s     67 922 opt/s         16.34×
AVX2 + 32 threads       0.012 s    708 663 opt/s        170.46×
```

**The honest part:** 16× from a 4-wide register is *more* than the SIMD width, and
a good interviewer will ask why. The answer is that SIMD compounds with two other
wins — the structure-of-arrays layout (lane-contiguous, cache-friendly) and
eliminating a per-option heap allocation. Data layout earned as much as the vector
instructions did. And critically, the SIMD path accumulates in the **same
fused-multiply-add order** as the scalar reference, so it's **bit-identical** —
verified before timing, so the speedup is honest: same work, same answer, faster.

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

The payoff is the signature result: a genuine **negative equity skew** that flat
Black–Scholes cannot produce. With correlation ρ = −0.7, implied vol runs from 31%
at low strikes through 20% at the money (= √v₀) down to 15% at high strikes:

![implied vol surface](docs/iv_surface.png)

## What I'd do next (honest limitations)

- The 2D Heston solver uses an **explicit** scheme, so its time-step count is
  stability-bound by the vol-of-vol diffusion. An ADI scheme (Douglas /
  Craig–Sneyd) would make it unconditionally stable — the correct production fix.
- American options under Crank–Nicolson currently use an operator-splitting
  projection; PSOR would solve the linear-complementarity problem properly.
- The stencil is a natural CUDA target — the same warp-shuffle, register-resident
  playbook the cellular engine already uses.

## What this demonstrates

Numerical correctness (convergence analysis, not vibes), hardware-aware
performance (SIMD + memory layout + threading, measured and attributed), and a
real quant model end-to-end (Heston, Fourier pricing, implied-vol inversion) —
with a determinism discipline carried over from a much larger systems project.

Code: dependency-free Rust. `cargo test` for the validation suite,
`cargo run --example bench` for the throughput ladder,
`cargo run --example surface` / `heston_surface` for the visualisations.
