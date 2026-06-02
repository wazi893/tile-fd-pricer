//! Render the two headline surfaces to static PNGs for the README / write-up
//! (GitHub does not render the interactive HTML). Dependency-free: includes a
//! minimal uncompressed-zlib PNG encoder.
//!
//! Run: `cargo run --release --example render_png`  →  docs/*.png

use tile_fd_pricer::black_scholes::{self, Params};
use tile_fd_pricer::fd::{self, FdConfig, Grid, Scheme};
use tile_fd_pricer::heston::{self, HestonParams};
use tile_fd_pricer::{Exercise, OptionType};

fn main() {
    std::fs::create_dir_all("docs").ok();
    render_gamma_surface();
    render_iv_surface();
    println!("Wrote docs/gamma_surface.png and docs/iv_surface.png");
}

// ---------------------------------------------------------------------------
// Surface 1: the Gamma surface Γ(S, τ) = ∂²V/∂S², read straight off the grid.
// A bright ridge at the strike that sharpens as expiry (τ → 0) approaches.
// ---------------------------------------------------------------------------
fn render_gamma_surface() {
    let p = Params {
        spot: 100.0,
        strike: 100.0,
        rate: 0.05,
        dividend: 0.0,
        vol: 0.20,
        t: 1.0,
        kind: OptionType::Put,
    };
    let cfg = FdConfig {
        n_space: 600,
        n_time: 600,
        num_std: 8.0,
        scheme: Scheme::CrankNicolson,
    };
    let grid = Grid::new(&p, cfg.n_space, cfg.num_std);
    let dx = grid.dx;
    // Interior window in price; needs one node of margin each side for Γ.
    let window: Vec<usize> = (1..grid.n)
        .filter(|&i| grid.s[i] >= 45.0 && grid.s[i] <= 175.0)
        .collect();

    let mut rows: Vec<Vec<f64>> = Vec::new();
    fd::solve_with(&p, Exercise::European, &cfg, |_tau, v| {
        let row: Vec<f64> = window
            .iter()
            .map(|&i| {
                let s = grid.s[i];
                let dvdx = (v[i + 1] - v[i - 1]) / (2.0 * dx);
                let d2 = (v[i + 1] - 2.0 * v[i] + v[i - 1]) / (dx * dx);
                ((d2 - dvdx) / (s * s)).max(0.0) // Γ in price space, clamped ≥0
            })
            .collect();
        rows.push(row);
    });
    let step = (rows.len() / 150).max(1);
    let rows: Vec<Vec<f64>> = rows.iter().step_by(step).cloned().collect();
    // Clamp the colour scale below the expiry singularity so the ridge is legible.
    let gmax = 0.06;
    let (w, h, px) = render_heatmap(&rows, gmax, 5);
    write_png("docs/gamma_surface.png", w, h, &px);
}

// ---------------------------------------------------------------------------
// Surface 2: Heston implied-volatility surface (strike × maturity).
// ---------------------------------------------------------------------------
fn render_iv_surface() {
    let base = HestonParams {
        spot: 100.0,
        strike: 100.0,
        rate: 0.03,
        dividend: 0.0,
        t: 1.0,
        kind: OptionType::Call,
        v0: 0.04,
        kappa: 1.5,
        theta: 0.045,
        xi: 0.5,
        rho: -0.7,
    };
    let mats: Vec<f64> = (0..24).map(|i| 0.1 + 1.9 * i as f64 / 23.0).collect();
    let strikes: Vec<f64> = (0..46).map(|i| 70.0 + 65.0 * i as f64 / 45.0).collect();

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for &t in &mats {
        let mut row = Vec::new();
        for &k in &strikes {
            let h = HestonParams {
                t,
                strike: k,
                ..base
            };
            let price = heston::analytic_price(&h);
            let bp = Params {
                spot: h.spot,
                strike: k,
                rate: h.rate,
                dividend: h.dividend,
                vol: 0.2,
                t,
                kind: h.kind,
            };
            row.push(black_scholes::implied_vol(&bp, price).unwrap_or(f64::NAN));
        }
        rows.push(row);
    }
    let (lo, hi) = rows
        .iter()
        .flatten()
        .filter(|x| x.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
    // Normalise into [0, vmax] so the shared colormap spans the IV range.
    let norm: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
            r.iter()
                .map(|&x| if x.is_finite() { x - lo } else { f64::NAN })
                .collect()
        })
        .collect();
    let (w, h, px) = render_heatmap(&norm, hi - lo, 18);
    write_png("docs/iv_surface.png", w, h, &px);
}

// ---------------------------------------------------------------------------
// Heatmap rendering (viridis-ish colormap, row 0 drawn at the bottom).
// ---------------------------------------------------------------------------
fn cmap(t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let r = (255.0 * (1.5 * t - 0.3).clamp(0.0, 1.0)) as u8;
    let g = (255.0
        * (if t < 0.5 {
            0.2 + 1.4 * t
        } else {
            0.9 - 0.1 * (t - 0.5)
        })
        .clamp(0.0, 1.0)) as u8;
    let b = (255.0 * (0.95 - 1.2 * t).clamp(0.0, 1.0)) as u8;
    [r, g, b]
}

/// Returns `(width, height, rgb)` for a `rows × cols` field, upscaled so each
/// cell is `block` pixels. `row[0]` is rendered at the bottom of the image.
fn render_heatmap(rows: &[Vec<f64>], vmax: f64, block: usize) -> (usize, usize, Vec<u8>) {
    let nr = rows.len();
    let nc = rows[0].len();
    let w = nc * block;
    let h = nr * block;
    let mut px = vec![0u8; w * h * 3];
    for (r, row) in rows.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            let color = if val.is_finite() {
                cmap(val / vmax)
            } else {
                [30, 33, 38]
            };
            // Flip vertically: data row 0 at image bottom.
            let y0 = (nr - 1 - r) * block;
            let x0 = c * block;
            for dy in 0..block {
                for dx in 0..block {
                    let idx = ((y0 + dy) * w + (x0 + dx)) * 3;
                    px[idx..idx + 3].copy_from_slice(&color);
                }
            }
        }
    }
    (w, h, px)
}

// ---------------------------------------------------------------------------
// Minimal PNG encoder: 8-bit RGB, single IDAT, uncompressed (stored) zlib.
// ---------------------------------------------------------------------------
fn write_png(path: &str, w: usize, h: usize, rgb: &[u8]) {
    // Filtered raw stream: each scanline prefixed with filter byte 0.
    let mut raw = Vec::with_capacity(h * (1 + w * 3));
    for y in 0..h {
        raw.push(0);
        raw.extend_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
    }
    let zlib = zlib_store(&raw);

    let mut out = Vec::new();
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]); // PNG signature

    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // bit depth 8, colour type 2 (RGB)
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib);
    chunk(&mut out, b"IEND", &[]);

    std::fs::write(path, out).expect("write png");
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// zlib stream wrapping the data in uncompressed "stored" deflate blocks.
fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut z = vec![0x78, 0x01]; // zlib header (deflate, default window)
    let mut i = 0;
    while i < data.len() {
        let len = (data.len() - i).min(0xFFFF);
        let final_block = i + len >= data.len();
        z.push(if final_block { 1 } else { 0 }); // BFINAL, BTYPE=00 (stored)
        z.extend_from_slice(&(len as u16).to_le_bytes());
        z.extend_from_slice(&(!(len as u16)).to_le_bytes());
        z.extend_from_slice(&data[i..i + len]);
        i += len;
    }
    z.extend_from_slice(&adler32(data).to_be_bytes());
    z
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}
