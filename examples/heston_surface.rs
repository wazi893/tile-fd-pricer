//! Generate an interactive HTML visualisation of the Heston model:
//!   • the implied-volatility surface (strike × maturity) — the negative-skew
//!     "smile" that flat Black–Scholes cannot produce,
//!   • smile cross-sections at several maturities,
//!   • the 2D finite-difference value surface U(S, v) the solver computes.
//!
//! Run: `cargo run --release --example heston_surface`  →  heston_surface.html

use std::fmt::Write as _;
use tile_fd_pricer::black_scholes::{self, Params};
use tile_fd_pricer::heston::{self, Config, HestonParams};
use tile_fd_pricer::OptionType;

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

fn main() {
    // Parameters chosen for a pronounced, realistic negative skew.
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

    let maturities = linspace(0.1, 2.0, 24);
    let strikes = linspace(70.0, 135.0, 46);

    // Implied-vol surface: Heston price → invert Black–Scholes for IV.
    let mut iv = Vec::new();
    for &t in &maturities {
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
        iv.push(row);
    }

    // 2D FD value surface U(S, v) at T = 1.
    let hfd = HestonParams {
        t: 1.0,
        strike: 100.0,
        ..base
    };
    // Douglas ADI: unconditionally stable, ~200 steps instead of ~20k.
    let cfg = Config {
        nx: 160,
        nv: 80,
        num_std: 6.0,
        v_max: 0.0,
        steps: 200,
    };
    let r = heston::solve_adi(&hfd, &cfg);
    // Clip the price axis to a readable window.
    let (s_lo, s_hi) = (50.0, 170.0);
    let xi_idx: Vec<usize> = (0..=r.nx)
        .filter(|&i| r.s[i] >= s_lo && r.s[i] <= s_hi)
        .collect();
    let s_disp: Vec<f64> = xi_idx.iter().map(|&i| r.s[i]).collect();
    let stride = r.nv + 1;
    let mut surf = Vec::new(); // surf[vrow][snode]
    for j in 0..=r.nv {
        let row: Vec<f64> = xi_idx.iter().map(|&i| r.values[i * stride + j]).collect();
        surf.push(row);
    }

    let data = build_json(
        &base,
        &maturities,
        &strikes,
        &iv,
        &s_disp,
        &r.v,
        &surf,
        r.time_steps,
    );
    let html = TEMPLATE.replace("__DATA__", &data);
    std::fs::write("heston_surface.html", html).expect("write");
    println!(
        "Wrote heston_surface.html  (IV surface {}×{}, FD grid {}×{}, {} steps)",
        maturities.len(),
        strikes.len(),
        s_disp.len(),
        r.v.len(),
        r.time_steps
    );
}

#[allow(clippy::too_many_arguments)]
fn build_json(
    p: &HestonParams,
    mats: &[f64],
    strikes: &[f64],
    iv: &[Vec<f64>],
    s: &[f64],
    v: &[f64],
    surf: &[Vec<f64>],
    steps: usize,
) -> String {
    let mut o = String::new();
    write!(
        o,
        "{{\"spot\":{},\"r\":{},\"v0\":{},\"kappa\":{},\"theta\":{},\"xi\":{},\"rho\":{},\"steps\":{},",
        p.spot, p.rate, p.v0, p.kappa, p.theta, p.xi, p.rho, steps
    )
    .unwrap();
    arr(&mut o, "mats", mats, 4);
    o.push(',');
    arr(&mut o, "strikes", strikes, 3);
    o.push(',');
    arr(&mut o, "s", s, 3);
    o.push(',');
    arr(&mut o, "v", v, 5);
    o.push_str(",\"iv\":[");
    grid(&mut o, iv, 5);
    o.push_str("],\"surf\":[");
    grid(&mut o, surf, 4);
    o.push_str("]}");
    o
}

fn arr(o: &mut String, name: &str, vals: &[f64], dp: usize) {
    write!(o, "\"{name}\":[").unwrap();
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        fmt_num(o, *v, dp);
    }
    o.push(']');
}

fn grid(o: &mut String, rows: &[Vec<f64>], dp: usize) {
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push('[');
        for (k, v) in row.iter().enumerate() {
            if k > 0 {
                o.push(',');
            }
            fmt_num(o, *v, dp);
        }
        o.push(']');
    }
}

fn fmt_num(o: &mut String, v: f64, dp: usize) {
    if v.is_finite() {
        write!(o, "{:.*}", dp, v).unwrap();
    } else {
        o.push_str("null");
    }
}

const TEMPLATE: &str = r##"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Heston Stochastic-Volatility Surface</title>
<style>
  :root{color-scheme:dark}
  body{margin:0;background:#0d1117;color:#e6edf3;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
  header{padding:18px 24px 4px}h1{font-size:18px;margin:0 0 4px;font-weight:600}
  .sub{color:#8b949e;font-size:13px}
  .wrap{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:14px 24px 28px}
  .card{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:12px 14px}
  .card.full{grid-column:1/-1}
  h2{font-size:13px;margin:0 0 8px;color:#8b949e;font-weight:600;letter-spacing:.3px;text-transform:uppercase}
  canvas{width:100%;display:block;border-radius:6px}
  .legend{display:flex;gap:16px;font-size:12px;color:#8b949e;margin-top:6px;flex-wrap:wrap;align-items:center}
  .sw{display:inline-block;width:20px;height:3px;vertical-align:middle;margin-right:5px}
</style></head>
<body>
<header><h1>Heston Stochastic-Volatility Model — Finite-Difference Pricer</h1>
<div class="sub" id="meta"></div></header>
<div class="wrap">
  <div class="card full">
    <h2>Implied-volatility surface &nbsp;—&nbsp; strike × maturity (the skew flat Black–Scholes can't make)</h2>
    <canvas id="ivsurf" width="1100" height="280"></canvas>
    <div class="legend"><span>color = implied vol</span><span>dashed = strike 100 (ATM)</span><span>↑ longer maturity</span></div>
  </div>
  <div class="card">
    <h2>Volatility smile by maturity</h2>
    <canvas id="smile" width="560" height="320"></canvas>
    <div class="legend" id="smileleg"></div>
  </div>
  <div class="card">
    <h2>2D FD value surface &nbsp;U(S, v)&nbsp; — the nine-point stencil's output</h2>
    <canvas id="valsurf" width="560" height="320"></canvas>
    <div class="legend"><span>x = spot price</span><span>y = variance v</span><span>color = option value</span></div>
  </div>
</div>
<script>
const D=__DATA__;
document.getElementById('meta').textContent =
 `S=${D.spot}  r=${D.r}  v0=${D.v0} (vol ${Math.sqrt(D.v0).toFixed(2)})  κ=${D.kappa}  θ=${D.theta}  ξ=${D.xi}  ρ=${D.rho}  •  FD ${D.steps} steps, nine-point stencil`;

function cmap(t){t=Math.max(0,Math.min(1,t));
 const r=Math.round(255*Math.min(1,Math.max(0,1.5*t-0.3)));
 const g=Math.round(255*Math.min(1,Math.max(0,t<.5?0.2+1.4*t:0.9-0.1*(t-.5))));
 const b=Math.round(255*Math.min(1,Math.max(0,0.95-1.2*t)));
 return `rgb(${r},${g},${b})`;}

// ---- IV surface heatmap ----
(function(){
 const cv=document.getElementById('ivsurf'),x=cv.getContext('2d');
 const NM=D.mats.length,NK=D.strikes.length,W=cv.width,H=cv.height;
 let lo=1e9,hi=-1e9;
 for(const row of D.iv)for(const u of row)if(u!=null){if(u<lo)lo=u;if(u>hi)hi=u;}
 const cw=W/NK,ch=H/NM;
 for(let m=0;m<NM;m++)for(let k=0;k<NK;k++){
   const u=D.iv[m][k]; if(u==null)continue;
   x.fillStyle=cmap((u-lo)/(hi-lo));
   x.fillRect(k*cw,H-(m+1)*ch,Math.ceil(cw)+1,Math.ceil(ch)+1);
 }
 // ATM line
 const ki=D.strikes.findIndex(s=>s>=D.spot);
 x.strokeStyle='rgba(255,255,255,.5)';x.setLineDash([5,4]);
 x.beginPath();x.moveTo(ki*cw,0);x.lineTo(ki*cw,H);x.stroke();x.setLineDash([]);
 x.fillStyle='#8b949e';x.font='11px sans-serif';
 x.fillText(`IV ${(lo*100).toFixed(1)}%–${(hi*100).toFixed(1)}%`,8,16);
 x.fillText(`K ${D.strikes[0]}`,4,H-4);x.fillText(`K ${D.strikes[NK-1]|0}`,W-46,H-4);
})();

// ---- smile cross-sections ----
(function(){
 const cv=document.getElementById('smile'),x=cv.getContext('2d');
 const W=cv.width,H=cv.height,pad=40;
 const sel=[0,Math.floor(D.mats.length/3),Math.floor(2*D.mats.length/3),D.mats.length-1];
 const cols=['#2f81f7','#3fb950','#f0883e','#db61a2'];
 let lo=1e9,hi=-1e9;
 for(const m of sel)for(const u of D.iv[m])if(u!=null){if(u<lo)lo=u;if(u>hi)hi=u;}
 lo-=0.01;hi+=0.01;
 const SK0=D.strikes[0],SK1=D.strikes[D.strikes.length-1];
 x.clearRect(0,0,W,H);x.strokeStyle='#30363d';
 x.beginPath();x.moveTo(pad,8);x.lineTo(pad,H-pad);x.lineTo(W-8,H-pad);x.stroke();
 x.fillStyle='#8b949e';x.font='11px sans-serif';
 x.fillText((hi*100).toFixed(0)+'%',6,14);x.fillText((lo*100).toFixed(0)+'%',6,H-pad);
 x.fillText('K='+SK0,pad-6,H-pad+14);x.fillText('K='+(SK1|0),W-46,H-pad+14);
 sel.forEach((m,ci)=>{
   x.strokeStyle=cols[ci];x.lineWidth=2;x.beginPath();let st=false;
   for(let k=0;k<D.strikes.length;k++){const u=D.iv[m][k];if(u==null)continue;
     const px=pad+(D.strikes[k]-SK0)/(SK1-SK0)*(W-pad-8);
     const py=(H-pad)-(u-lo)/(hi-lo)*(H-pad-8);
     if(!st){x.moveTo(px,py);st=true}else x.lineTo(px,py);}
   x.stroke();
 });
 document.getElementById('smileleg').innerHTML=
   sel.map((m,ci)=>`<span><span class="sw" style="background:${cols[ci]}"></span>T=${D.mats[m].toFixed(2)}y</span>`).join('');
})();

// ---- FD value surface U(S,v) ----
(function(){
 const cv=document.getElementById('valsurf'),x=cv.getContext('2d');
 const NV=D.surf.length,NS=D.s.length,W=cv.width,H=cv.height;
 let hi=0;for(const row of D.surf)for(const u of row)if(u>hi)hi=u;
 const cw=W/NS,ch=H/NV;
 for(let j=0;j<NV;j++)for(let i=0;i<NS;i++){
   x.fillStyle=cmap(D.surf[j][i]/hi);
   x.fillRect(i*cw,H-(j+1)*ch,Math.ceil(cw)+1,Math.ceil(ch)+1);
 }
 // v0 line
 const vi=D.v.findIndex(v=>v>=D.v0);
 x.strokeStyle='rgba(255,255,255,.5)';x.setLineDash([5,4]);
 x.beginPath();x.moveTo(0,H-(vi+1)*ch);x.lineTo(W,H-(vi+1)*ch);x.stroke();x.setLineDash([]);
 x.fillStyle='#8b949e';x.font='11px sans-serif';
 x.fillText('v=v0',4,H-(vi+1)*ch-4);
 x.fillText('S='+(D.s[0]|0),4,H-4);x.fillText('S='+(D.s[NS-1]|0),W-44,H-4);
})();
</script></body></html>"##;
