//! Generate a self-contained interactive HTML visualisation of the option
//! value surface as it evolves backward in time from expiry to today.
//!
//! Run: `cargo run --release --example surface`
//! Output: `option_surface.html` (open in any browser — no dependencies).

use std::fmt::Write as _;
use tile_fd_pricer::black_scholes::Params;
use tile_fd_pricer::fd::{self, FdConfig, Grid, Scheme};
use tile_fd_pricer::{Exercise, OptionType};

fn main() {
    let p = Params {
        spot: 100.0,
        strike: 100.0,
        rate: 0.05,
        dividend: 0.0,
        vol: 0.20,
        t: 1.0,
        kind: OptionType::Put,
    };
    let cfg = FdConfig { n_space: 600, n_time: 600, num_std: 8.0, scheme: Scheme::CrankNicolson };

    // Rebuild the same grid to recover node prices and pick a display window.
    let grid = Grid::new(&p, cfg.n_space, cfg.num_std);
    let (s_lo, s_hi) = (p.strike * 0.25, p.strike * 2.2);
    let mut idx: Vec<usize> =
        (0..=grid.n).filter(|&i| grid.s[i] >= s_lo && grid.s[i] <= s_hi).collect();
    // Subsample the window to keep the embedded data compact.
    let target_nodes = 180;
    if idx.len() > target_nodes {
        let stride = idx.len() / target_nodes;
        idx = idx.iter().step_by(stride.max(1)).copied().collect();
    }
    let window: Vec<usize> = idx;
    let s_disp: Vec<f64> = window.iter().map(|&i| grid.s[i]).collect();

    // Capture the full evolution for both exercise styles.
    let record_surface = |exercise: Exercise| -> Vec<(f64, Vec<f64>)> {
        let mut frames = Vec::new();
        fd::solve_with(&p, exercise, &cfg, |tau, v| {
            frames.push((tau, window.iter().map(|&i| v[i]).collect()));
        });
        frames
    };
    let euro = record_surface(Exercise::European);
    let amer = record_surface(Exercise::American);

    // Downsample the time axis to ~140 frames for a smooth but light animation.
    let target_frames = 140;
    let stride = (euro.len() / target_frames).max(1);
    let frame_ids: Vec<usize> = (0..euro.len()).step_by(stride).collect();

    // The grid is uniform in x = ln(S); pass dx so Greeks can be differenced in JS.
    let dx = grid.dx;

    let data = build_json(&p, &s_disp, dx, &euro, &amer, &frame_ids);
    let html = TEMPLATE.replace("__DATA__", &data);

    let path = "option_surface.html";
    std::fs::write(path, html).expect("write html");
    println!("Wrote {path} ({} frames, {} price nodes)", frame_ids.len(), s_disp.len());
}

fn build_json(
    p: &Params,
    s: &[f64],
    dx: f64,
    euro: &[(f64, Vec<f64>)],
    amer: &[(f64, Vec<f64>)],
    frame_ids: &[usize],
) -> String {
    let mut out = String::new();
    out.push('{');
    write!(
        out,
        "\"kind\":\"{}\",\"spot\":{},\"strike\":{},\"rate\":{},\"div\":{},\"vol\":{},\"T\":{},\"dx\":{:.8},",
        match p.kind { OptionType::Call => "Call", OptionType::Put => "Put" },
        p.spot, p.strike, p.rate, p.dividend, p.vol, p.t, dx
    )
    .unwrap();

    out.push_str("\"s\":[");
    for (k, v) in s.iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        write!(out, "{:.4}", v).unwrap();
    }
    out.push_str("],\"frames\":[");

    for (fi, &id) in frame_ids.iter().enumerate() {
        if fi > 0 {
            out.push(',');
        }
        let (tau, ref ve) = euro[id];
        let va = &amer[id].1;
        write!(out, "{{\"tau\":{:.5},\"ve\":[", tau).unwrap();
        push_arr(&mut out, ve);
        out.push_str("],\"va\":[");
        push_arr(&mut out, va);
        out.push_str("]}");
    }
    out.push_str("]}");
    out
}

fn push_arr(out: &mut String, vals: &[f64]) {
    for (k, v) in vals.iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        write!(out, "{:.5}", v).unwrap();
    }
}

const TEMPLATE: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Finite-Difference Option Value Surface</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0d1117; color:#e6edf3; font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:18px 24px 6px; }
  h1 { font-size:18px; margin:0 0 4px; font-weight:600; }
  .sub { color:#8b949e; font-size:13px; }
  .wrap { display:grid; grid-template-columns: 1.15fr 1fr; gap:18px; padding:12px 24px 28px; }
  .card { background:#161b22; border:1px solid #21262d; border-radius:10px; padding:12px 14px; }
  .card h2 { font-size:13px; margin:0 0 8px; color:#8b949e; font-weight:600; letter-spacing:.3px; text-transform:uppercase; }
  canvas { width:100%; display:block; border-radius:6px; }
  .full { grid-column:1 / -1; }
  .controls { display:flex; align-items:center; gap:14px; padding:6px 24px 4px; flex-wrap:wrap; }
  button { background:#238636; color:#fff; border:0; padding:7px 16px; border-radius:6px; cursor:pointer; font-weight:600; }
  button.alt { background:#30363d; }
  input[type=range] { flex:1; min-width:160px; accent-color:#2f81f7; }
  .legend { display:flex; gap:16px; font-size:12px; color:#8b949e; align-items:center; flex-wrap:wrap; }
  .swatch { display:inline-block; width:22px; height:3px; vertical-align:middle; margin-right:6px; }
  .readout { font-variant-numeric:tabular-nums; color:#e6edf3; }
  b.eu{color:#2f81f7}b.am{color:#f0883e}b.an{color:#8b949e}b.pay{color:#3fb950}
</style>
</head>
<body>
<header>
  <h1>Finite-Difference Option Value Surface</h1>
  <div class="sub" id="meta"></div>
</header>

<div class="controls">
  <button id="play">⏸ Pause</button>
  <button id="mode" class="alt">Show American boundary</button>
  <input type="range" id="scrub" min="0" max="100" value="0">
  <div class="readout" id="clock"></div>
</div>

<div class="wrap">
  <div class="card full">
    <h2>Value surface &nbsp; V(S, τ) &nbsp;—&nbsp; price × time-to-maturity</h2>
    <canvas id="heat" width="1100" height="300"></canvas>
  </div>

  <div class="card">
    <h2>Value vs spot at current τ</h2>
    <canvas id="curve" width="560" height="320"></canvas>
    <div class="legend">
      <span><span class="swatch" style="background:#2f81f7"></span>FD European</span>
      <span><span class="swatch" style="background:#f0883e"></span>FD American</span>
      <span><span class="swatch" style="background:#8b949e"></span>Black–Scholes (exact)</span>
      <span><span class="swatch" style="background:#3fb950"></span>Payoff</span>
    </div>
  </div>

  <div class="card">
    <h2>Greeks at current τ (European)</h2>
    <canvas id="greeks" width="560" height="320"></canvas>
    <div class="legend">
      <span><span class="swatch" style="background:#a371f7"></span>Delta = ∂V/∂S</span>
      <span><span class="swatch" style="background:#db61a2"></span>Gamma = ∂²V/∂S²</span>
    </div>
    <div class="readout" id="greekread" style="margin-top:6px"></div>
  </div>
</div>

<script>
const DATA = __DATA__;
const S = DATA.s, F = DATA.frames, dx = DATA.dx;
const N = S.length, NF = F.length;
const SQRT2PI = Math.sqrt(2*Math.PI);

document.getElementById('meta').textContent =
  `${DATA.kind}  •  S=${DATA.spot}  K=${DATA.strike}  r=${DATA.rate}  q=${DATA.div}  σ=${DATA.vol}  T=${DATA.T}  •  Crank–Nicolson, ${N} price nodes × ${NF} frames`;

// ---- Black–Scholes closed form (for the dashed overlay) ----
function ncdf(x){ if(x<0) return 1-ncdf(-x);
  const t=1/(1+0.2316419*x);
  const poly=t*(0.319381530+t*(-0.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));
  return 1 - Math.exp(-0.5*x*x)/SQRT2PI*poly;
}
function bs(s,tau){
  const {strike:k,rate:r,div:q,vol:sig,kind}=DATA;
  if(tau<=1e-9||sig<=0){ return kind==='Call'?Math.max(s-k,0):Math.max(k-s,0); }
  const sq=Math.sqrt(tau);
  const d1=(Math.log(s/k)+(r-q+0.5*sig*sig)*tau)/(sig*sq), d2=d1-sig*sq;
  return kind==='Call'
    ? s*Math.exp(-q*tau)*ncdf(d1)-k*Math.exp(-r*tau)*ncdf(d2)
    : k*Math.exp(-r*tau)*ncdf(-d2)-s*Math.exp(-q*tau)*ncdf(-d1);
}
function payoff(s){ return DATA.kind==='Call'?Math.max(s-DATA.strike,0):Math.max(DATA.strike-s,0); }

// ---- Greeks by central difference (uniform in x=ln S) ----
function greeks(v){ const d=[],g=[];
  for(let i=0;i<N;i++){
    if(i===0||i===N-1){ d.push(NaN); g.push(NaN); continue; }
    const s=S[i];
    const dvdx=(v[i+1]-v[i-1])/(2*dx);
    const d2=(v[i+1]-2*v[i]+v[i-1])/(dx*dx);
    d.push(dvdx/s); g.push((d2-dvdx)/(s*s));
  } return {d,g};
}

// ---- colormap (viridis-ish) ----
function cmap(t){ t=Math.max(0,Math.min(1,t));
  const r=Math.round(255*Math.min(1,Math.max(0,1.4*t-0.2)));
  const g=Math.round(255*Math.min(1,Math.max(0, t<0.5? 0.3+1.2*t : 0.9)));
  const b=Math.round(255*Math.min(1,Math.max(0, 0.9-1.1*t+0.3*Math.max(0,0.3-t))));
  return `rgb(${r},${g},${b})`;
}

// global max value for color/scale normalisation
let VMAX=0; for(const f of F) for(const x of f.ve) if(x>VMAX) VMAX=x;
const SMIN=S[0], SMAX=S[N-1];

// ---- static heatmap (European surface) drawn once ----
const heat=document.getElementById('heat'), hx=heat.getContext('2d');
function drawHeat(){
  const W=heat.width, H=heat.height;
  const cw=W/N, ch=H/NF;
  for(let fi=0; fi<NF; fi++){
    const v=F[fi].ve;
    const y=H - (fi+1)*ch; // τ=0 (expiry) at bottom, today at top
    for(let i=0;i<N;i++){
      hx.fillStyle=cmap(v[i]/VMAX);
      hx.fillRect(i*cw, y, Math.ceil(cw)+1, Math.ceil(ch)+1);
    }
  }
  // strike line
  const kx=(DATA.strike-SMIN)/(SMAX-SMIN)*W;
  hx.strokeStyle='rgba(255,255,255,.35)'; hx.setLineDash([4,4]);
  hx.beginPath(); hx.moveTo(kx,0); hx.lineTo(kx,H); hx.stroke(); hx.setLineDash([]);
}
let showBoundary=false;
function drawHeatOverlay(fi){
  drawHeat();
  const W=heat.width,H=heat.height,ch=H/NF;
  if(showBoundary){
    // American early-exercise boundary S*(τ): for a put, largest S where va≈payoff
    hx.strokeStyle='#fff'; hx.lineWidth=2; hx.beginPath(); let started=false;
    for(let f=0;f<NF;f++){
      const va=F[f].va; let sb=null;
      for(let i=N-1;i>=0;i--){ if(Math.abs(va[i]-payoff(S[i]))<1e-3){ sb=S[i]; break; } }
      if(sb===null) continue;
      const x=(sb-SMIN)/(SMAX-SMIN)*W, y=H-(f+1)*ch;
      if(!started){hx.moveTo(x,y);started=true;} else hx.lineTo(x,y);
    }
    hx.stroke(); hx.lineWidth=1;
  }
  // playhead
  const y=H-(fi+1)*(H/NF);
  hx.strokeStyle='#f85149'; hx.lineWidth=2;
  hx.beginPath(); hx.moveTo(0,y); hx.lineTo(W,y); hx.stroke(); hx.lineWidth=1;
}

// ---- line charts ----
function axes(ctx,W,H,pad){ ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,8); ctx.lineTo(pad,H-pad); ctx.lineTo(W-8,H-pad); ctx.stroke();
}
function plot(ctx,W,H,pad,vals,ymin,ymax,color,dash){
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.setLineDash(dash||[]);
  ctx.beginPath(); let started=false;
  for(let i=0;i<N;i++){ const val=vals[i]; if(isNaN(val))continue;
    const x=pad+(S[i]-SMIN)/(SMAX-SMIN)*(W-pad-8);
    const y=(H-pad)-(val-ymin)/(ymax-ymin)*(H-pad-8);
    if(!started){ctx.moveTo(x,y);started=true;} else ctx.lineTo(x,y);
  } ctx.stroke(); ctx.setLineDash([]);
}
const curve=document.getElementById('curve'), cx=curve.getContext('2d');
const greeksC=document.getElementById('greeks'), gx=greeksC.getContext('2d');

function drawFrame(fi){
  const fr=F[fi], tau=fr.tau;
  // value curve
  const W=curve.width,H=curve.height,pad=34;
  axes(cx,W,H,pad);
  const an=S.map(s=>bs(s,tau)), pay=S.map(s=>payoff(s));
  plot(cx,W,H,pad,pay,0,VMAX,'#3fb950');
  plot(cx,W,H,pad,an,0,VMAX,'#8b949e',[5,4]);
  plot(cx,W,H,pad,fr.va,0,VMAX,'#f0883e');
  plot(cx,W,H,pad,fr.ve,0,VMAX,'#2f81f7');

  // greeks
  const {d,g}=greeks(fr.ve);
  let gmin=Infinity,gmax=-Infinity;
  for(const x of g){ if(!isNaN(x)){ if(x<gmin)gmin=x; if(x>gmax)gmax=x; } }
  const W2=greeksC.width,H2=greeksC.height; axes(gx,W2,H2,pad);
  plot(gx,W2,H2,pad,d,-1.1,1.1,'#a371f7');
  // scale gamma into the same frame
  const gscale=g.map(x=>isNaN(x)?NaN:(x/Math.max(gmax,1e-9))*1.0);
  plot(gx,W2,H2,pad,gscale,-1.1,1.1,'#db61a2');

  // readouts
  const i0=S.findIndex(s=>s>=DATA.spot);
  document.getElementById('greekread').innerHTML =
    `at S=${DATA.spot}: <b class="eu">V=${fr.ve[i0].toFixed(4)}</b> &nbsp; Δ=${d[i0].toFixed(4)} &nbsp; Γ=${g[i0].toFixed(5)}`;
  document.getElementById('clock').innerHTML =
    `time to maturity τ = <b>${tau.toFixed(3)}</b> y &nbsp;•&nbsp; <b class="eu">European ${fr.ve[i0].toFixed(4)}</b> &nbsp; <b class="am">American ${fr.va[i0].toFixed(4)}</b> &nbsp; <b class="an">exact ${bs(DATA.spot,tau).toFixed(4)}</b>`;

  drawHeatOverlay(fi);
}

// ---- animation loop ----
let cur=0, playing=true;
const scrub=document.getElementById('scrub'); scrub.max=NF-1;
function tick(){ if(playing){ cur=(cur+1)%NF; scrub.value=cur; drawFrame(cur); } setTimeout(tick,55); }
document.getElementById('play').onclick=e=>{ playing=!playing; e.target.textContent=playing?'⏸ Pause':'▶ Play'; };
document.getElementById('mode').onclick=e=>{ showBoundary=!showBoundary; e.target.textContent=showBoundary?'Hide American boundary':'Show American boundary'; drawFrame(cur); };
scrub.oninput=e=>{ cur=+e.target.value; playing=false; document.getElementById('play').textContent='▶ Play'; drawFrame(cur); };

drawFrame(0); tick();
</script>
</body>
</html>"##;
