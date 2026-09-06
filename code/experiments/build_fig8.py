# =============================================================================
# Builds fig8.html (train vs. held-out shading, added in response to
# Reviewer 2 #12) from code/figures/control_with_pool.json.
# Writes code/figures/fig8.html.
# =============================================================================
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
D = json.loads((FIG / "control_with_pool.json").read_text())

TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Figure 8 - Control Chart (top-bottom)</title>
<style>body{margin:0;background:#fff;display:flex;justify-content:center;
align-items:center;padding:16px;}canvas{display:block;}</style>
</head><body>
<canvas id="cv" width="700" height="560"></canvas>
<script>
/* -- data (regenerated for round-1 revision; adds is_train_pool for R2#12) -- */
const D=__DATA_JSON__;

const cv=document.getElementById('cv');
const ctx=cv.getContext('2d');
const W=700, H=560;
const DPR=4; cv.width=W*DPR; cv.height=H*DPR; cv.style.width=W+'px'; cv.style.height=H+'px'; ctx.scale(DPR,DPR);
const FONT="'Times New Roman', Times, serif";
const N=D.sp.length;

/* -- layout -- */
const pl=62, pr=16, pw=W-pl-pr;   // plot width = 622
const lgH=58;                       // legend height at top (2 rows now)
const gap=16;                       // gap between panels
const pbBot=46;                     // bottom margin (state labels + "State")
const ph=Math.floor((H-lgH-gap-pbBot)/2);  // each panel height

// Panel 0 (top) and Panel 1 (bottom)
const P=[
  {y0:lgH,          scores:D.sp, thrD:D.thr_d_sp, color:'#1565c0', label:'AE (plain)'},
  {y0:lgH+ph+gap,   scores:D.sa, thrD:D.thr_d_sa, color:'#2e7d32', label:'PGAE-A (attention)'}
];

function pxf(i){ return pl + i/N*pw; }
function pyf(p,v){ return p.y0 + ph - Math.max(0,Math.min(1,v/D.ymax))*ph; }

/* -- legend (top, shared, 2 rows) -- */
(function(){
  const row1=[
    {color:'#c62828', dash:[8,4], lw:1.8, text:'95th-percentile healthy threshold'},
    {color:'#6a1b9a', dash:[6,3], lw:1.8, text:'95th-percentile damaged threshold'},
  ];
  const fs=11, lw_=22, gap_=6, between=26;
  ctx.font=`${fs}px ${FONT}`;
  let tw=0;
  row1.forEach((it,k)=>{ tw+=lw_+gap_+ctx.measureText(it.text).width+(k<row1.length-1?between:0); });
  let cx=(W-tw)/2, cy=16;
  row1.forEach(it=>{
    ctx.strokeStyle=it.color; ctx.lineWidth=it.lw; ctx.setLineDash(it.dash);
    ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx+lw_,cy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#111'; ctx.font=`${fs}px ${FONT}`; ctx.textAlign='left';
    ctx.fillText(it.text, cx+lw_+gap_, cy+4);
    cx+=lw_+gap_+ctx.measureText(it.text).width+between;
  });
  // row 2: held-out healthy swatch
  const sw=16;
  const text2='healthy test-pool windows (never used in training)';
  ctx.font=`${fs}px ${FONT}`;
  const tw2=sw+gap_+ctx.measureText(text2).width;
  let cx2=(W-tw2)/2, cy2=cy+18;
  ctx.fillStyle='rgba(46,125,50,0.35)';
  ctx.fillRect(cx2, cy2-6, sw, 10);
  ctx.strokeStyle='#2e7d32'; ctx.lineWidth=1; ctx.strokeRect(cx2, cy2-6, sw, 10);
  ctx.fillStyle='#111'; ctx.textAlign='left';
  ctx.fillText(text2, cx2+sw+gap_, cy2+3);
})();

/* -- draw each panel -- */
P.forEach((p,pi)=>{
  const nhx=pxf(D.n_healthy);

  // region backgrounds
  ctx.fillStyle='rgba(197,225,165,0.22)'; ctx.fillRect(pl,p.y0,nhx-pl,ph);
  ctx.fillStyle='rgba(255,204,188,0.22)'; ctx.fillRect(nhx,p.y0,pl+pw-nhx,ph);

  // R2#12: shade the healthy TEST-POOL windows (held out from training)
  // with a darker overlay band, distinguishing them from the healthy
  // TRAINING-POOL windows that share the same light-green region.
  ctx.fillStyle='rgba(46,125,50,0.30)';
  let runStart=-1;
  for(let i=0;i<D.n_healthy;i++){
    const isTestPool = !D.is_train_pool[i];
    if(isTestPool && runStart<0) runStart=i;
    if((!isTestPool || i===D.n_healthy-1) && runStart>=0){
      const runEnd = isTestPool ? i+1 : i;
      ctx.fillRect(pxf(runStart), p.y0, pxf(runEnd)-pxf(runStart), ph);
      runStart=-1;
    }
  }

  // horizontal grid every 4 units
  ctx.strokeStyle='#e0e0e0'; ctx.lineWidth=0.7;
  for(let v=0;v<=D.ymax+.01;v+=4){
    const y=pyf(p,v);
    if(y>=p.y0-1 && y<=p.y0+ph+1){
      ctx.beginPath(); ctx.moveTo(pl,y); ctx.lineTo(pl+pw,y); ctx.stroke();
    }
  }

  // state boundaries
  ctx.strokeStyle='#bbb'; ctx.lineWidth=0.8; ctx.setLineDash([3,3]);
  D.bounds.forEach(i=>{
    const x=pxf(i);
    ctx.beginPath(); ctx.moveTo(x,p.y0); ctx.lineTo(x,p.y0+ph); ctx.stroke();
  });
  ctx.setLineDash([]);

  // healthy threshold — healthy region only
  const tyH=pyf(p,D.thr_h);
  ctx.strokeStyle='#c62828'; ctx.lineWidth=1.8; ctx.setLineDash([8,4]);
  ctx.beginPath(); ctx.moveTo(pl,tyH); ctx.lineTo(nhx,tyH); ctx.stroke();
  ctx.setLineDash([]);

  // damaged threshold — damaged region only
  if(p.thrD<=D.ymax*1.02){
    const tyD=pyf(p,p.thrD);
    ctx.strokeStyle='#6a1b9a'; ctx.lineWidth=1.8; ctx.setLineDash([6,3]);
    ctx.beginPath(); ctx.moveTo(nhx,tyD); ctx.lineTo(pl+pw,tyD); ctx.stroke();
    ctx.setLineDash([]);
  }

  // score line
  ctx.strokeStyle=p.color; ctx.lineWidth=1.1;
  ctx.beginPath();
  p.scores.forEach((v,i)=>{
    const x=pxf(i), y=pyf(p,v);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();

  // axes frame
  ctx.strokeStyle='#222'; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(pl,p.y0); ctx.lineTo(pl,p.y0+ph); ctx.lineTo(pl+pw,p.y0+ph); ctx.stroke();

  // y ticks (every 4)
  ctx.fillStyle='#333'; ctx.font=`10px ${FONT}`; ctx.textAlign='right';
  for(let v=0;v<=D.ymax+.01;v+=4){
    const y=pyf(p,v);
    if(y>=p.y0+2 && y<=p.y0+ph-2){
      ctx.beginPath(); ctx.moveTo(pl-3,y); ctx.lineTo(pl,y); ctx.stroke();
      ctx.fillText(v.toFixed(0),pl-5,y+3.5);
    }
  }

  // region labels — only above the top panel
  if(pi===0){
    ctx.font=`11px ${FONT}`; ctx.textAlign='center';
    ctx.fillStyle='#2e7d32'; ctx.fillText('Healthy',(pl+nhx)/2, p.y0-4);
    ctx.fillStyle='#c62828'; ctx.fillText('Damaged',(nhx+pl+pw)/2, p.y0-4);
  }

  // panel label (top-left inside)
  ctx.fillStyle='#111'; ctx.font=`bold 12px ${FONT}`; ctx.textAlign='left';
  ctx.fillText(p.label, pl+5, p.y0+14);

  // state labels S1-S17 — show only on bottom panel, or both
  const stateStarts=[0,...D.bounds];
  ctx.fillStyle='#333'; ctx.font=`9.5px ${FONT}`; ctx.textAlign='center';
  stateStarts.forEach((si,idx)=>{
    const end=idx+1<stateStarts.length?stateStarts[idx+1]:N;
    ctx.fillText('S'+(idx+1), pxf((si+end)/2), p.y0+ph+14);
  });
});

/* -- shared y-axis label -- */
const midY = lgH + ph + gap/2;  // vertical center of combined panels
ctx.save(); ctx.translate(12, midY); ctx.rotate(-Math.PI/2);
ctx.fillStyle='#333'; ctx.textAlign='center'; ctx.font=`11px ${FONT}`;
ctx.fillText('Normalized anomaly score', 0, 0); ctx.restore();

/* -- x-axis title (below bottom panel, shared) -- */
ctx.fillStyle='#333'; ctx.font=`12px ${FONT}`; ctx.textAlign='center';
ctx.fillText('State', W/2, H-6);
</script>
</body></html>
"""

html = TEMPLATE.replace("__DATA_JSON__", json.dumps(D, separators=(",", ":")))
(FIG / "fig8.html").write_text(html, encoding="utf-8")
print(f"wrote {FIG / 'fig8.html'}")
