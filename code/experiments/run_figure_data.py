# =============================================================================
# Author: Truong Thanh Chung (1,2)
#   (1) University of Transport and Communications, No. 3 Cau Giay Street,
#       Hanoi, Vietnam
#   (2) Nha Trang University, 02 Nguyen Dinh Chieu, Khanh Hoa, Vietnam
#
# Code and reproducible baselines for the PGAE-A study: Physics-Guided
# Autoencoders with Unsupervised Feature Attention for One-Class Structural
# Damage Detection (Z24 bridge benchmark).
# =============================================================================

# -*- coding: utf-8 -*-
"""
z24_recompute_newbeta.py
NEW FILE — recomputes every seed-0 figure/table quantity at the AUTOMATICALLY
selected temperature beta* = geometric-mean of the window [lower knee, inflection]
of the PR(beta) curve (Kneedle lower knee + max-slope inflection). Dumps JSON in
the exact formats the current figure HTML files expect, and prints all scalar
numbers needed for the text and tables. Does not overwrite any figure.
"""
import json
import numpy as np
import torch
from torch import nn
from pathlib import Path
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent          # code/experiments
FEAT = _HERE.parent / "features"                 # feature matrices (code/features)
FIG  = _HERE.parent / "figures"                  # code/figures
DATA = FIG / "newbeta_data"
DATA.mkdir(exist_ok=True)

NH, BOTT, SEED, VAL = 8, 32, 0, 0.2
CHANNELS = ["R1V", "R2L", "R2T", "R2V", "R3V"]

# ── time-domain 49 names (utils4 order), + PSD/TRANS descriptor decoder ────────
TD = ["mean","median","mode","std","variance","skewness","kurtosis","coeff. of variation",
      "frac. zero values","frac. NaNs","longest strike above mean","longest strike below mean",
      "number of peaks (lag)","sum of reoccurring values","count of zeros","pct. reoccurring points",
      "ratio unique values","autocorrelation (lag)","partial autocorr. (lag)","Benford correlation",
      "c3 nonlinearity","cid-ce complexity","symmetry-looking flag","time-reversal asymmetry",
      "RMS","absolute energy","absolute sum of changes","mean abs. change","mean change",
      "mean 2nd derivative","variance>std flag","large-std flag","ratio beyond 1-sigma",
      "var. of STA histogram","var. of MTA histogram","var. of LTA histogram",
      "homogeneity of STA hist.","homogeneity of MTA hist.","homogeneity of LTA hist.",
      "PSD entropy","FFT entropy","PSD kurtosis","FFT kurtosis","PSD homogeneity","FFT homogeneity",
      "PSD bandwidth (95pct)","PSD bandwidth (50pct)","FFT bandwidth (95pct)","FFT bandwidth (50pct)"]

def desc_name(k):
    if k < 10:
        return ["mean amplitude","number of peaks","std of peak positions","mean peak position",
                "kurtosis of peak positions","skew of peak positions","mean prominence",
                "std of prominence","mean peak height","std of peak height"][k]
    if k < 34:
        seg=(k-10)//4; term=["x^3 coeff.","x^2 coeff.","x^1 coeff.","const. term"][(k-10)%4]
        return f"piecewise-cubic {term} (segment {seg+1}/6)"
    if k < 38: return f"peak-index rank {k-34+1} (top-4 by prominence)"
    if k < 42: return f"sharp-peak position, rank {k-38+1}"
    if k < 46: return f"sharp-peak height, rank {k-42+1}"
    if k < 58: lo=(k-46)*2.5; return f"sub-band peak height, {lo:.1f}--{lo+2.5:.1f}~Hz"
    if k < 70: lo=(k-58)*2.5; return f"sub-band peak rel.-position, {lo:.1f}--{lo+2.5:.1f}~Hz"
    return f"spectral centroid, quarter {k-70+1}/4"

def name_of(i):
    if i < 245:
        return "TD", CHANNELS[i//49], TD[i%49]
    elif i < 615:
        k=i-245; return "PSD", CHANNELS[k//74], desc_name(k%74)
    else:
        k=i-615; return "TRANS", f"{CHANNELS[1:][k//74]}/R1V", desc_name(k%74)

class AE(nn.Module):
    def __init__(self, d=906, b=32):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d,512), nn.ReLU(), nn.Linear(512,128), nn.ReLU(), nn.Linear(128,b), nn.ReLU())
        self.d = nn.Sequential(nn.Linear(b,128), nn.ReLU(), nn.Linear(128,512), nn.ReLU(), nn.Linear(512,d))
    def forward(self,x): return self.d(self.e(x))

def train_ae(Xtrain, Xval, device):
    torch.manual_seed(SEED)
    m = AE(Xtrain.shape[1], BOTT).to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3); crit = nn.MSELoss()
    xt = torch.from_numpy(Xtrain).to(device); xv = torch.from_numpy(Xval).to(device)
    best, wait, bs = 1e9, 0, None
    tr_hist, vl_hist = [], []
    for _ in range(600):
        m.train(); opt.zero_grad(); crit(m(xt), xt).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            tl = crit(m(xt), xt).item(); vl = crit(m(xv), xv).item()
        tr_hist.append(round(tl,6)); vl_hist.append(round(vl,6))
        if vl < best - 1e-7: best, wait, bs = vl, 0, {k:v.clone() for k,v in m.state_dict().items()}
        else: wait += 1
        if wait >= 60: break
    m.load_state_dict(bs); return m, tr_hist, vl_hist

# ── train seed 0 ──────────────────────────────────────────────────────────────
Xtr = np.load(FEAT / "X_train_new.npy").astype(np.float32)
ytr = np.load(FEAT / "labels_train_new.npy").astype(int)
Xte = np.load(FEAT / "X_test_new.npy").astype(np.float32)
yte = np.load(FEAT / "labels_test_new.npy").astype(int)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
d = Xtr.shape[1]
Xh = Xtr[ytr < NH]
rng = np.random.default_rng(SEED); idx = rng.permutation(len(Xh)); nv = int(round(len(idx)*VAL))
Xval, Xtrain = Xh[idx[:nv]], Xh[idx[nv:]]
print("Training AE (seed 0)...")
m, tr_hist, vl_hist = train_ae(Xtrain, Xval, device); m.eval()
with torch.no_grad():
    r_tr = (m(torch.from_numpy(Xtrain).to(device)) - torch.from_numpy(Xtrain).to(device)).cpu().numpy()
mi = (r_tr**2).mean(0)

# ── AUTOMATIC beta: knee-window log-midpoint ──────────────────────────────────
betas = np.geomspace(1e-3, 10, 2000); t = np.log10(betas)
def pr_curve():
    out=[]
    for b in betas:
        loga=-mi/b; loga-=loga.max(); a=np.exp(loga); a/=a.sum(); out.append(1.0/(a**2).sum())
    return np.array(out)
prs = pr_curve()
xn=(t-t.min())/(t.max()-t.min()); yn=(prs-prs.min())/(prs.max()-prs.min())
b_knee = betas[int(np.argmin(yn-xn))]
b_infl = betas[int(np.argmax(np.gradient(prs,t)))]
beta = float(np.sqrt(b_knee*b_infl))
# attention weights at beta*
a = np.exp(-mi/beta - (-mi/beta).max()); a /= a.sum()
PR = 1.0/(a**2).sum(); alpha = PR/d
print(f"knee  beta={b_knee:.4f}")
print(f"infl  beta={b_infl:.4f}")
print(f"beta* (log-mid) = {beta:.4f}   PR={PR:.1f}   alpha={alpha:.3f}")

# ── residuals for all 612 windows ─────────────────────────────────────────────
Xall = np.vstack([Xtr, Xte]); yall = np.concatenate([ytr, yte])
with torch.no_grad():
    r_all = (m(torch.from_numpy(Xall).to(device)) - torch.from_numpy(Xall).to(device)).cpu().numpy()
r2_all = r_all**2
healthy = yall < NH; damaged = ~healthy

scores_plain = r2_all.sum(1)
scores_attn  = (a[None,:]*r2_all).sum(1)

# held-out eval AUC (324 damaged vs 32 test-healthy)
is_test = np.zeros(len(yall), bool); is_test[len(ytr):] = True
ev = damaged | (is_test & healthy)
def auc_scores(s): return roc_auc_score(damaged[ev].astype(int), s[ev])
auc_plain = auc_scores(scores_plain)
auc_attn  = auc_scores(scores_attn)
print(f"\nAUC (held-out): AE-plain={auc_plain:.4f}   PGAE-A={auc_attn:.4f}")

# ── contribution decomposition (fig6 + Table 3) ───────────────────────────────
contrib = a[None,:]*r2_all
delta = contrib[damaged].mean(0) - contrib[healthy].mean(0)
total_gap = float(delta.sum())
order = np.argsort(delta)[::-1]; top20 = order[:20].tolist()
delta_norm = (delta/delta.max()).tolist()

print("\n% ---- Table 3 rows (rank, block, channel, descriptor, %gap) ----")
for rk,i in enumerate(top20,1):
    blk,ch,nm = name_of(i); print(f"{rk} & {blk} & {ch} & {nm} & {100*delta[i]/total_gap:.2f}\\%\\\\")
n_td=sum(1 for i in top20 if i<245); n_psd=sum(1 for i in top20 if 245<=i<615); n_tr=sum(1 for i in top20 if i>=615)
print(f"top-20 composition: TD={n_td}, PSD={n_psd}, TRANS={n_tr}")

# ── block AUCs (Table 4) ──────────────────────────────────────────────────────
b1, b2 = 245, 615
def block_auc(lo,hi):
    aw = a.copy(); mask=np.zeros(d,bool); mask[lo:hi]=True
    s = ((a*mask)[None,:]*r2_all).sum(1)
    return auc_scores(s)
auc_td   = block_auc(0,b1)
auc_psd  = block_auc(b1,b2)
auc_tr   = block_auc(b2,d)
# PSD+TRANS
mask=np.zeros(d,bool); mask[b1:d]=True
auc_psdtr = auc_scores(((a*mask)[None,:]*r2_all).sum(1))
print(f"\nTable 4 block AUCs: TD={auc_td:.3f}  TRANS={auc_tr:.3f}  PSD={auc_psd:.3f}  PSD+TRANS={auc_psdtr:.3f}  FULL={auc_attn:.3f}")

# ── control chart data (fig8) — normalized by 95th-pct healthy ────────────────
def pct(arr,q): s=np.sort(arr); return float(s[int(len(s)*q)])
order_state = np.argsort(yall, kind="stable")
sp = scores_plain[order_state]; sa = scores_attn[order_state]; ys = yall[order_state]
thr_sp = pct(scores_plain[healthy],0.95); thr_sa = pct(scores_attn[healthy],0.95)
sp_n = (sp/thr_sp); sa_n = (sa/thr_sa)
bounds = [int(np.searchsorted(ys, k)) for k in range(1,17)]
n_h = int(healthy.sum())
thr_d_sp = pct((scores_plain[damaged]/thr_sp),0.95)
thr_d_sa = pct((scores_attn[damaged]/thr_sa),0.95)
ctrl = {"sp":[round(v,5) for v in sp_n.tolist()], "sa":[round(v,5) for v in sa_n.tolist()],
        "bounds":bounds, "ymax":28.0, "n_healthy":n_h, "thr_h":1.0,
        "thr_d_sp":round(thr_d_sp,4), "thr_d_sa":round(thr_d_sa,4)}
(DATA/"control.json").write_text(json.dumps(ctrl));
print(f"\ncontrol: thr_d_sp(norm)={thr_d_sp:.3f}  thr_d_sa(norm)={thr_d_sa:.3f}")

# ── KDE data (fig9) — raw scores + AUCs ───────────────────────────────────────
kde = {"sp_h":[round(v,5) for v in scores_plain[healthy].tolist()],
       "sp_d":[round(v,5) for v in scores_plain[damaged].tolist()],
       "sa_h":[round(v,5) for v in scores_attn[healthy].tolist()],
       "sa_d":[round(v,5) for v in scores_attn[damaged].tolist()]}
(DATA/"kde.json").write_text(json.dumps(kde))
# AUCs on ALL windows (as the KDE caption uses in-sample seed-0 illustration)
auc_plain_all = roc_auc_score(damaged.astype(int), scores_plain)
auc_attn_all  = roc_auc_score(damaged.astype(int), scores_attn)
print(f"KDE AUC (all 612): AE-plain={auc_plain_all:.4f}  PGAE-A={auc_attn_all:.4f}")

# ── feature profile (fig7) — sorted by attention weight desc ──────────────────
rh = (r2_all[healthy]).mean(0); rd = (r2_all[damaged]).mean(0)
o = np.argsort(a)[::-1]
prof = {"rh":[round(float(v),6) for v in rh[o]], "rd":[round(float(v),6) for v in rd[o]],
        "awrh":[round(float(v),8) for v in (a[o]*rh[o])], "awrd":[round(float(v),8) for v in (a[o]*rd[o])],
        "top":906}
(DATA/"profile.json").write_text(json.dumps(prof))

# ── contribution (fig6) ───────────────────────────────────────────────────────
contribj = {"delta_norm":[round(v,8) for v in delta_norm], "top20":top20,
            "block1":245, "block2":615, "d":906}
(DATA/"contrib.json").write_text(json.dumps(contribj))

# ── PR curve + knee-window (fig5) ─────────────────────────────────────────────
prj = {"betas":[round(float(v),6) for v in betas], "prs":[round(float(v),2) for v in prs],
       "d":906, "b_knee":round(b_knee,4), "b_infl":round(b_infl,4),
       "beta":round(beta,4), "pr":round(float(PR),1), "alpha":round(float(alpha),3),
       "a_knee":round(float((1.0/((np.exp(-mi/b_knee-(-mi/b_knee).max())/np.exp(-mi/b_knee-(-mi/b_knee).max()).sum())**2).sum())/d),3),
       "a_infl":round(float((1.0/((np.exp(-mi/b_infl-(-mi/b_infl).max())/np.exp(-mi/b_infl-(-mi/b_infl).max()).sum())**2).sum())/d),3)}
(DATA/"pr.json").write_text(json.dumps(prj))

print(f"\nall JSON written to {DATA}")
print(f"SUMMARY: beta*={beta:.4f}, alpha={alpha:.3f}, PR={PR:.0f}, "
      f"knee(b={b_knee:.3f},a={prj['a_knee']:.2f}), infl(b={b_infl:.3f},a={prj['a_infl']:.2f})")
