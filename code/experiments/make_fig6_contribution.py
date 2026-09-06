# =============================================================================
# Author: Truong Thanh Chung (1,2)
#   (1) University of Transport and Communications, No. 3 Cau Giay Street,
#       Hanoi, Vietnam
#   (2) Nha Trang University, 02 Nguyen Dinh Chieu, Khanh Hoa, Vietnam
#
# Code and reproducible baselines for the PIAE-A study: Physics-Informed
# Autoencoders with Unsupervised Feature Attention for Label-Free Structural
# Damage Detection (Z24 bridge benchmark).
# =============================================================================

# -*- coding: utf-8 -*-
"""
z24_fig9_v2_contribution.py
NEW FILE — regenerates Figure 9 panel (b) to show per-feature CONTRIBUTION
to the healthy-vs-damaged score gap (delta_i), highlighting the top-20
contributors, instead of raw attention weight (which was dominated by
zero-variance dead columns). Also prints the top-20 name table for the paper.
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

NH, BOTT, SEED, VAL = 8, 32, 0, 0.2
TARGET_FRAC = 0.25
CHANNELS = ["R1V", "R2L", "R2T", "R2V", "R3V"]
FONT = "'Computer Modern', 'CMU Serif', 'Latin Modern Roman', serif"

TD = [
    "mean", "median", "mode", "std", "variance", "skewness", "kurtosis", "coeff. of variation",
    "frac. zero values", "frac. NaNs", "longest strike above mean", "longest strike below mean",
    "number of peaks (lag)", "sum of reoccurring values", "count of zeros",
    "pct. reoccurring points", "ratio unique values",
    "autocorrelation (lag)", "partial autocorr. (lag)", "Benford correlation",
    "c3 nonlinearity", "cid-ce complexity", "symmetry-looking flag", "time-reversal asymmetry",
    "RMS", "absolute energy", "absolute sum of changes", "mean abs. change",
    "mean change", "mean 2nd derivative", "variance>std flag", "large-std flag",
    "ratio beyond 1-sigma",
    "var. of STA histogram", "var. of MTA histogram", "var. of LTA histogram",
    "homogeneity of STA hist.", "homogeneity of MTA hist.", "homogeneity of LTA hist.",
    "PSD entropy", "FFT entropy", "PSD kurtosis", "FFT kurtosis", "PSD homogeneity", "FFT homogeneity",
    "PSD bandwidth (95pct)", "PSD bandwidth (50pct)", "FFT bandwidth (95pct)", "FFT bandwidth (50pct)",
]
assert len(TD) == 49

def name_of_desc(k):
    """k in 0..73, frequency_domain_features() return order."""
    if k < 10:
        labels = ["mean amplitude", "number of peaks", "std of peak positions",
                  "mean peak position", "kurtosis of peak positions", "skew of peak positions",
                  "mean prominence", "std of prominence", "mean peak height", "std of peak height"]
        return labels[k]
    if k < 34:
        seg = (k - 10) // 4
        term = ["x^3 coeff.", "x^2 coeff.", "x^1 coeff.", "const. term"][(k - 10) % 4]
        return f"piecewise-cubic {term} (segment {seg+1}/6)"
    if k < 38:
        return f"peak-index rank {k-34+1} (top-4 by prominence)"
    if k < 42:
        return f"sharp-peak position, rank {k-38+1}"
    if k < 46:
        return f"sharp-peak height, rank {k-42+1}"
    if k < 58:
        lo = (k - 46) * 2.5
        return f"sub-band peak height [{lo:.1f}-{lo+2.5:.1f} Hz]"
    if k < 70:
        lo = (k - 58) * 2.5
        return f"sub-band peak rel.-position [{lo:.1f}-{lo+2.5:.1f} Hz]"
    return f"spectral centroid, quarter {k-70+1}/4"

def name_of(i):
    if i < 245:
        ch, off = CHANNELS[i // 49], i % 49
        return "TD", ch, TD[off]
    elif i < 615:
        k = i - 245
        return "PSD", CHANNELS[k // 74], name_of_desc(k % 74)
    else:
        k = i - 615
        return "TRANS", f"{CHANNELS[1:][k // 74]}/R1V", name_of_desc(k % 74)

class AE(nn.Module):
    def __init__(self, d=906, b=32):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d,512), nn.ReLU(), nn.Linear(512,128), nn.ReLU(), nn.Linear(128,b), nn.ReLU())
        self.d = nn.Sequential(nn.Linear(b,128), nn.ReLU(), nn.Linear(128,512), nn.ReLU(), nn.Linear(512,d))
    def forward(self,x): return self.d(self.e(x))

def train_ae(Xtrain, Xval, device):
    torch.manual_seed(SEED)
    m = AE(Xtrain.shape[1], BOTT).to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    crit = nn.MSELoss()
    xt = torch.from_numpy(Xtrain).to(device); xv = torch.from_numpy(Xval).to(device)
    best, wait, bs = 1e9, 0, None
    for _ in range(600):
        m.train(); opt.zero_grad(); crit(m(xt), xt).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(xv), xv).item()
        if vl < best - 1e-7: best, wait, bs = vl, 0, {k:v.clone() for k,v in m.state_dict().items()}
        else: wait += 1
        if wait >= 60: break
    m.load_state_dict(bs)
    return m

# ── train + compute weights/contributions (identical protocol to earlier scripts) ──
Xtr = np.load(FEAT / "X_train_new.npy").astype(np.float32)
ytr = np.load(FEAT / "labels_train_new.npy").astype(int)
Xte = np.load(FEAT / "X_test_new.npy").astype(np.float32)
yte = np.load(FEAT / "labels_test_new.npy").astype(int)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
d = Xtr.shape[1]

Xh = Xtr[ytr < NH]
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(Xh))
nv = int(round(len(idx)*VAL))
Xval, Xtrain = Xh[idx[:nv]], Xh[idx[nv:]]

m = train_ae(Xtrain, Xval, device)
m.eval()
with torch.no_grad():
    r_train_healthy = (m(torch.from_numpy(Xtrain).to(device)) - torch.from_numpy(Xtrain).to(device)).cpu().numpy()
mi = (r_train_healthy**2).mean(0)

betas = np.geomspace(0.001, 10, 500)
target = TARGET_FRAC*d
best_beta, best_a, best_pr = None, None, 1e9
for b in betas:
    loga = -mi/b; loga -= loga.max()
    a = np.exp(loga); a /= a.sum()
    p = 1.0/(a**2).sum()
    if abs(p-target) < abs(best_pr-target):
        best_beta, best_a, best_pr = b, a, p
a = best_a
print(f"beta*={best_beta:.4f}  PR={best_pr:.1f}")

Xall = np.vstack([Xtr, Xte]); yall = np.concatenate([ytr, yte])
with torch.no_grad():
    r_all = (m(torch.from_numpy(Xall).to(device)) - torch.from_numpy(Xall).to(device)).cpu().numpy()
r2_all = r_all**2
contrib_all = a[None,:] * r2_all
healthy = yall < NH; damaged = ~healthy
delta = contrib_all[damaged].mean(0) - contrib_all[healthy].mean(0)
total_gap = float(delta.sum())

order = np.argsort(delta)[::-1]
TOP20 = order[:20].tolist()
block1, block2 = 245, 615

# ── print LaTeX-ready top-20 table ──
print("\n% ---- LaTeX table rows (rank, block, channel, name, delta_i, %gap) ----")
for rank, i in enumerate(TOP20, 1):
    block, ch, nm = name_of(i)
    pct = 100*delta[i]/total_gap
    print(f"{rank} & {block} & {ch} & {nm} & {delta[i]:.4f} & {pct:.2f}\\%\\\\")

# ── write JSON for the figure ──
out = {
    "delta_norm": [round(float(v), 8) for v in (delta/delta.max())],
    "top20": TOP20,
    "block1": block1, "block2": block2, "d": int(d),
    "beta_op": best_beta, "pr_op": best_pr,
    # keep panel (a) AUC-vs-beta data identical to the previous figure
    "auc_betas": [0.01, 0.05, 0.1, 0.3, 1.0],
    "auc_vals":  [0.6056, 0.8009, 0.7904, 0.7267, 0.6771],
    "auc_plain": 0.6458,
}
(FIG / "fig9_v2_data.json").write_text(json.dumps(out), encoding="utf-8")
print(f"\nwrote {FIG / 'fig9_v2_data.json'}")
