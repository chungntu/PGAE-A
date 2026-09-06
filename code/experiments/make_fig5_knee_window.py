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
z24_fig5_knee_window.py
NEW FILE — paper Figure 5: automatic, one-class temperature selection.
Top panel: PR(beta) from healthy residuals, with the operating window
[lower knee, inflection] shaded and the chosen log-midpoint beta*.
Bottom panel: held-out AUC(beta) (labels used for validation only), confirming
the automatic point sits at the performance optimum. Saves png/fig5.png.
"""
import numpy as np
import torch
from torch import nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"
rcParams["axes.linewidth"] = 1.0

_HERE = Path(__file__).resolve().parent          # code/experiments
FEAT = _HERE.parent / "features"                 # feature matrices (code/features)
OUT  = _HERE.parent / "figures" / "png" / "fig5.png"

NH, BOTT, SEED, VAL = 8, 32, 0, 0.2
class AE(nn.Module):
    def __init__(self, d=906, b=32):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d,512), nn.ReLU(), nn.Linear(512,128), nn.ReLU(), nn.Linear(128,b), nn.ReLU())
        self.d = nn.Sequential(nn.Linear(b,128), nn.ReLU(), nn.Linear(128,512), nn.ReLU(), nn.Linear(512,d))
    def forward(self,x): return self.d(self.e(x))
def train_ae(Xtrain, Xval, device):
    torch.manual_seed(SEED); m = AE(Xtrain.shape[1], BOTT).to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3); crit = nn.MSELoss()
    xt = torch.from_numpy(Xtrain).to(device); xv = torch.from_numpy(Xval).to(device)
    best, wait, bs = 1e9, 0, None
    for _ in range(600):
        m.train(); opt.zero_grad(); crit(m(xt), xt).backward(); opt.step(); m.eval()
        with torch.no_grad(): vl = crit(m(xv), xv).item()
        if vl < best-1e-7: best,wait,bs = vl,0,{k:v.clone() for k,v in m.state_dict().items()}
        else: wait += 1
        if wait>=60: break
    m.load_state_dict(bs); return m

Xtr = np.load(FEAT/"X_train_new.npy").astype(np.float32); ytr = np.load(FEAT/"labels_train_new.npy").astype(int)
Xte = np.load(FEAT/"X_test_new.npy").astype(np.float32);  yte = np.load(FEAT/"labels_test_new.npy").astype(int)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); d = Xtr.shape[1]
Xh = Xtr[ytr<NH]; rng = np.random.default_rng(SEED); idx = rng.permutation(len(Xh)); nv = int(round(len(idx)*VAL))
Xval, Xtrain = Xh[idx[:nv]], Xh[idx[nv:]]
print("Training AE (seed 0)..."); m = train_ae(Xtrain, Xval, device); m.eval()
with torch.no_grad():
    r = (m(torch.from_numpy(Xtrain).to(device))-torch.from_numpy(Xtrain).to(device)).cpu().numpy()
mi = (r**2).mean(0)

Xall = np.vstack([Xtr,Xte]); yall = np.concatenate([ytr,yte])
is_test = np.zeros(len(yall),bool); is_test[len(ytr):] = True
ev = (yall>=NH) | (is_test & (yall<NH))
with torch.no_grad():
    r_ev = (m(torch.from_numpy(Xall[ev]).to(device))-torch.from_numpy(Xall[ev]).to(device)).cpu().numpy()
r2_ev = r_ev**2; dam = (yall[ev]>=NH).astype(int)

def attn(b): l=-mi/b; l-=l.max(); a=np.exp(l); a/=a.sum(); return a
def pr_of(b): a=attn(b); return 1.0/(a**2).sum()
def auc_of(b): a=attn(b); return roc_auc_score(dam,(a[None,:]*r2_ev).sum(1))

betas = np.geomspace(1e-3,10,1500); t = np.log10(betas)
prs = np.array([pr_of(b) for b in betas])
xn=(t-t.min())/(t.max()-t.min()); yn=(prs-prs.min())/(prs.max()-prs.min())
b_lo = betas[int(np.argmin(yn-xn))]; b_hi = betas[int(np.argmax(np.gradient(prs,t)))]
b_star = float(np.sqrt(b_lo*b_hi)); b_star = betas[int(np.argmin(np.abs(betas-b_star)))]
a_star = pr_of(b_star)/d
aucs = np.array([auc_of(b) for b in betas])
print(f"knee={b_lo:.3f} infl={b_hi:.3f} beta*={b_star:.3f} alpha={a_star:.2f} AUC={auc_of(b_star):.4f}")

fig,(axP,axA)=plt.subplots(2,1,figsize=(7.0,6.6),sharex=True)
for ax in (axP,axA):
    ax.axvspan(b_lo,b_hi,color="#bbdefb",alpha=0.55,lw=0)
    ax.axvline(b_star,ls="--",color="#6a1b9a",lw=1.5)
    ax.grid(True,which="both",ls=":",alpha=0.4)

axP.plot(betas,prs,color="#111",lw=2.2)
axP.scatter([b_lo],[pr_of(b_lo)],color="#2e7d32",s=70,marker="o",zorder=5,
            label=fr"lower knee ($\beta$={b_lo:.3f})")
axP.scatter([b_hi],[pr_of(b_hi)],color="#1565c0",s=70,marker="D",zorder=5,
            label=fr"inflection ($\beta$={b_hi:.3f})")
axP.scatter([b_star],[pr_of(b_star)],color="#6a1b9a",s=120,marker="*",zorder=6,
            label=fr"operating point $\beta^*$={b_star:.3f}")
axP.set_xscale("log"); axP.set_ylabel(r"PR($\beta$)  (effective features)")
axP.legend(fontsize=9,loc="upper left",framealpha=0.95)

axA.plot(betas,aucs,color="#c62828",lw=2.2)
axA.scatter([b_star],[auc_of(b_star)],color="#6a1b9a",s=120,marker="*",zorder=6,
            label=fr"$\beta^*$: AUC={auc_of(b_star):.3f}")
axA.scatter([b_lo],[auc_of(b_lo)],color="#2e7d32",s=48,marker="o",zorder=5)
axA.scatter([b_hi],[auc_of(b_hi)],color="#1565c0",s=48,marker="D",zorder=5)
axA.set_xscale("log"); axA.set_xlabel(r"sharpness parameter $\beta$  (log scale)")
axA.set_ylabel("ROC-AUC")
axA.legend(fontsize=9,loc="lower left",framealpha=0.95)

plt.tight_layout(); plt.savefig(OUT,dpi=200,bbox_inches="tight"); print(f"wrote {OUT}")
