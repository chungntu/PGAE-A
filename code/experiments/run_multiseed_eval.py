# =============================================================================
# Author: Truong Thanh Chung (1,2)
#   (1) University of Transport and Communications, No. 3 Cau Giay Street,
#       Hanoi, Vietnam
#   (2) Nha Trang University, 02 Nguyen Dinh Chieu, Khanh Hoa, Vietnam
#
# Code and reproducible baselines for the PGAE-A study: Physics-Informed
# Autoencoders with Unsupervised Feature Attention for One-Class Structural
# Damage Detection (Z24 bridge benchmark).
# =============================================================================

# -*- coding: utf-8 -*-
"""
z24_piae_multiseed_kneewin.py
NEW FILE — 10-seed evaluation of PGAE-A with the AUTOMATIC knee-window
log-midpoint temperature rule (no target retention fraction). For each seed:
train AE on healthy-train; select beta as the geometric mean of [lower knee,
inflection] of the PR(beta) curve (healthy residuals only); score plain and
attention, plus the supervised elliptic-envelope reference on the same split.
Outputs mean +/- std, 95% t-CI, paired test, Hanley-McNeil SE.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from scipy import stats

_HERE = Path(__file__).resolve().parent          # code/experiments
FEAT = _HERE.parent / "features"                 # feature matrices (code/features)
OUT  = _HERE / "results"                          # code/experiments/results
NH, BOTT, N_SEEDS, VAL = 8, 32, 10, 0.2
SELECTED = [673, 191, 691, 670]

class AE(nn.Module):
    def __init__(self, d=906, b=32):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d,512), nn.ReLU(), nn.Linear(512,128), nn.ReLU(), nn.Linear(128,b), nn.ReLU())
        self.d = nn.Sequential(nn.Linear(b,128), nn.ReLU(), nn.Linear(128,512), nn.ReLU(), nn.Linear(512,d))
    def forward(self,x): return self.d(self.e(x))

def train_ae(Xtrain, Xval, device, seed):
    torch.manual_seed(seed)
    m = AE(Xtrain.shape[1], BOTT).to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3); crit = nn.MSELoss()
    xt = torch.from_numpy(Xtrain).to(device); xv = torch.from_numpy(Xval).to(device)
    best, wait, bs = 1e9, 0, None
    for _ in range(600):
        m.train(); opt.zero_grad(); crit(m(xt), xt).backward(); opt.step(); m.eval()
        with torch.no_grad(): vl = crit(m(xv), xv).item()
        if vl < best-1e-7: best,wait,bs = vl,0,{k:v.clone() for k,v in m.state_dict().items()}
        else: wait += 1
        if wait >= 60: break
    m.load_state_dict(bs); return m

def auc(sd, sh):
    return roc_auc_score(np.r_[np.zeros(len(sh)), np.ones(len(sd))], np.r_[sh, sd])

def beta_kneewin(mi):
    """Automatic label-free beta: geometric mean of [lower knee, inflection]."""
    betas = np.geomspace(1e-3, 10, 2000); t = np.log10(betas)
    prs = np.empty(len(betas))
    for i,b in enumerate(betas):
        l = -mi/b; l -= l.max(); a = np.exp(l); a /= a.sum(); prs[i] = 1.0/(a**2).sum()
    xn = (t-t.min())/(t.max()-t.min()); yn = (prs-prs.min())/(prs.max()-prs.min())
    b_lo = betas[int(np.argmin(yn-xn))]          # lower knee (Kneedle)
    b_hi = betas[int(np.argmax(np.gradient(prs,t)))]  # inflection (max slope)
    return float(np.sqrt(b_lo*b_hi)), float(b_lo), float(b_hi)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Xtr = np.load(FEAT/"X_train_new.npy").astype(np.float32); ytr = np.load(FEAT/"labels_train_new.npy").astype(int)
    Xte = np.load(FEAT/"X_test_new.npy").astype(np.float32);  yte = np.load(FEAT/"labels_test_new.npy").astype(int)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); d = Xtr.shape[1]
    Xh_all = Xtr[ytr<NH]
    Xd  = np.concatenate([Xtr[ytr>=NH], Xte[yte>=NH]])   # all 324 damaged
    Xht = Xte[yte<NH]                                     # 32 test healthy
    Xd4, Xht4 = Xd[:,SELECTED], Xht[:,SELECTED]

    rows=[]
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed); idx = rng.permutation(len(Xh_all)); nv = int(round(len(idx)*VAL))
        Xval, Xtrain = Xh_all[idx[:nv]], Xh_all[idx[nv:]]
        m = train_ae(Xtrain, Xval, device, seed)
        def res(X):
            with torch.no_grad():
                x = torch.from_numpy(X).to(device); return (m(x)-x).cpu().numpy()
        mi = (res(Xtrain)**2).mean(0)
        beta, b_lo, b_hi = beta_kneewin(mi)
        l = -mi/beta; l -= l.max(); a = np.exp(l); a /= a.sum()
        pr = 1.0/(a**2).sum()
        rd, rh = res(Xd), res(Xht)
        auc_plain = auc((rd**2).sum(1), (rh**2).sum(1))
        auc_attn  = auc((a*rd**2).sum(1), (a*rh**2).sum(1))
        mu = Xtrain[:,SELECTED].mean(0); cov = np.cov(Xtrain[:,SELECTED].T); ic = np.linalg.pinv(cov)
        def maha(X): dl=X-mu; return np.einsum("ij,jk,ik->i", dl, ic, dl)
        auc_ell = auc(maha(Xd4), maha(Xht4))
        rows.append({"seed":seed,"beta":beta,"b_knee":b_lo,"b_infl":b_hi,"pr":float(pr),
                     "alpha":float(pr/d),"auc_plain":float(auc_plain),"auc_attn":float(auc_attn),
                     "auc_elliptic":float(auc_ell)})
        print(f"seed {seed}: beta={beta:.4f} (knee={b_lo:.3f} infl={b_hi:.3f} PR={pr:.0f} a={pr/d:.2f}) "
              f"| plain={auc_plain:.3f} attn={auc_attn:.3f} ell={auc_ell:.3f}", flush=True)

    def summ(k):
        v = np.array([r[k] for r in rows]); m_,s_ = v.mean(), v.std(ddof=1)
        h = stats.t.ppf(0.975, len(v)-1)*s_/np.sqrt(len(v)); return m_,s_,h,v
    mp,sp,hp,vp = summ("auc_plain"); ma,sa,ha,va = summ("auc_attn")
    me,se_,he,ve = summ("auc_elliptic"); mb,sb,hb,vb = summ("beta")
    mal,sal,_,_ = summ("alpha")
    diff = va-vp; t_stat,p_val = stats.ttest_rel(va,vp)
    A=ma; nD,nHt=len(Xd),len(Xht); Q1=A/(2-A); Q2=2*A*A/(1+A)
    hm_se = np.sqrt((A*(1-A)+(nD-1)*(Q1-A*A)+(nHt-1)*(Q2-A*A))/(nD*nHt))
    print(f"\n=== SUMMARY (N={N_SEEDS}, knee-window log-mid beta) ===")
    print(f"beta   : {mb:.4f} +/- {sb:.4f}    alpha: {mal:.3f} +/- {sal:.3f}")
    print(f"AE-plain: AUC {mp:.4f} +/- {sp:.4f}   95%CI [{mp-hp:.3f},{mp+hp:.3f}]")
    print(f"PGAE-A  : AUC {ma:.4f} +/- {sa:.4f}   95%CI [{ma-ha:.3f},{ma+ha:.3f}]")
    print(f"Elliptic: AUC {me:.4f} +/- {se_:.4f}   95%CI [{me-he:.3f},{me+he:.3f}]")
    print(f"paired (attn-plain): {diff.mean():.4f} +/- {diff.std(ddof=1):.4f}  p={p_val:.2e}")
    print(f"Hanley-McNeil SE at AUC={A:.3f}: {hm_se:.4f}")
    (OUT/"piae_multiseed_kneewin.json").write_text(json.dumps({"rows":rows,"summary":{
        "beta_mean":mb,"beta_std":sb,"alpha_mean":mal,"alpha_std":sal,
        "plain":{"mean":mp,"std":sp},"piae_a":{"mean":ma,"std":sa},"elliptic":{"mean":me,"std":se_},
        "paired_diff_mean":float(diff.mean()),"paired_diff_std":float(diff.std(ddof=1)),
        "paired_t_p":float(p_val),"hanley_mcneil_se":float(hm_se),"n_damage":nD,"n_healthy_test":nHt}}, indent=2))
    print("saved results/piae_multiseed_kneewin.json")

if __name__ == "__main__":
    main()
