# =============================================================================
# NEW FILE — response to Reviewer 2 #17 (RMAT-2026-0601, round 1)
#
# Standalone, isolated analysis. Reads ONLY the copied feature matrices in
# review vong 1/response/data/ (copies of code/features/X_*_new.npy). Does not
# import from, or write into, the original code/ tree.
#
# Reviewer's request, verbatim: "The statistical analysis should account for
# dependence among windows from the same structural state and repeated
# evaluation on the same test set."
#
# The manuscript's Hanley-McNeil standard error (Eq. 8) and the paired t-test
# in Sec. 3.5 both treat the 324 damaged and 32 healthy test-pool windows as
# independent observations. In fact they come from only 9 damaged states and
# 8 healthy states, and windows from the same state are correlated (same
# measurement campaign, same physical condition). This script replaces the
# i.i.d. assumption with a cluster (block) bootstrap: states, not individual
# windows, are resampled with replacement, which is the standard way to get
# a valid confidence interval for clustered data.
#
# Uses a single representative trained model (seed 0), matching the
# "representative run" convention already used in Sec. 3.4 of the main text
# (Fig. 5 / Table 3 / Table 4 all come from one representative run there).
# =============================================================================
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
DATA = _HERE.parent / "data"
OUT = _HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

NH, BOTT, VAL = 8, 32, 0.2
REP_SEED = 0
N_BOOT = 5000


class AE(nn.Module):
    def __init__(self, d=906, b=32):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, b), nn.ReLU())
        self.d = nn.Sequential(nn.Linear(b, 128), nn.ReLU(), nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, d))

    def forward(self, x):
        return self.d(self.e(x))


def train_ae(Xtrain, Xval, device, seed):
    torch.manual_seed(seed)
    m = AE(Xtrain.shape[1], BOTT).to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    crit = nn.MSELoss()
    xt = torch.from_numpy(Xtrain).to(device)
    xv = torch.from_numpy(Xval).to(device)
    best, wait, bs = 1e9, 0, None
    for _ in range(600):
        m.train(); opt.zero_grad(); crit(m(xt), xt).backward(); opt.step(); m.eval()
        with torch.no_grad():
            vl = crit(m(xv), xv).item()
        if vl < best - 1e-7:
            best, wait, bs = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
        if wait >= 60:
            break
    m.load_state_dict(bs)
    return m


def auc(sd, sh):
    return roc_auc_score(np.r_[np.zeros(len(sh)), np.ones(len(sd))], np.r_[sh, sd])


def beta_kneewin(mi):
    betas = np.geomspace(1e-3, 10, 2000)
    t = np.log10(betas)
    prs = np.empty(len(betas))
    for i, b in enumerate(betas):
        l = -mi / b; l -= l.max(); a = np.exp(l); a /= a.sum(); prs[i] = 1.0 / (a ** 2).sum()
    xn = (t - t.min()) / (t.max() - t.min())
    yn = (prs - prs.min()) / (prs.max() - prs.min())
    b_lo = betas[int(np.argmin(yn - xn))]
    b_hi = betas[int(np.argmax(np.gradient(prs, t)))]
    return float(np.sqrt(b_lo * b_hi))


def main():
    Xtr = np.load(DATA / "X_train_new.npy").astype(np.float32)
    ytr = np.load(DATA / "labels_train_new.npy").astype(int)
    Xte = np.load(DATA / "X_test_new.npy").astype(np.float32)
    yte = np.load(DATA / "labels_test_new.npy").astype(int)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    Xh_all = Xtr[ytr < NH]
    Xd = np.concatenate([Xtr[ytr >= NH], Xte[yte >= NH]])
    yd = np.concatenate([ytr[ytr >= NH], yte[yte >= NH]])   # damaged state labels, 8..16
    Xht = Xte[yte < NH]
    yht = yte[yte < NH]                                      # healthy-test state labels, 0..7
    print(f"Xd={Xd.shape} ({len(set(yd.tolist()))} damaged states)  "
          f"Xht={Xht.shape} ({len(set(yht.tolist()))} healthy-test states)")

    rng = np.random.default_rng(REP_SEED)
    idx = rng.permutation(len(Xh_all))
    nv = int(round(len(idx) * VAL))
    Xval, Xtrain = Xh_all[idx[:nv]], Xh_all[idx[nv:]]
    m = train_ae(Xtrain, Xval, device, REP_SEED)

    def res(X):
        with torch.no_grad():
            x = torch.from_numpy(X).to(device)
            return (m(x) - x).cpu().numpy()

    mi = (res(Xtrain) ** 2).mean(0)
    beta = beta_kneewin(mi)
    l = -mi / beta; l -= l.max(); a = np.exp(l); a /= a.sum()

    score_d = (a * res(Xd) ** 2).sum(1)    # per-window damaged scores
    score_h = (a * res(Xht) ** 2).sum(1)   # per-window healthy-test scores
    point_auc = auc(score_d, score_h)
    print(f"representative run (seed={REP_SEED}): beta={beta:.4f}  point AUC={point_auc:.4f}")

    damaged_states = sorted(set(yd.tolist()))
    healthy_states = sorted(set(yht.tolist()))
    d_by_state = {s: score_d[yd == s] for s in damaged_states}
    h_by_state = {s: score_h[yht == s] for s in healthy_states}
    print(f"damaged states: {damaged_states} (n per state: {[len(v) for v in d_by_state.values()]})")
    print(f"healthy-test states: {healthy_states} (n per state: {[len(v) for v in h_by_state.values()]})")

    # ---- cluster (block) bootstrap: resample STATES with replacement ----
    boot_rng = np.random.default_rng(12345)
    boot_aucs = []
    for _ in range(N_BOOT):
        d_states_bs = boot_rng.choice(damaged_states, size=len(damaged_states), replace=True)
        h_states_bs = boot_rng.choice(healthy_states, size=len(healthy_states), replace=True)
        sd_bs = np.concatenate([d_by_state[s] for s in d_states_bs])
        sh_bs = np.concatenate([h_by_state[s] for s in h_states_bs])
        # a resample could in principle be all-one-class in a degenerate draw; skip such draws
        if len(sd_bs) == 0 or len(sh_bs) == 0:
            continue
        boot_aucs.append(auc(sd_bs, sh_bs))
    boot_aucs = np.array(boot_aucs)

    block_se = float(boot_aucs.std(ddof=1))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

    # ---- naive i.i.d.-window bootstrap for comparison (resamples individual windows) ----
    iid_rng = np.random.default_rng(54321)
    iid_aucs = []
    for _ in range(N_BOOT):
        sd_bs = iid_rng.choice(score_d, size=len(score_d), replace=True)
        sh_bs = iid_rng.choice(score_h, size=len(score_h), replace=True)
        iid_aucs.append(auc(sd_bs, sh_bs))
    iid_aucs = np.array(iid_aucs)
    iid_se = float(iid_aucs.std(ddof=1))
    iid_ci_lo, iid_ci_hi = np.percentile(iid_aucs, [2.5, 97.5])

    # Hanley-McNeil reference (assumes i.i.d. windows), as computed in the main text, Eq. 8
    A = point_auc
    nD, nHt = len(Xd), len(Xht)
    Q1 = A / (2 - A); Q2 = 2 * A * A / (1 + A)
    hm_se = float(np.sqrt((A * (1 - A) + (nD - 1) * (Q1 - A * A) + (nHt - 1) * (Q2 - A * A)) / (nD * nHt)))

    report = {
        "representative_seed": REP_SEED, "point_auc": float(point_auc), "beta": float(beta),
        "n_damaged_states": len(damaged_states), "n_healthy_test_states": len(healthy_states),
        "n_boot": N_BOOT,
        "block_bootstrap_by_state": {"se": block_se, "ci95": [float(ci_lo), float(ci_hi)]},
        "iid_window_bootstrap": {"se": iid_se, "ci95": [float(iid_ci_lo), float(iid_ci_hi)]},
        "hanley_mcneil_se_iid_assumption": hm_se,
    }
    (OUT / "block_bootstrap_ci.json").write_text(json.dumps(report, indent=2))

    print("\n=== SUMMARY ===")
    print(f"Point AUC (representative run): {point_auc:.4f}")
    print(f"Hanley-McNeil SE (i.i.d. windows, as in main text Eq. 8): {hm_se:.4f}")
    print(f"i.i.d.-window bootstrap SE (resample windows):            {iid_se:.4f}  "
          f"95% CI [{iid_ci_lo:.4f}, {iid_ci_hi:.4f}]")
    print(f"Block bootstrap SE (resample STATES, accounts for dependence): {block_se:.4f}  "
          f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"\nsaved: {OUT/'block_bootstrap_ci.json'}")


if __name__ == "__main__":
    main()
