# =============================================================================
# NEW FILE — response to Reviewer 2 #10 (RMAT-2026-0601, round 1)
#
# Reviewer's request, verbatim: "Such constant or near-constant features
# should preferably be removed, followed by a sensitivity analysis of their
# effect on beta, participation ratio, and AUC."
#
# Removes the 70 constant features (std < 1e-9 on the healthy training pool,
# same threshold as code/experiments/verify_constant_columns.py) from the
# 906-feature bank, retrains PGAE-A end to end on the remaining 836 features,
# and compares beta, participation ratio, and AUC against the original
# 906-feature protocol -- same ten-seed evaluation as Table 5.
# =============================================================================
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from scipy import stats

_HERE = Path(__file__).resolve().parent            # response/code
DATA = _HERE.parent / "features"
OUT = _HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

NH, BOTT, N_SEEDS, VAL = 8, 32, 10, 0.2
CONST_THRESH = 1e-9


class AE(nn.Module):
    def __init__(self, d, b=32):
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


def run_condition(Xh_all, Xd, Xht, device, label):
    rows = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(Xh_all))
        nv = int(round(len(idx) * VAL))
        Xval, Xtrain = Xh_all[idx[:nv]], Xh_all[idx[nv:]]
        m = train_ae(Xtrain, Xval, device, seed)

        def res(X):
            with torch.no_grad():
                x = torch.from_numpy(X).to(device)
                return (m(x) - x).cpu().numpy()

        mi = (res(Xtrain) ** 2).mean(0)
        beta = beta_kneewin(mi)
        l = -mi / beta; l -= l.max(); a = np.exp(l); a /= a.sum()
        pr = 1.0 / (a ** 2).sum()

        rd, rh = res(Xd), res(Xht)
        a_auc = auc((a * rd ** 2).sum(1), (a * rh ** 2).sum(1))
        rows.append({"seed": seed, "beta": beta, "pr": float(pr), "auc": float(a_auc)})
        print(f"  [{label}] seed {seed}: beta={beta:.4f} PR={pr:.1f} AUC={a_auc:.4f}", flush=True)

    def summ(k):
        v = np.array([r[k] for r in rows])
        return float(v.mean()), float(v.std(ddof=1)), v

    bm, bs, bv = summ("beta")
    pm, ps, pv = summ("pr")
    am, as_, av = summ("auc")
    return {"label": label, "d": int(Xh_all.shape[1]),
            "beta_mean": bm, "beta_std": bs, "beta_runs": bv.tolist(),
            "pr_mean": pm, "pr_std": ps, "pr_runs": pv.tolist(),
            "auc_mean": am, "auc_std": as_, "auc_runs": av.tolist()}


def main():
    Xtr = np.load(DATA / "X_train_new.npy").astype(np.float32)
    ytr = np.load(DATA / "labels_train_new.npy").astype(int)
    Xte = np.load(DATA / "X_test_new.npy").astype(np.float32)
    yte = np.load(DATA / "labels_test_new.npy").astype(int)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    Xh_all = Xtr[ytr < NH]
    Xd = np.concatenate([Xtr[ytr >= NH], Xte[yte >= NH]])
    Xht = Xte[yte < NH]

    std_train = Xh_all.std(axis=0)
    const_mask = std_train < CONST_THRESH
    print(f"constant features (std < {CONST_THRESH:g} on healthy training pool): "
          f"{const_mask.sum()} / {Xh_all.shape[1]}")
    keep = ~const_mask

    print("\n=== Condition 1/2: FULL (906 features, original protocol) ===")
    full = run_condition(Xh_all, Xd, Xht, device, "full_906")

    print("\n=== Condition 2/2: EXCL-CONSTANT (constant features removed) ===")
    excl = run_condition(Xh_all[:, keep], Xd[:, keep], Xht[:, keep], device, f"excl_const_{keep.sum()}")

    beta_p = stats.ttest_ind(full["beta_runs"], excl["beta_runs"], equal_var=False).pvalue
    pr_p = stats.ttest_ind(full["pr_runs"], excl["pr_runs"], equal_var=False).pvalue
    auc_p = stats.ttest_ind(full["auc_runs"], excl["auc_runs"], equal_var=False).pvalue

    report = {"n_constant_removed": int(const_mask.sum()), "full": full, "excl_constant": excl,
              "welch_p": {"beta": float(beta_p), "pr": float(pr_p), "auc": float(auc_p)}}
    (OUT / "constant_feature_removal.json").write_text(json.dumps(report, indent=2))

    print("\n=== SUMMARY ===")
    print(f"FULL (d=906)          : beta={full['beta_mean']:.4f}+/-{full['beta_std']:.4f}  "
          f"PR={full['pr_mean']:.1f}+/-{full['pr_std']:.1f}  AUC={full['auc_mean']:.4f}+/-{full['auc_std']:.4f}")
    print(f"EXCL-CONSTANT (d={excl['d']}) : beta={excl['beta_mean']:.4f}+/-{excl['beta_std']:.4f}  "
          f"PR={excl['pr_mean']:.1f}+/-{excl['pr_std']:.1f}  AUC={excl['auc_mean']:.4f}+/-{excl['auc_std']:.4f}")
    print(f"Welch p-values: beta={beta_p:.3g}  PR={pr_p:.3g}  AUC={auc_p:.3g}")
    print(f"\nsaved: {OUT/'constant_feature_removal.json'}")


if __name__ == "__main__":
    main()
