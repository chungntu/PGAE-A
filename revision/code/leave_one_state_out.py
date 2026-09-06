# =============================================================================
# NEW FILE — response to Reviewer 2 #11 (RMAT-2026-0601, round 1)
#
# Standalone, isolated analysis. Reads ONLY the copied feature matrices in
# review vong 1/response/data/ (copies of code/features/X_*_new.npy). Does not
# import from, or write into, the original code/ tree.
#
# Reviewer's request, verbatim: "The random 80/20 train-validation split may
# place highly correlated windows from the same structural state in both
# subsets. A state-wise or temporally separated validation, preferably
# including leave-one-state/scenario-out analysis, would provide stronger
# evidence of generalization."
#
# Replaces the random 80/20 split of the 256-window healthy training pool
# with leave-one-state-out: each of the 8 healthy states (S1-S8, 32 windows
# each) is held out in turn as the validation split (used for AE early
# stopping and, matching the standard protocol, NOT used for attention-weight
# calibration -- mi is computed from the 7-state training fold, exactly as
# mi is computed from Xtrain in the random-split protocol). The evaluation
# set (324 damaged + 32 healthy test-pool windows) is unchanged across folds
# and identical to Table 5, so AUC is directly comparable.
# =============================================================================
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from scipy import stats

_HERE = Path(__file__).resolve().parent
DATA = _HERE.parent / "data"
OUT = _HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

NH, BOTT = 8, 32
STATE_NAMES = {0: "S1", 1: "S2", 2: "S3", 3: "S4", 4: "S5", 5: "S6", 6: "S7", 7: "S8"}


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
    yh_all = ytr[ytr < NH]  # 0..7, one of the 8 healthy states
    Xd = np.concatenate([Xtr[ytr >= NH], Xte[yte >= NH]])
    Xht = Xte[yte < NH]
    print(f"Xh_all={Xh_all.shape}  per-state counts: "
          f"{ {int(s): int((yh_all==s).sum()) for s in range(NH)} }")

    rows = []
    for held_out in range(NH):
        val_mask = yh_all == held_out
        Xval, Xtrain = Xh_all[val_mask], Xh_all[~val_mask]
        m = train_ae(Xtrain, Xval, device, seed=held_out)

        def res(X):
            with torch.no_grad():
                x = torch.from_numpy(X).to(device)
                return (m(x) - x).cpu().numpy()

        mi = (res(Xtrain) ** 2).mean(0)
        beta = beta_kneewin(mi)
        l = -mi / beta; l -= l.max(); a = np.exp(l); a /= a.sum()

        rd, rh = res(Xd), res(Xht)
        a_auc = auc((a * rd ** 2).sum(1), (a * rh ** 2).sum(1))
        rows.append({"held_out_state": STATE_NAMES[held_out], "n_train": int((~val_mask).sum()),
                      "n_val": int(val_mask.sum()), "beta": beta, "auc": float(a_auc)})
        print(f"held out {STATE_NAMES[held_out]}: n_train={(~val_mask).sum()} "
              f"n_val={val_mask.sum()} beta={beta:.4f} AUC={a_auc:.4f}", flush=True)

    auc_v = np.array([r["auc"] for r in rows])
    beta_v = np.array([r["beta"] for r in rows])

    # random-split reference numbers from Table 5 / run_multiseed_eval.py (10 seeds)
    random_split_auc_mean, random_split_auc_std = 0.7947, 0.0136

    t_stat, p_val = stats.ttest_1samp(auc_v, random_split_auc_mean)

    report = {"rows": rows,
              "loso_auc_mean": float(auc_v.mean()), "loso_auc_std": float(auc_v.std(ddof=1)),
              "loso_beta_mean": float(beta_v.mean()), "loso_beta_std": float(beta_v.std(ddof=1)),
              "random_split_auc_mean": random_split_auc_mean, "random_split_auc_std": random_split_auc_std,
              "ttest_vs_random_split_mean_p": float(p_val)}
    (OUT / "leave_one_state_out.json").write_text(json.dumps(report, indent=2))

    print("\n=== SUMMARY ===")
    print(f"Leave-one-state-out (n=8 folds): AUC {auc_v.mean():.4f} +/- {auc_v.std(ddof=1):.4f}  "
          f"(min={auc_v.min():.4f}, max={auc_v.max():.4f})")
    print(f"Random 80/20 split (n=10 seeds, Table 5): AUC {random_split_auc_mean:.4f} +/- {random_split_auc_std:.4f}")
    print(f"one-sample t-test (LOSO folds vs. random-split mean): p={p_val:.3g}")
    print(f"\nsaved: {OUT/'leave_one_state_out.json'}")


if __name__ == "__main__":
    main()
