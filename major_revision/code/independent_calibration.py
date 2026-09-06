# =============================================================================
# NEW FILE — response to Reviewer 2 #13 (RMAT-2026-0601, round 1)
#
# Standalone, isolated analysis. Reads ONLY the copied feature matrices in
# review vong 1/response/data/ (copies of code/features/X_*_new.npy). Does not
# import from, or write into, the original code/ tree.
#
# Reviewer's request, verbatim: "Attention weights are estimated from the
# same healthy observations used to train the autoencoder. [...] It would be
# useful to evaluate whether weights estimated from an independent healthy
# calibration subset lead to similar results."
#
# Standard protocol: within each seed's 80/20 healthy split, the residual
# statistics that set beta and the attention weights (mi) are computed from
# Xtrain -- the SAME rows used to fit the autoencoder's parameters by
# backprop. This script adds a second condition, identical in every other
# respect, where mi is instead computed from Xval -- the held-out 20% of the
# healthy training pool, never used for a gradient update, only for early
# stopping. Xval is disjoint from the evaluation set (Xd + Xht), so the AUC
# comparison below is not circular. Both conditions share the same trained
# AE per seed and are scored on the same evaluation set as Table 5.
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

NH, BOTT, N_SEEDS, VAL = 8, 32, 10, 0.2


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


def weights_from(mi):
    beta = beta_kneewin(mi)
    l = -mi / beta; l -= l.max(); a = np.exp(l); a /= a.sum()
    return a, beta


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
    print(f"Xh_all={Xh_all.shape}  Xd={Xd.shape}  Xht={Xht.shape}")

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

        r_train, r_val, r_d, r_ht = res(Xtrain), res(Xval), res(Xd), res(Xht)

        mi_self = (r_train ** 2).mean(0)
        a_self, beta_self = weights_from(mi_self)
        auc_self = auc((a_self * r_d ** 2).sum(1), (a_self * r_ht ** 2).sum(1))

        mi_indep = (r_val ** 2).mean(0)
        a_indep, beta_indep = weights_from(mi_indep)
        auc_indep = auc((a_indep * r_d ** 2).sum(1), (a_indep * r_ht ** 2).sum(1))

        # agreement between the two weight vectors themselves
        weight_corr = float(np.corrcoef(a_self, a_indep)[0, 1])

        rows.append({"seed": seed, "n_train": len(Xtrain), "n_val": len(Xval),
                      "beta_self": beta_self, "beta_indep": beta_indep,
                      "auc_self": float(auc_self), "auc_indep": float(auc_indep),
                      "weight_corr": weight_corr})
        print(f"seed {seed}: beta_self={beta_self:.4f} beta_indep={beta_indep:.4f} "
              f"AUC_self={auc_self:.4f} AUC_indep={auc_indep:.4f} weight_corr={weight_corr:.3f}", flush=True)

    def summ(k):
        v = np.array([r[k] for r in rows])
        return float(v.mean()), float(v.std(ddof=1)), v

    bsm, bss, bsv = summ("beta_self"); bim, bis, biv = summ("beta_indep")
    asm, ass_, asv = summ("auc_self"); aim, ais, aiv = summ("auc_indep")
    wcm, wcs, wcv = summ("weight_corr")

    p_auc = stats.ttest_rel(asv, aiv).pvalue  # paired: same seed, same eval set, same trained AE

    report = {"rows": rows,
              "summary": {"beta_self": {"mean": bsm, "std": bss}, "beta_indep": {"mean": bim, "std": bis},
                          "auc_self": {"mean": asm, "std": ass_}, "auc_indep": {"mean": aim, "std": ais},
                          "weight_corr": {"mean": wcm, "std": wcs},
                          "paired_auc_diff_mean": float((asv - aiv).mean()),
                          "paired_auc_diff_std": float((asv - aiv).std(ddof=1)),
                          "paired_t_p": float(p_auc)}}
    (OUT / "independent_calibration.json").write_text(json.dumps(report, indent=2))

    print("\n=== SUMMARY ===")
    print(f"Standard  (weights from Xtrain, n~{rows[0]['n_train']}): "
          f"beta={bsm:.4f}+/-{bss:.4f}  AUC={asm:.4f}+/-{ass_:.4f}")
    print(f"Independent (weights from Xval, n~{rows[0]['n_val']})  : "
          f"beta={bim:.4f}+/-{bis:.4f}  AUC={aim:.4f}+/-{ais:.4f}")
    print(f"weight vector correlation (self vs independent): {wcm:.3f}+/-{wcs:.3f}")
    print(f"paired AUC diff (self-indep): {(asv-aiv).mean():+.4f}+/-{(asv-aiv).std(ddof=1):.4f}  "
          f"paired t-test p={p_auc:.3g}")
    print(f"\nsaved: {OUT/'independent_calibration.json'}")


if __name__ == "__main__":
    main()
