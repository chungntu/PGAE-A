# =============================================================================
# NEW FILE — response to Reviewer 1 #5, Reviewer 1 #6, Reviewer 2 #15
# (RMAT-2026-0601, round 1)
#
# Reuses the exact AE architecture / training loop / one-class beta-selection
# rule from code/experiments/run_multiseed_eval.py (copied here verbatim, not
# imported) so results are directly comparable to the published Table 5.
#
# Adds five one-class alternatives to PGAE-A's softmax attention rule, all
# trained on the SAME 256-window healthy training pool and scored on the SAME
# evaluation set (324 damaged + 32 healthy test-pool windows), ten seeds each:
#
#   1. Variance-normalized residual   (R1#5: divide each residual by its
#      healthy reference scale -- the natural "simple scaling" alternative to
#      the softmax attention rule; scale = per-feature residual variance on
#      the healthy training split)
#   2. MAD-normalized residual        (R2#15: robust version of (1), scale =
#      per-feature residual median absolute deviation)
#   3. PCA squared prediction error   (R1#6: score damage by the RESIDUAL
#      (discarded) subspace after projecting onto the leading PCs, rather
#      than the leading-PC Mahalanobis distance already reported in Sec. 3.4)
#   4. One-Class SVM                  (R2#15)
#   5. Isolation Forest               (R2#15)
#
# AE-plain and PGAE-A (attention) are recomputed from the same trained model
# per seed as a reproduction check against the published 0.643 / 0.795.
# =============================================================================
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

_HERE = Path(__file__).resolve().parent            # code/experiments
DATA = _HERE.parent / "features"
OUT = _HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

NH, BOTT, N_SEEDS, VAL = 8, 32, 10, 0.2
PCA_K = 32  # matches the AE bottleneck width, for a like-for-like comparison


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

    Xh_all = Xtr[ytr < NH]                                  # 256 healthy training-pool rows
    Xd = np.concatenate([Xtr[ytr >= NH], Xte[yte >= NH]])    # 324 damaged (evaluation set, unchanged)
    Xht = Xte[yte < NH]                                       # 32 healthy test-pool rows (evaluation set, unchanged)
    print(f"Xh_all={Xh_all.shape} Xd={Xd.shape} Xht={Xht.shape}")

    methods = ["ae_plain", "pgae_a", "var_norm", "mad_norm", "pca_spe", "ocsvm", "iso_forest"]
    results = {m: [] for m in methods}

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(Xh_all))
        nv = int(round(len(idx) * VAL))
        Xval, Xtrain = Xh_all[idx[:nv]], Xh_all[idx[nv:]]

        # ---- AE-based methods (share one trained AE per seed) ----
        m = train_ae(Xtrain, Xval, device, seed)

        def res(X):
            with torch.no_grad():
                x = torch.from_numpy(X).to(device)
                return (m(x) - x).cpu().numpy()

        r_train, r_d, r_ht = res(Xtrain), res(Xd), res(Xht)

        # AE-plain
        results["ae_plain"].append(auc((r_d ** 2).sum(1), (r_ht ** 2).sum(1)))

        # PGAE-A (softmax attention over healthy-residual reliability)
        mi = (r_train ** 2).mean(0)
        beta = beta_kneewin(mi)
        l = -mi / beta; l -= l.max(); a = np.exp(l); a /= a.sum()
        results["pgae_a"].append(auc((a * r_d ** 2).sum(1), (a * r_ht ** 2).sum(1)))

        # Variance-normalized residual (R1#5)
        var_i = r_train.var(axis=0, ddof=1)
        var_i = np.maximum(var_i, 1e-12)
        results["var_norm"].append(auc(((r_d ** 2) / var_i).sum(1), ((r_ht ** 2) / var_i).sum(1)))

        # MAD-normalized residual (R2#15), scaled so MAD ~ std under normality
        med = np.median(r_train, axis=0)
        mad_i = np.median(np.abs(r_train - med), axis=0) * 1.4826
        mad_i = np.maximum(mad_i, 1e-12)
        results["mad_norm"].append(auc(((r_d / mad_i) ** 2).sum(1), ((r_ht / mad_i) ** 2).sum(1)))

        # ---- Non-AE methods, fit directly on the standardized 906-dim features ----
        pca = PCA(n_components=PCA_K, random_state=seed).fit(Xtrain)

        def spe(X):
            Xp = pca.inverse_transform(pca.transform(X))
            return ((X - Xp) ** 2).sum(1)

        results["pca_spe"].append(auc(spe(Xd), spe(Xht)))

        ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(Xtrain)
        results["ocsvm"].append(auc(-ocsvm.decision_function(Xd), -ocsvm.decision_function(Xht)))

        iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=seed).fit(Xtrain)
        results["iso_forest"].append(auc(-iso.score_samples(Xd), -iso.score_samples(Xht)))

        print(f"seed {seed}: " + " ".join(f"{k}={results[k][-1]:.3f}" for k in methods), flush=True)

    summary = {}
    for k in methods:
        v = np.array(results[k])
        summary[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "runs": [float(x) for x in v]}

    (OUT / "alt_baselines.json").write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY (10 seeds) ===")
    name_map = {
        "ae_plain": "AE-plain (reproduction)", "pgae_a": "PGAE-A (reproduction)",
        "var_norm": "Variance-normalized residual", "mad_norm": "MAD-normalized residual",
        "pca_spe": f"PCA squared prediction error (k={PCA_K})",
        "ocsvm": "One-Class SVM (RBF, nu=0.05)", "iso_forest": "Isolation Forest",
    }
    for k in methods:
        s = summary[k]
        print(f"{name_map[k]:38s}: AUC {s['mean']:.4f} +/- {s['std']:.4f}")
    print(f"\nsaved: {OUT/'alt_baselines.json'}")


if __name__ == "__main__":
    main()
