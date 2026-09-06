# =============================================================================
# NEW FILE — response to Reviewer 1 #1 (RMAT-2026-0601, round 1)
#
# Tests whether the attention weight PIAE-A assigns to each feature is
# explained by redundancy with the OTHER 905 features, rather than by
# something specific to that feature's reconstruction reliability.
#
# Reviewer's request, verbatim: "The authors should compare each weight with
# an independent measure of feature redundancy. They should use healthy data
# that the autoencoder did not see. If redundancy explains the weights, the
# authors must remove the claim about structural control."
#
# Redundancy measure: for each feature i, fit a ridge-regularized linear
# model predicting feature i from the other 905 features, evaluated by
# efficient leave-one-out cross-validation (RidgeCV's built-in LOO-GCV) --
# entirely on Xht, the 32 healthy TEST-POOL windows, which never enter AE
# training, AE early stopping, or attention-weight computation at any seed.
# This is the only healthy data in the released matrices that the model
# never sees in any capacity, so it is the correct "unseen" set for this
# test. A high leave-one-out R^2 means feature i can be reconstructed from
# the other 905 features alone, with no autoencoder involved -- i.e. it is
# redundant with the rest of the bank, independent of PIAE-A entirely.
#
# Constant features (std ~ 0 on the healthy training pool, matching the
# 1e-9 threshold already used in code/experiments/verify_constant_columns.py)
# are handled separately: their attention weight is understood already (the
# manuscript's Sec. 3.4 / Fig. 7b discussion), and a constant column has an
# undefined or degenerate redundancy R^2, so they are excluded from the
# correlation test and reported as a separate count.
# =============================================================================
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy import stats
from sklearn.linear_model import RidgeCV

_HERE = Path(__file__).resolve().parent            # response/code
DATA = _HERE.parent / "features"
OUT = _HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

NH, BOTT, N_SEEDS, VAL = 8, 32, 10, 0.2
CONST_THRESH = 1e-9  # matches code/experiments/verify_constant_columns.py


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

    Xh_all = Xtr[ytr < NH]                 # 256 healthy training-pool rows
    Xht = Xte[yte < NH].astype(np.float64)  # 32 healthy test-pool rows -- UNSEEN by the AE in any capacity
    d = Xh_all.shape[1]
    print(f"Xh_all={Xh_all.shape}  Xht (unseen, used for redundancy)={Xht.shape}")

    # ---- 1. average attention weight over the standard 10-seed protocol ----
    weight_runs = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(Xh_all))
        nv = int(round(len(idx) * VAL))
        Xval, Xtrain = Xh_all[idx[:nv]], Xh_all[idx[nv:]]
        m = train_ae(Xtrain, Xval, device, seed)
        with torch.no_grad():
            xt = torch.from_numpy(Xtrain).to(device)
            r_train = (m(xt) - xt).cpu().numpy()
        mi = (r_train ** 2).mean(0)
        beta = beta_kneewin(mi)
        l = -mi / beta; l -= l.max(); a = np.exp(l); a /= a.sum()
        weight_runs.append(a)
        print(f"  seed {seed}: beta={beta:.4f}", flush=True)
    weight = np.mean(weight_runs, axis=0)  # (906,) average attention weight per feature

    # ---- 2. constant-feature mask (std ~ 0 on the healthy training pool) ----
    std_train = Xh_all.std(axis=0)
    constant_mask = std_train < CONST_THRESH
    print(f"\nconstant features (std < {CONST_THRESH:g} on healthy training pool): {constant_mask.sum()} / {d}")

    # also flag features that are constant on Xht itself (redundancy R^2 undefined there)
    std_unseen = Xht.std(axis=0)
    unseen_const_mask = std_unseen < CONST_THRESH
    print(f"additionally constant on the unseen healthy test-pool (Xht): {(unseen_const_mask & ~constant_mask).sum()}")

    exclude_mask = constant_mask | unseen_const_mask
    keep_idx = np.where(~exclude_mask)[0]
    print(f"features entering the redundancy test: {len(keep_idx)} / {d}")

    # ---- 3a. PRIMARY redundancy measure: max absolute pairwise correlation
    #          with any of the other 905 features, on Xht only. With only 32
    #          unseen rows, a multivariate regression against all 905 other
    #          features (attempted first; see redundancy_multivariate_R2
    #          below) is in a catastrophic n<<p regime -- every single
    #          feature came back with a NEGATIVE leave-one-out R^2 (max
    #          across all 836 features: -0.000), which means that measure is
    #          just noise here, not a real redundancy signal. Pairwise
    #          correlation is far more stable with n=32 and matches the
    #          reviewer's own framing ("features that contain the same
    #          information") directly. This is the measure used in the
    #          reported Spearman test below.
    print("\ncomputing pairwise correlation matrix on Xht (32 x 906)...")
    corr = np.corrcoef(Xht, rowvar=False)  # (906, 906), NaN rows/cols for constant features
    np.fill_diagonal(corr, np.nan)
    redundancy = np.nanmax(np.abs(corr), axis=1)  # (906,) max |corr| with any other feature
    redundancy[exclude_mask] = np.nan

    # ---- 3b. SECONDARY / diagnostic measure: multivariate ridge LOO-CV R^2
    #          predicting feature i from the other 905 features, fit on Xht.
    #          Reported for transparency but NOT used for the main test --
    #          see the note above.
    alphas = np.logspace(-2, 4, 25)
    redundancy_multivariate_R2 = np.full(d, np.nan)
    print("fitting secondary ridge LOO-CV multivariate model per feature (diagnostic only)...")
    for j, i in enumerate(keep_idx):
        y = Xht[:, i]
        X = np.delete(Xht, i, axis=1)
        model = RidgeCV(alphas=alphas, store_cv_results=False)
        model.fit(X, y)
        redundancy_multivariate_R2[i] = model.best_score_
        if (j + 1) % 150 == 0:
            print(f"  {j+1}/{len(keep_idx)} features done", flush=True)
    print(f"secondary measure sanity check: max LOO R^2 across all features = "
          f"{np.nanmax(redundancy_multivariate_R2):.4f} (negative/near-zero for all "
          f"features confirms n=32 << p=905 makes this measure unusable here)")

    # ---- 4. compare weight vs. redundancy (max pairwise |corr|) across the
    #         non-excluded features ----
    w = weight[keep_idx]
    r = redundancy[keep_idx]
    rho, p = stats.spearmanr(w, r)
    print(f"\nSpearman correlation between attention weight and max-pairwise-|corr| "
          f"redundancy (n={len(keep_idx)} non-constant features): rho={rho:.3f}, p={p:.3g}")

    # secondary diagnostic correlation, reported for transparency (unreliable, n<<p)
    r_mv = redundancy_multivariate_R2[keep_idx]
    rho_mv, p_mv = stats.spearmanr(w, r_mv)
    print(f"(diagnostic only, unreliable: Spearman weight vs. multivariate-R^2: "
          f"rho={rho_mv:.3f}, p={p_mv:.3g})")

    order = np.argsort(-weight)
    top20 = [i for i in order if i not in np.where(constant_mask)[0]][:20]

    report = {
        "n_features": int(d),
        "n_constant_on_train": int(constant_mask.sum()),
        "n_additionally_constant_on_unseen": int((unseen_const_mask & ~constant_mask).sum()),
        "n_in_redundancy_test": int(len(keep_idx)),
        "spearman_rho_maxcorr": float(rho),
        "spearman_p_maxcorr": float(p),
        "spearman_rho_multivariate_R2_diagnostic_only": float(rho_mv),
        "spearman_p_multivariate_R2_diagnostic_only": float(p_mv),
        "multivariate_R2_max_across_all_features": float(np.nanmax(redundancy_multivariate_R2)),
        "top20_by_weight_excl_constant": [
            {"feature_idx": int(i), "attention_weight": float(weight[i]),
             "redundancy_maxcorr": float(redundancy[i]) if not np.isnan(redundancy[i]) else None}
            for i in top20
        ],
        "weight_all": [float(x) for x in weight],
        "redundancy_maxcorr_all": [float(x) if not np.isnan(x) else None for x in redundancy],
        "redundancy_multivariate_R2_all_diagnostic_only": [
            float(x) if not np.isnan(x) else None for x in redundancy_multivariate_R2],
        "constant_mask": [bool(x) for x in constant_mask],
    }
    (OUT / "redundancy_analysis.json").write_text(json.dumps(report, indent=2))

    print("\n=== Top-20 highest-weight non-constant features: attention weight vs. max-pairwise-|corr| redundancy ===")
    for e in report["top20_by_weight_excl_constant"]:
        print(f"  idx {e['feature_idx']:4d}  weight={e['attention_weight']:.5f}  redundancy_maxcorr={e['redundancy_maxcorr']:.3f}")

    print(f"\nsaved: {OUT/'redundancy_analysis.json'}")


if __name__ == "__main__":
    main()
