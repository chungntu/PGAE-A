# =============================================================================
# NEW FILE — response to Reviewer 1 #4 / Reviewer 2 #3 (RMAT-2026-0601, round 1)
#
# Standalone, isolated analysis. Reads ONLY the copied feature matrices in
# review vong 1/response/data/ (copies of code/features/X_*_new.npy). Does not
# import from, or write into, the original code/ tree.
#
# Reuses the exact AE architecture / training loop / label-free beta-selection
# rule from code/experiments/run_multiseed_eval.py (copied here verbatim, not
# imported, to keep this folder self-contained) so the reproduced baseline AUC
# is directly comparable to the published Table 5 numbers.
#
# What it does:
#   1. "Full" model (original protocol): AE trained on all 256 healthy
#      training-pool windows (states 1-8, incl. pier-settlement states 4-7),
#      10 seeds. Reports per-state mean attention-weighted anomaly score for
#      all 17 states (R1#4: "give the score for each of the 17 states"),
#      and the overall AUC as a sanity check against the published 0.795.
#   2. "Excl-settlement" model: identical protocol, but states 4-7 (pier
#      settlement, 20-95mm) are removed from the healthy training/validation
#      pool before the 80/20 split. 10 seeds. (R1#4 / R2#3: "repeat training
#      without the settlement states").
#   3. Compares AUC of (1) vs (2) on the *same* evaluation set (all damaged +
#      all 32 healthy test-pool windows, unchanged) so the comparison isolates
#      the effect of the training-pool composition only.
# =============================================================================
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from scipy import stats

_HERE = Path(__file__).resolve().parent            # response/code
DATA = _HERE.parent / "data"                        # response/data (copied matrices)
OUT = _HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

NH, BOTT, N_SEEDS, VAL = 8, 32, 10, 0.2
# 0-indexed state labels: state 1 -> 0, ..., state 17 -> 16.
# Pier-settlement states 4,5,6,7 (20/40/80/95 mm) -> labels 3,4,5,6.
SETTLEMENT_LABELS = {3, 4, 5, 6}
STATE_NAMES = {
    0: "S1 undamaged reference", 1: "S2 reference (settlement sys installed)",
    2: "S3 reference measurement", 3: "S4 pier settlement 20mm",
    4: "S5 pier settlement 40mm", 5: "S6 pier settlement 80mm",
    6: "S7 pier settlement 95mm", 7: "S8 pier lifted, new reference",
    8: "S9 concrete spalling 12m2", 9: "S10 concrete spalling 24m2",
    10: "S11 landslide at abutment", 11: "S12 concrete hinge failure",
    12: "S13 failure of 2 anchor heads", 13: "S14 failure of 4 anchor heads",
    14: "S15 rupture of 54 tendon wires", 15: "S16 rupture of 100 tendon wires",
    16: "S17 rupture of 154 tendon wires",
}


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


def run_condition(Xh_pool, Xd, Xht, Xall, yall, device, seeds, label):
    """Xh_pool: healthy training-pool rows eligible for the 80/20 AE split.
    Xd/Xht: fixed evaluation sets (unchanged across conditions).
    Xall/yall: all 612 windows, for the per-state score table."""
    aucs, per_state_runs = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(Xh_pool))
        nv = int(round(len(idx) * VAL))
        Xval, Xtrain = Xh_pool[idx[:nv]], Xh_pool[idx[nv:]]
        m = train_ae(Xtrain, Xval, device, seed)

        def res(X):
            with torch.no_grad():
                x = torch.from_numpy(X).to(device)
                return (m(x) - x).cpu().numpy()

        mi = (res(Xtrain) ** 2).mean(0)
        beta = beta_kneewin(mi)
        l = -mi / beta; l -= l.max(); a = np.exp(l); a /= a.sum()

        rd, rh = res(Xd), res(Xht)
        a_auc = auc((a * rd ** 2).sum(1), (a * rh ** 2).sum(1))
        aucs.append(a_auc)

        r_all = res(Xall)
        score_all = (a * r_all ** 2).sum(1)
        per_state_runs.append(score_all)
        print(f"  [{label}] seed {seed}: beta={beta:.4f} AUC={a_auc:.4f}", flush=True)

    per_state_runs = np.array(per_state_runs)  # (n_seeds, 612)
    mean_score_per_window = per_state_runs.mean(0)
    per_state_mean, per_state_std = {}, {}
    for s in range(17):
        vals = mean_score_per_window[yall == s]
        per_state_mean[s] = float(vals.mean()) if len(vals) else float("nan")
        per_state_std[s] = float(vals.std()) if len(vals) else float("nan")

    v = np.array(aucs)
    return {
        "label": label,
        "auc_mean": float(v.mean()), "auc_std": float(v.std(ddof=1)),
        "auc_runs": [float(x) for x in v],
        "n_train_pool_healthy": int(len(Xh_pool)),
        "per_state_mean_score": per_state_mean,
        "per_state_std_score": per_state_std,
    }


def main():
    Xtr = np.load(DATA / "X_train_new.npy").astype(np.float32)
    ytr = np.load(DATA / "labels_train_new.npy").astype(int)
    Xte = np.load(DATA / "X_test_new.npy").astype(np.float32)
    yte = np.load(DATA / "labels_test_new.npy").astype(int)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    Xall = np.vstack([Xtr, Xte])
    yall = np.concatenate([ytr, yte])
    print(f"total windows: {len(yall)}; per-state counts: "
          f"{ {int(s): int((yall==s).sum()) for s in range(17)} }")
    print(f"healthy test-pool per-state counts: "
          f"{ {int(s): int((yte==s).sum()) for s in range(NH)} }")

    Xh_all = Xtr[ytr < NH]                                # 256 healthy training-pool rows (all 8 states)
    Xd = np.concatenate([Xtr[ytr >= NH], Xte[yte >= NH]])  # 324 damaged (unchanged evaluation set)
    Xht = Xte[yte < NH]                                     # 32 healthy test-pool rows (unchanged evaluation set)

    settlement_mask = np.isin(ytr, list(SETTLEMENT_LABELS))
    Xh_excl = Xtr[(ytr < NH) & (~settlement_mask)]         # healthy training-pool rows, S4-S7 removed
    print(f"Xh_all (full)          : {Xh_all.shape}")
    print(f"Xh_excl (no settlement): {Xh_excl.shape}  "
          f"(removed {Xh_all.shape[0]-Xh_excl.shape[0]} settlement-state rows)")

    print("\n=== Condition 1/2: FULL (original protocol, S1-S8 all healthy) ===")
    full = run_condition(Xh_all, Xd, Xht, Xall, yall, device, range(N_SEEDS), "full")

    print("\n=== Condition 2/3: EXCL-SETTLEMENT (S4-S7 removed from healthy train pool) ===")
    excl = run_condition(Xh_excl, Xd, Xht, Xall, yall, device, range(N_SEEDS), "excl_settlement")

    # Control condition: removing S4-S7 both (a) drops settlement-state windows
    # AND (b) halves the healthy training-pool size (256 -> 128). To separate
    # "settlement states specifically matter" from "less training data hurts",
    # match a random subsample of Xh_all (all 8 states represented, just fewer
    # windows) to the SAME size (128) as Xh_excl, per seed.
    print("\n=== Condition 3/3: SIZE-MATCHED RANDOM SUBSAMPLE (all 8 states, N=128, controls for sample size) ===")
    n_match = Xh_excl.shape[0]
    aucs_m = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        sub_idx = rng.choice(len(Xh_all), size=n_match, replace=False)
        Xh_sub = Xh_all[sub_idx]
        r = run_condition(Xh_sub, Xd, Xht, Xall, yall, device, [seed], f"matched_seed{seed}")
        aucs_m.append(r["auc_runs"][0])
    v_m = np.array(aucs_m)
    matched = {
        "label": "matched_subsample", "auc_mean": float(v_m.mean()), "auc_std": float(v_m.std(ddof=1)),
        "auc_runs": [float(x) for x in v_m], "n_train_pool_healthy": n_match,
    }

    full_v = np.array(full["auc_runs"]); excl_v = np.array(excl["auc_runs"])
    t_full_excl = stats.ttest_ind(full_v, excl_v)
    t_matched_excl = stats.ttest_ind(v_m, excl_v)
    t_full_matched = stats.ttest_ind(full_v, v_m)

    report = {
        "settlement_labels_0idx": sorted(SETTLEMENT_LABELS),
        "state_names": STATE_NAMES,
        "full": full,
        "excl_settlement": excl,
        "matched_subsample": matched,
        "auc_diff_full_minus_excl": float(full_v.mean() - excl_v.mean()),
        "auc_diff_matched_minus_excl": float(v_m.mean() - excl_v.mean()),
        "welch_full_vs_excl_p": float(t_full_excl.pvalue),
        "welch_matched_vs_excl_p": float(t_matched_excl.pvalue),
        "welch_full_vs_matched_p": float(t_full_matched.pvalue),
    }
    (OUT / "settlement_sensitivity.json").write_text(json.dumps(report, indent=2))

    print("\n=== SUMMARY ===")
    print(f"FULL (N=256, incl. settlement)      : AUC {full['auc_mean']:.4f} +/- {full['auc_std']:.4f}")
    print(f"EXCL-SETTLEMENT (N=128)             : AUC {excl['auc_mean']:.4f} +/- {excl['auc_std']:.4f}")
    print(f"MATCHED RANDOM SUBSAMPLE (N=128)    : AUC {matched['auc_mean']:.4f} +/- {matched['auc_std']:.4f}")
    print(f"full - excl        diff={report['auc_diff_full_minus_excl']:+.4f}  p={t_full_excl.pvalue:.3g}  (settlement-removal + sample-size, confounded)")
    print(f"matched - excl     diff={report['auc_diff_matched_minus_excl']:+.4f}  p={t_matched_excl.pvalue:.3g}  (settlement-removal EFFECT ALONE, size held fixed)")
    print(f"full - matched     diff={float(full_v.mean()-v_m.mean()):+.4f}  p={t_full_matched.pvalue:.3g}  (sample-size EFFECT ALONE, settlement held in)")

    print("\nPer-state mean attention-weighted anomaly score (FULL-protocol model, avg over 10 seeds):")
    print(f"{'state':>3}  {'name':<38} {'mean_score':>11} {'std':>9} {'class':>8}")
    for s in range(17):
        cls = "healthy" if s < NH else "damaged"
        print(f"{s+1:3d}  {STATE_NAMES[s]:<38} {full['per_state_mean_score'][s]:11.4f} {full['per_state_std_score'][s]:9.4f} {cls:>8}")

    # CSV for easy inclusion in the manuscript / response letter
    import csv
    with open(OUT / "per_state_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["state", "name", "class", "full_mean_score", "full_std_score",
                    "excl_settlement_mean_score", "excl_settlement_std_score"])
        for s in range(17):
            cls = "healthy" if s < NH else "damaged"
            w.writerow([s + 1, STATE_NAMES[s], cls,
                        f"{full['per_state_mean_score'][s]:.6f}", f"{full['per_state_std_score'][s]:.6f}",
                        f"{excl['per_state_mean_score'][s]:.6f}", f"{excl['per_state_std_score'][s]:.6f}"])

    print(f"\nsaved: {OUT/'settlement_sensitivity.json'}")
    print(f"saved: {OUT/'per_state_scores.csv'}")


if __name__ == "__main__":
    main()
