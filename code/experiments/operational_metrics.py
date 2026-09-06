# =============================================================================
# NEW FILE — response to Reviewer 2 #5 (RMAT-2026-0601, round 1)
#
# Reviewer's request: complement ROC-AUC with operational metrics (sensitivity,
# specificity/false-alarm rate, balanced accuracy) at the 95th-percentile
# healthy alarm threshold already used in the main text (Sec. 3.3).
#
# Reuses the standard ten-seed protocol (same as Table 5 / run_multiseed_eval.py):
# the alarm threshold is the 95th percentile of the validation-normal healthy
# scores, computed per seed from healthy data alone, exactly as in the main
# text's alarm rule.
# =============================================================================
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

_HERE = Path(__file__).resolve().parent
DATA = _HERE.parent / "features"
OUT = _HERE / "results"
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

    Xh_all = Xtr[ytr < NH]
    Xd = np.concatenate([Xtr[ytr >= NH], Xte[yte >= NH]])
    Xht = Xte[yte < NH]

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

        r_train = res(Xtrain)
        mi = (r_train ** 2).mean(0)
        beta = beta_kneewin(mi)
        l = -mi / beta; l -= l.max(); a = np.exp(l); a /= a.sum()

        # alarm threshold: 95th percentile of validation-normal (Xval) scores,
        # exactly as in the main text's alarm rule (Sec. 3.3)
        s_val = (a * res(Xval) ** 2).sum(1)
        thr = float(np.percentile(s_val, 95))

        s_d = (a * res(Xd) ** 2).sum(1)
        s_ht = (a * res(Xht) ** 2).sum(1)

        tp = int((s_d > thr).sum()); fn = int((s_d <= thr).sum())
        tn = int((s_ht <= thr).sum()); fp = int((s_ht > thr).sum())
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        false_alarm_rate = 1 - specificity
        balanced_acc = 0.5 * (sensitivity + specificity)

        rows.append({"seed": seed, "threshold": thr, "sensitivity": sensitivity,
                      "specificity": specificity, "false_alarm_rate": false_alarm_rate,
                      "balanced_accuracy": balanced_acc, "tp": tp, "fn": fn, "tn": tn, "fp": fp})
        print(f"seed {seed}: sens={sensitivity:.3f} spec={specificity:.3f} "
              f"FAR={false_alarm_rate:.3f} bal_acc={balanced_acc:.3f}", flush=True)

    def summ(k):
        v = np.array([r[k] for r in rows])
        return float(v.mean()), float(v.std(ddof=1))

    summary = {k: dict(zip(("mean", "std"), summ(k)))
               for k in ["sensitivity", "specificity", "false_alarm_rate", "balanced_accuracy"]}
    report = {"rows": rows, "summary": summary}
    (OUT / "operational_metrics.json").write_text(json.dumps(report, indent=2))

    print("\n=== SUMMARY (10 seeds, threshold = 95th-pct of validation-normal scores) ===")
    for k, v in summary.items():
        print(f"{k:20s}: {v['mean']:.3f} +/- {v['std']:.3f}")
    print(f"\nsaved: {OUT/'operational_metrics.json'}")


if __name__ == "__main__":
    main()
