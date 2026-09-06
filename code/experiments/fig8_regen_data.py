# =============================================================================
# NEW FILE — response to Reviewer 2 #12 (RMAT-2026-0601, round 1)
#
# Reproduces the EXACT seed-0 "representative run" computation used by
# code/experiments/run_figure_data.py to build code/figures/fig8.html's
# control.json (same SEED/NH/BOTT/VAL constants, same AE architecture, same
# 80/20 split, same normalization), so the numeric scores are bit-identical
# to the released figure. Adds one new field the original script does not
# compute: for every one of the 612 state-sorted windows, whether it belongs
# to the 256-window healthy TRAINING POOL or the 32-window healthy TEST POOL
# (never used in training at any seed) -- the distinction Reviewer 2 #12
# asks Fig. 8 to show. Damaged windows are always test-pool (never trained
# on), so this only adds information for the healthy region of the plot.
# =============================================================================
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

_HERE = Path(__file__).resolve().parent
DATA = _HERE.parent / "features"
OUT = _HERE.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NH, BOTT, SEED, VAL = 8, 32, 0, 0.2


class AE(nn.Module):
    def __init__(self, d=906, b=32):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, b), nn.ReLU())
        self.d = nn.Sequential(nn.Linear(b, 128), nn.ReLU(), nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, d))

    def forward(self, x):
        return self.d(self.e(x))


def train_ae(Xtrain, Xval, device):
    torch.manual_seed(SEED)
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


def main():
    Xtr = np.load(DATA / "X_train_new.npy").astype(np.float32)
    ytr = np.load(DATA / "labels_train_new.npy").astype(int)
    Xte = np.load(DATA / "X_test_new.npy").astype(np.float32)
    yte = np.load(DATA / "labels_test_new.npy").astype(int)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = Xtr.shape[1]

    Xh = Xtr[ytr < NH]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(Xh))
    nv = int(round(len(idx) * VAL))
    Xval, Xtrain = Xh[idx[:nv]], Xh[idx[nv:]]
    print("Training AE (seed 0, matching run_figure_data.py)...")
    m = train_ae(Xtrain, Xval, device)
    m.eval()

    with torch.no_grad():
        xt = torch.from_numpy(Xtrain).to(device)
        r_tr = (m(xt) - xt).cpu().numpy()
    mi = (r_tr ** 2).mean(0)

    betas = np.geomspace(1e-3, 10, 2000)
    t = np.log10(betas)
    prs = np.empty(len(betas))
    for i, b in enumerate(betas):
        loga = -mi / b; loga -= loga.max(); aa = np.exp(loga); aa /= aa.sum()
        prs[i] = 1.0 / (aa ** 2).sum()
    xn = (t - t.min()) / (t.max() - t.min())
    yn = (prs - prs.min()) / (prs.max() - prs.min())
    b_knee = betas[int(np.argmin(yn - xn))]
    b_infl = betas[int(np.argmax(np.gradient(prs, t)))]
    beta = float(np.sqrt(b_knee * b_infl))
    a = np.exp(-mi / beta - (-mi / beta).max()); a /= a.sum()
    print(f"beta*={beta:.4f}")

    Xall = np.vstack([Xtr, Xte])
    yall = np.concatenate([ytr, yte])
    is_test = np.zeros(len(yall), bool); is_test[len(ytr):] = True  # Xte rows = test pool

    with torch.no_grad():
        xa = torch.from_numpy(Xall).to(device)
        r_all = (m(xa) - xa).cpu().numpy()
    r2_all = r_all ** 2
    healthy = yall < NH

    scores_plain = r2_all.sum(1)
    scores_attn = (a[None, :] * r2_all).sum(1)

    def pct(arr, q):
        s = np.sort(arr); return float(s[int(len(s) * q)])

    order_state = np.argsort(yall, kind="stable")
    sp = scores_plain[order_state]; sa = scores_attn[order_state]
    ys = yall[order_state]; is_test_s = is_test[order_state]; healthy_s = healthy[order_state]
    damaged = ~healthy
    thr_sp = pct(scores_plain[healthy], 0.95); thr_sa = pct(scores_attn[healthy], 0.95)
    sp_n = sp / thr_sp; sa_n = sa / thr_sa
    bounds = [int(np.searchsorted(ys, k)) for k in range(1, 17)]
    n_h = int(healthy.sum())
    thr_d_sp = pct((scores_plain[damaged] / thr_sp), 0.95)
    thr_d_sa = pct((scores_attn[damaged] / thr_sa), 0.95)

    # is_train_pool: True only for healthy windows that are in the 256-window
    # training pool (Xtr & healthy & not test); damaged windows and healthy
    # test-pool windows are both "held out" from AE training.
    is_train_pool = (healthy_s & ~is_test_s).tolist()

    n_test_pool_healthy = int((healthy_s & is_test_s).sum())
    n_train_pool_healthy = int((healthy_s & ~is_test_s).sum())
    print(f"healthy windows: {n_train_pool_healthy} training-pool, {n_test_pool_healthy} test-pool (held out)")

    ctrl = {"sp": [round(v, 5) for v in sp_n.tolist()], "sa": [round(v, 5) for v in sa_n.tolist()],
            "bounds": bounds, "ymax": 28.0, "n_healthy": n_h, "thr_h": 1.0,
            "thr_d_sp": round(thr_d_sp, 4), "thr_d_sa": round(thr_d_sa, 4),
            "is_train_pool": is_train_pool}
    (OUT / "control_with_pool.json").write_text(json.dumps(ctrl))
    print(f"saved: {OUT / 'control_with_pool.json'}")


if __name__ == "__main__":
    main()
