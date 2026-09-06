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
z24_verify_top25_by_class.py
NEW FILE — splits train/test by healthy vs damaged label and checks the
top-25 attention-weight columns *within each class separately*, to test
whether they carry ANY healthy-vs-damaged discriminative signal at all.
"""
import numpy as np
from pathlib import Path

FEAT = Path(__file__).resolve().parent.parent / "features"   # code/features
NH = 8  # states 1-8 healthy (0-indexed state < NH)

TOP25 = [9, 15, 24, 25, 30, 38, 57, 63, 73, 87, 105, 109, 126, 135,
         152, 157, 160, 201, 206, 207, 212, 216, 217, 230, 231]

Xtr = np.load(FEAT / "X_train_new.npy")
ytr = np.load(FEAT / "labels_train_new.npy").astype(int)
Xte = np.load(FEAT / "X_test_new.npy")
yte = np.load(FEAT / "labels_test_new.npy").astype(int)

print(f"train labels present: {sorted(set(ytr.tolist()))}, n={len(ytr)}")
print(f"test  labels present: {sorted(set(yte.tolist()))}, n={len(yte)}")

Xall = np.vstack([Xtr, Xte])
yall = np.concatenate([ytr, yte])
healthy = yall < NH
damaged = ~healthy
print(f"\ncombined: {healthy.sum()} healthy windows, {damaged.sum()} damaged windows")

print(f"\n{'idx':>4} {'healthy_std':>12} {'healthy_mean':>13} {'damaged_std':>12} {'damaged_mean':>13} {'mean_diff':>11}")
any_diff = False
for i in TOP25:
    h = Xall[healthy, i]
    d = Xall[damaged, i]
    diff = abs(h.mean() - d.mean())
    if diff > 1e-9 or h.std() > 1e-9 or d.std() > 1e-9:
        any_diff = True
    print(f"{i:4d} {h.std():12.3e} {h.mean():13.3e} {d.std():12.3e} {d.mean():13.3e} {diff:11.3e}")

print(f"\nAny healthy-vs-damaged difference detected in top-25: {any_diff}")
