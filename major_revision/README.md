# Major-revision materials

This folder contains everything added **after the original submission**, in
response to the two reviewers' comments on manuscript RMAT-2026-0601. It is
kept separate from `code/` (which reproduces the original submission's
results) so it is always clear which script, result, or figure belongs to
which round of the paper.

```text
major_revision/
|-- code/       # one script per new analysis requested by the reviewers
|-- data/       # copies of the feature matrices these scripts read
|-- results/    # JSON/CSV outputs backing the numbers in the response letter
`-- figures/    # figures regenerated at higher resolution during revision
```

Each script in `code/` is referenced directly from the corresponding comment
in the point-by-point response to reviewers and from Supplementary Material
Sections S1-S5. For example:

- `redundancy_analysis.py` -> independent redundancy score vs. attention weight
- `settlement_sensitivity.py` -> sensitivity to the healthy-state definition
- `constant_feature_removal.py` -> effect of removing constant features
- `independent_calibration.py` -> attention weights from a held-out calibration set
- `leave_one_state_out.py` -> leave-one-state-out validation
- `alt_baselines.py` -> One-Class SVM, Isolation Forest, PCA-SPE, variance-/MAD-normalized baselines
- `operational_metrics.py` -> sensitivity, specificity, and balanced accuracy at the alarm threshold
- `block_bootstrap_ci.py` -> cluster-bootstrap confidence interval for AUC
- `fig8_regen_data.py` / `build_fig8.py` -> regenerated Figure 8 (train vs. held-out shading)
