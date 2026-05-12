"""
PyImpetus-SHAP Pipeline — Step 2: Feature Selection
=====================================================
Runs four independent PyImpetus Markov Blanket experiments crossing
two significance thresholds (p=0.10, p=0.05) × two random seeds
(42, 123) to identify stable probe candidates.

Inputs  : ADNI_Gene_Expression_Final_96_clean.csv
Outputs : selected_features_p{thresh}_seed{seed}.json  (4 files)
          feature_stability_report.txt

Authors : Asif Hassan Syed et al., King Abdulaziz University
License : MIT
"""

import pandas as pd
import numpy as np
import json
import warnings
from collections import Counter
from itertools import combinations
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from PyImpetus import PPIMBR

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────
DATA_FILE    = "ADNI_Gene_Expression_Final_96_clean.csv"
TARGET       = "Delta_MMSE"
EXCLUDE_COLS = [
    "RID", "VISCODE2", "Phase", "SubjectID", TARGET,
    "MMSE_current", "MMSE_12m_future", "future_months_12m",
]
N_OUTER_FOLDS  = 5
VAR_THRESHOLD  = 0.01
K_BEST         = 2000
PY_NUM_SIMUL   = 5
PY_SIMUL_SIZE  = 0.2

EXPERIMENTS = [
    {"seed": 42,  "p_thresh": 0.10},
    {"seed": 42,  "p_thresh": 0.05},
    {"seed": 123, "p_thresh": 0.10},
    {"seed": 123, "p_thresh": 0.05},
]

# ── Load data ─────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_FILE)
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
X = df[feature_cols]
y = df[TARGET].values

# Discretise target for stratified CV
y_binned = pd.qcut(y, q=4, labels=False, duplicates="drop")
print(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features")

# ── Run all four experiments ──────────────────────────────────────
all_results = {}

for exp in EXPERIMENTS:
    seed     = exp["seed"]
    p_thresh = exp["p_thresh"]
    tag      = f"p{p_thresh}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"Experiment: {tag}")
    print(f"{'='*60}")

    np.random.seed(seed)
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=seed
    )
    selected_per_fold = []
    fallback_folds    = []

    for fold, (train_idx, _) in enumerate(outer_cv.split(X, y_binned)):
        print(f"  Fold {fold+1}/{N_OUTER_FOLDS} …", end=" ")
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        original_cols = X_train.columns.tolist()

        # Stage 1: Variance filtering
        vt = VarianceThreshold(threshold=VAR_THRESHOLD)
        X_lv = vt.fit_transform(X_train)
        lv_names = [original_cols[i]
                    for i, k in enumerate(vt.get_support()) if k]

        # Stage 2: SelectKBest
        k_actual = min(K_BEST, X_lv.shape[1])
        skb = SelectKBest(f_regression, k=k_actual)
        X_kb = skb.fit_transform(X_lv, y_train)
        kb_names = [lv_names[i]
                    for i, k in enumerate(skb.get_support()) if k]

        # Stage 3: PyImpetus Markov Blanket
        try:
            X_df = pd.DataFrame(X_kb, columns=kb_names)
            fs = PPIMBR(
                model=DecisionTreeRegressor(random_state=seed),
                p_val_thresh=p_thresh,
                num_simul=PY_NUM_SIMUL,
                simul_size=PY_SIMUL_SIZE,
                sig_test_type="non-parametric",
                cv=0,
                random_state=seed,
                n_jobs=-1,
                verbose=0,
            )
            fs.fit_transform(X_df, y_train)
            selected = fs.MB
            n_sel = len(selected)
        except Exception as e:
            print(f"[PyImpetus error: {e}]", end=" ")
            selected = []
            n_sel = 0

        if n_sel == 0:
            print(f"fallback (0 selected → using top-{k_actual})")
            selected = kb_names
            fallback_folds.append(fold + 1)
        else:
            print(f"{n_sel} probes selected")

        selected_per_fold.append(selected)

    # Save JSON
    json_file = f"selected_features_{tag}.json"
    with open(json_file, "w") as f:
        json.dump(selected_per_fold, f, indent=2)
    print(f"  → Saved: {json_file}")
    print(f"  Fallback folds: {fallback_folds if fallback_folds else 'None'}")
    all_results[tag] = {
        "folds": selected_per_fold,
        "fallback_folds": fallback_folds,
    }

# ── Stability analysis ────────────────────────────────────────────
print("\n" + "="*60)
print("STABILITY ANALYSIS")
print("="*60)

report_lines = ["PyImpetus Feature Selection — Stability Report\n"]


def analyze_stability(folds, fallback_folds, tag):
    """Compute selection frequency and Jaccard similarity (non-fallback folds)."""
    non_fb = [f for i, f in enumerate(folds)
              if (i + 1) not in fallback_folds]
    if not non_fb:
        return set(), []

    freq    = Counter(feat for fold in non_fb for feat in fold)
    union   = set(freq.keys())
    jaccards = []
    for i in range(len(non_fb) - 1):
        s1, s2 = set(non_fb[i]), set(non_fb[i + 1])
        if s1 or s2:
            jaccards.append(len(s1 & s2) / len(s1 | s2))

    line = (f"{tag}: {len(union)} unique probes in non-fallback folds | "
            f"mean Jaccard = {np.mean(jaccards):.3f} "
            f"(n_jaccards={len(jaccards)})")
    print(f"  {line}")
    report_lines.append(line)
    return union, jaccards


unions = {}
for tag, data in all_results.items():
    u, j = analyze_stability(data["folds"], data["fallback_folds"], tag)
    unions[tag] = u

# Cross-threshold intersection
print("\nCross-threshold intersections:")
report_lines.append("\nCross-threshold intersections:")
p010_union = unions.get("p0.1_seed42", set()) | unions.get("p0.1_seed123", set())
p005_union = unions.get("p0.05_seed42", set()) | unions.get("p0.05_seed123", set())

cross_intersect = p010_union & p005_union
j_cross = (len(cross_intersect) / len(p010_union | p005_union)
           if (p010_union | p005_union) else 0)

line1 = f"p=0.1 union: {len(p010_union)} probes"
line2 = f"p=0.05 union: {len(p005_union)} probes"
line3 = (f"Cross-threshold intersection: {len(cross_intersect)} probes | "
         f"Jaccard = {j_cross:.3f}")
line4 = f"Final core probes: {sorted(cross_intersect)}"

for line in [line1, line2, line3, line4]:
    print(f"  {line}")
    report_lines.append(line)

# Save core probe order for downstream use
core_probes = sorted(cross_intersect)
with open("six_gene_order.json", "w") as f:
    json.dump(core_probes, f, indent=2)
print(f"\n  → six_gene_order.json saved ({len(core_probes)} probes)")

# Save stability report
with open("feature_stability_report.txt", "w") as f:
    f.write("\n".join(report_lines))
print("  → feature_stability_report.txt saved")
print("\n✓ Feature selection complete.")
