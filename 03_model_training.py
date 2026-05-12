"""
PyImpetus-SHAP Pipeline — Step 3: Model Training & Evaluation
==============================================================
Trains the six-probe Elastic Net pipeline on the full dataset,
performs LOOCV, 5-fold CV, and subgroup analysis (CN / MCI / AD).
Saves the fitted pipeline as six_gene_model.pkl.

Inputs  : ADNI_Gene_Expression_Final_96_clean.csv, six_gene_order.json
Outputs : six_gene_model.pkl
          loocv_results.csv
          cv_results.csv
          subgroup_results.csv

Authors : Asif Hassan Syed et al., King Abdulaziz University
License : MIT
"""

import pandas as pd
import numpy as np
import json
import joblib
import warnings
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, LeaveOneOut, GridSearchCV
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────
DATA_FILE    = "ADNI_Gene_Expression_Final_96_clean.csv"
GENE_FILE    = "six_gene_order.json"
TARGET       = "Delta_MMSE"
EXCLUDE_COLS = [
    "RID", "VISCODE2", "Phase", "SubjectID", TARGET,
    "MMSE_current", "MMSE_12m_future", "future_months_12m",
]
RANDOM_STATE = 42
ALPHA        = 0.01
L1_RATIO     = 0.5
MAX_ITER     = 10_000

# ── Load data ─────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_FILE)
with open(GENE_FILE) as f:
    six_genes = json.load(f)

# Verify all six probes are present
missing = [g for g in six_genes if g not in df.columns]
if missing:
    raise ValueError(f"Missing probes in dataset: {missing}")

X6 = df[six_genes].values
y  = df[TARGET].values
n  = len(y)
print(f"Dataset: {n} samples × {len(six_genes)} probes")
print(f"Probes : {six_genes}")

# ── Helper: build pipeline ─────────────────────────────────────────
def make_pipeline(alpha=ALPHA, l1_ratio=L1_RATIO):
    return Pipeline([
        ("standardscaler", StandardScaler()),
        ("elasticnet", ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio,
            max_iter=MAX_ITER, random_state=RANDOM_STATE,
        )),
    ])

# ── 1. Train final model on full dataset ──────────────────────────
print("\n[1/4] Training final model on full dataset …")
final_model = make_pipeline()
final_model.fit(X6, y)
joblib.dump(final_model, "six_gene_model.pkl")
print("  → six_gene_model.pkl saved")

coef = final_model.named_steps["elasticnet"].coef_
print("\n  Elastic Net coefficients:")
for gene, c in zip(six_genes, coef):
    direction = "protective" if c > 0 else "harmful"
    print(f"    {gene:22s}  {c:+.4f}  ({direction})")

# ── 2. Leave-one-out cross-validation (LOOCV) ─────────────────────
print("\n[2/4] Running LOOCV …")
loo = LeaveOneOut()
y_true_loo, y_pred_loo = [], []

for i, (train_idx, test_idx) in enumerate(loo.split(X6)):
    print(f"  Iteration {i+1:3d}/{n}", end="\r")
    pipe = make_pipeline()
    pipe.fit(X6[train_idx], y[train_idx])
    y_pred_loo.append(pipe.predict(X6[test_idx])[0])
    y_true_loo.append(y[test_idx][0])

print()
y_true_loo = np.array(y_true_loo)
y_pred_loo = np.array(y_pred_loo)
errors_loo  = y_true_loo - y_pred_loo

loocv_mae  = mean_absolute_error(y_true_loo, y_pred_loo)
loocv_rmse = np.sqrt(mean_squared_error(y_true_loo, y_pred_loo))
loocv_r2   = r2_score(y_true_loo, y_pred_loo)

print(f"\n  LOOCV Results:")
print(f"    MAE  = {loocv_mae:.3f}")
print(f"    RMSE = {loocv_rmse:.3f}")
print(f"    R²   = {loocv_r2:.3f}")

# Save LOOCV predictions
loocv_df = pd.DataFrame({
    "y_true":     y_true_loo,
    "y_pred":     y_pred_loo,
    "error":      errors_loo,
    "abs_error":  np.abs(errors_loo),
})
loocv_df.to_csv("loocv_results.csv", index=False)
print("  → loocv_results.csv saved")

# ── 3. Stratified 5-fold CV ────────────────────────────────────────
print("\n[3/4] Running 5-fold stratified cross-validation …")
y_binned = pd.qcut(y, q=4, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_mae, cv_rmse, cv_r2 = [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X6, y_binned)):
    pipe = make_pipeline()
    pipe.fit(X6[train_idx], y[train_idx])
    preds = pipe.predict(X6[test_idx])
    cv_mae.append(mean_absolute_error(y[test_idx], preds))
    cv_rmse.append(np.sqrt(mean_squared_error(y[test_idx], preds)))
    cv_r2.append(r2_score(y[test_idx], preds))
    print(f"  Fold {fold+1}: MAE={cv_mae[-1]:.3f}  "
          f"RMSE={cv_rmse[-1]:.3f}  R²={cv_r2[-1]:.3f}")

print(f"\n  5-fold CV Summary:")
print(f"    MAE  = {np.mean(cv_mae):.3f} ± {np.std(cv_mae):.3f}")
print(f"    RMSE = {np.mean(cv_rmse):.3f} ± {np.std(cv_rmse):.3f}")
print(f"    R²   = {np.mean(cv_r2):.3f} ± {np.std(cv_r2):.3f}")

cv_results = pd.DataFrame({
    "fold":  range(1, 6),
    "MAE":   cv_mae,
    "RMSE":  cv_rmse,
    "R2":    cv_r2,
})
cv_results.to_csv("cv_results.csv", index=False)
print("  → cv_results.csv saved")

# ── 4. Subgroup analysis (CN / MCI / AD) ──────────────────────────
print("\n[4/4] Running subgroup analysis …")

# Reconstruct diagnosis from one-hot columns
if all(c in df.columns for c in ["DX_1.0", "DX_2.0", "DX_3.0"]):
    dx_raw = np.argmax(df[["DX_1.0", "DX_2.0", "DX_3.0"]].values, axis=1) + 1
elif "DIAGNOSIS" in df.columns:
    dx_raw = df["DIAGNOSIS"].values
else:
    raise ValueError("No diagnosis column found.")

GROUP_MAP = {1: "CN", 2: "MCI", 3: "AD"}
subgroup_records = []

for g_code, g_name in GROUP_MAP.items():
    mask = dx_raw == g_code
    Xg, yg = X6[mask], y[mask]
    ng = len(yg)
    print(f"\n  {g_name} (n={ng})")

    if ng < 3:
        print("    Too few samples — skipped.")
        continue

    cv_g = (StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            if ng >= 10 else LeaveOneOut())

    mae_g, rmse_g = [], []
    for tr, te in cv_g.split(Xg, yg):
        pipe = make_pipeline()
        pipe.fit(Xg[tr], yg[tr])
        p = pipe.predict(Xg[te])
        mae_g.append(mean_absolute_error(yg[te], p))
        rmse_g.append(np.sqrt(mean_squared_error(yg[te], p)))

    print(f"    MAE  = {np.mean(mae_g):.3f} ± {np.std(mae_g):.3f}")
    print(f"    RMSE = {np.mean(rmse_g):.3f} ± {np.std(rmse_g):.3f}")

    subgroup_records.append({
        "group":    g_name,
        "n":        ng,
        "MAE_mean": round(np.mean(mae_g),  3),
        "MAE_std":  round(np.std(mae_g),   3),
        "RMSE_mean":round(np.mean(rmse_g), 3),
        "RMSE_std": round(np.std(rmse_g),  3),
    })

subgroup_df = pd.DataFrame(subgroup_records)
subgroup_df.to_csv("subgroup_results.csv", index=False)
print("\n  → subgroup_results.csv saved")

print("\n✓ Model training and evaluation complete.")
print(f"\n  Summary — LOOCV: MAE={loocv_mae:.3f}, "
      f"RMSE={loocv_rmse:.3f}, R²={loocv_r2:.3f}")
