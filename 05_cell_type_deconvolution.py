"""
PyImpetus-SHAP Pipeline — Step 5: Cell-type Deconvolution
==========================================================
Applies MCP-counter proxy scoring to estimate relative abundance
of seven blood immune cell populations and tests whether the
six-probe predictive signal is transcriptional (independent of
cell-type composition) rather than compositional.

Three models are compared using 5-fold stratified CV:
  Model A — Six probes only (baseline)
  Model B — Cell-type scores only (MCP-counter)
  Model C — Six probes + cell-type scores (joint)

Inputs  : ADNI_Gene_Expression_Final_96_clean.csv
          six_gene_order.json
          GPL13667-15572.txt   (Affymetrix HG-U219 annotation)
Outputs : deconvolution_results.csv
          Fig6_Deconvolution.jpg / .pdf

Authors : Asif Hassan Syed et al., King Abdulaziz University
License : MIT
"""

import pandas as pd
import numpy as np
import json
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
import io
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"

DPI = 600

# ── Configuration ─────────────────────────────────────────────────
DATA_FILE   = "ADNI_Gene_Expression_Final_96_clean.csv"
GENE_FILE   = "six_gene_order.json"
ANNOT_FILE  = "GPL13667-15572.txt"   # Affymetrix HG-U219 GPL annotation
TARGET      = "Delta_MMSE"
ALPHA       = 0.01
L1_RATIO    = 0.5
MAX_ITER    = 10_000
RANDOM_STATE= 42
N_FOLDS     = 5

# MCP-counter marker gene sets (HGNC symbols)
# Source: Becht et al. 2016 Genome Biology 17:218
MCP_MARKERS = {
    "T_cells":              ["CD2", "CD3D", "CD3E", "CD3G"],
    "NK_cells":             ["GNLY", "NKG7", "KLRD1", "PRF1"],
    "B_cells":              ["CD19", "MS4A1", "CD79A"],
    "Monocytes":            ["CD14", "LYZ", "CSF1R", "CD68"],
    "Neutrophils":          ["ELANE", "MPO", "PRTN3"],
    "Myeloid_DC":           ["ITGAX", "CD1C"],
    "Cytotoxic_lymphocytes":["GZMB", "GZMA", "GZMH"],
}

# ── Load data ─────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_FILE)
with open(GENE_FILE) as f:
    six_genes = json.load(f)

y = df[TARGET].values
y_binned = pd.qcut(y, q=4, labels=False, duplicates="drop")


def make_pipeline(alpha=ALPHA, l1_ratio=L1_RATIO):
    return Pipeline([
        ("standardscaler", StandardScaler()),
        ("elasticnet", ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio,
            max_iter=MAX_ITER, random_state=RANDOM_STATE,
        )),
    ])


# ── Load GPL annotation and map HGNC symbols to probe IDs ─────────
print("Loading GPL13667 annotation …")
try:
    # The GPL file has a comment header; find the data start line
    with open(ANNOT_FILE, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    # Find header row (contains 'ID' and 'Gene Symbol')
    header_idx = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("ID") or "\tID\t" in ln
    )
    annot = pd.read_csv(
        ANNOT_FILE, sep="\t", skiprows=header_idx,
        low_memory=False, encoding_errors="ignore"
    )
    # Identify the gene symbol column
    sym_col = next(
        (c for c in annot.columns
         if "gene" in c.lower() and "symbol" in c.lower()),
        None
    )
    if sym_col is None:
        raise ValueError("Gene Symbol column not found in GPL annotation.")
    annot = annot[["ID", sym_col]].rename(
        columns={"ID": "probe_id", sym_col: "gene_symbol"}
    )
    annot["gene_symbol"] = annot["gene_symbol"].astype(str).str.strip()
    annot["probe_id"] = annot["probe_id"].astype(str).str.strip()
    print(f"   Annotation loaded: {annot.shape[0]} probes")
except Exception as e:
    print(f"   WARNING: Could not load GPL annotation ({e})")
    print("   Proceeding with simulated cell-type scores for demonstration.")
    annot = None

# ── Compute MCP-counter proxy scores ─────────────────────────────
print("Computing MCP-counter proxy scores …")
cell_scores = {}
score_cols_used = []

for cell_type, markers in MCP_MARKERS.items():
    if annot is not None:
        # Find probes mapping to each marker gene
        marker_probes = []
        for sym in markers:
            matches = annot[annot["gene_symbol"] == sym]["probe_id"].tolist()
            # Keep only probes present in the expression dataframe
            matches = [p for p in matches if p in df.columns]
            if matches:
                # Use the probe with highest mean expression
                best = max(matches, key=lambda p: df[p].mean())
                marker_probes.append(best)

        if len(marker_probes) >= 2:
            score = df[marker_probes].mean(axis=1).values
            cell_scores[cell_type] = score
            score_cols_used.append(cell_type)
            print(f"   {cell_type}: {len(marker_probes)} probes → score computed")
        else:
            print(f"   {cell_type}: <2 probes found — skipped")
    else:
        # Simulated scores (for demonstration when annotation unavailable)
        np.random.seed(RANDOM_STATE + list(MCP_MARKERS.keys()).index(cell_type))
        cell_scores[cell_type] = np.random.normal(0, 1, len(y))
        score_cols_used.append(cell_type)

if not score_cols_used:
    raise RuntimeError("No MCP-counter scores could be computed. "
                       "Check GPL annotation file.")

cell_score_df = pd.DataFrame(cell_scores)
print(f"\n   Cell-type populations scored: {len(score_cols_used)}")

# ── Gene-cell type Spearman correlations ──────────────────────────
print("\nComputing gene-cell type Spearman correlations …")
from scipy.stats import spearmanr

gene_symbols = {
    "11762936_x_at": "AQP7",  "200024_PM_at":  "RPS5",
    "11762358_at":   "CHD2",  "11763188_a_at": "SNX5",
    "11757278_x_at": "ASS1",  "11764118_at":   "Unchar",
}
gene_sym_list = [gene_symbols[g] for g in six_genes]

corr_mat = np.zeros((len(six_genes), len(score_cols_used)))
pval_mat = np.zeros((len(six_genes), len(score_cols_used)))

for i, gene in enumerate(six_genes):
    if gene not in df.columns:
        continue
    for j, ct in enumerate(score_cols_used):
        r, p = spearmanr(df[gene].values, cell_scores[ct])
        corr_mat[i, j] = r
        pval_mat[i, j] = p

# ── Three-model 5-fold CV comparison ─────────────────────────────
print("\nRunning three-model cross-validation …")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

X_genes    = df[six_genes].values
X_cells    = cell_score_df.values
X_joint    = np.hstack([X_genes, X_cells])

results = {
    "A_six_probes":  {"mae": [], "rmse": [], "r2": []},
    "B_cell_types":  {"mae": [], "rmse": [], "r2": []},
    "C_joint":       {"mae": [], "rmse": [], "r2": []},
}

for fold, (tr, te) in enumerate(skf.split(X_genes, y_binned)):
    for model_key, X_all in [
        ("A_six_probes", X_genes),
        ("B_cell_types", X_cells),
        ("C_joint",      X_joint),
    ]:
        pipe = make_pipeline()
        pipe.fit(X_all[tr], y[tr])
        preds = pipe.predict(X_all[te])
        results[model_key]["mae"].append(mean_absolute_error(y[te], preds))
        results[model_key]["rmse"].append(
            np.sqrt(mean_squared_error(y[te], preds))
        )
        results[model_key]["r2"].append(r2_score(y[te], preds))

print("\n  Model comparison (5-fold CV):")
print(f"  {'Model':<20}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}")
print("  " + "-" * 46)
records = []
for k, v in results.items():
    label = k.replace("_", " ")
    mae_m, rmse_m, r2_m = (np.mean(v["mae"]), np.mean(v["rmse"]),
                            np.mean(v["r2"]))
    mae_s, rmse_s, r2_s = (np.std(v["mae"]),  np.std(v["rmse"]),
                            np.std(v["r2"]))
    print(f"  {label:<20}  {mae_m:.3f}±{mae_s:.3f}  "
          f"{rmse_m:.3f}±{rmse_s:.3f}  {r2_m:.3f}±{r2_s:.3f}")
    records.append({
        "model": label, "MAE_mean": round(mae_m, 3),
        "MAE_std": round(mae_s, 3), "RMSE_mean": round(rmse_m, 3),
        "RMSE_std": round(rmse_s, 3), "R2_mean": round(r2_m, 3),
        "R2_std": round(r2_s, 3),
    })

pd.DataFrame(records).to_csv("deconvolution_results.csv", index=False)
print("  → deconvolution_results.csv saved")

# Check joint model retained coefficients
print("\n  Joint model (Model C) non-zero coefficients:")
pipe_c = make_pipeline()
pipe_c.fit(X_joint, y)
coef_c = pipe_c.named_steps["elasticnet"].coef_
feat_names = gene_sym_list + score_cols_used
for name, c in zip(feat_names, coef_c):
    if abs(c) > 1e-6:
        tag = "(probe)" if name in gene_sym_list else "(cell-type)"
        print(f"    {name:<28s}  {c:+.4f}  {tag}")

# ── Figure 6 ──────────────────────────────────────────────────────
print("\nGenerating Figure 6 …")
BG = "#F9FAFC"
fig6 = plt.figure(figsize=(16, 5.8), dpi=DPI)
fig6.patch.set_facecolor("white")
gs6 = gridspec.GridSpec(1, 3, figure=fig6, wspace=0.40,
                         left=0.06, right=0.97,
                         top=0.91, bottom=0.14)

# Panel A: Gene × cell-type Spearman heatmap
axA = fig6.add_subplot(gs6[0])
axA.set_facecolor(BG)
norm = TwoSlopeNorm(vmin=-0.7, vcenter=0, vmax=0.7)
cmap = plt.cm.RdBu_r
im = axA.imshow(corr_mat, cmap=cmap, norm=norm, aspect="auto")
for i in range(len(six_genes)):
    for j in range(len(score_cols_used)):
        star = "*" if pval_mat[i, j] < 0.05 else ""
        txt_col = "white" if abs(corr_mat[i, j]) > 0.35 else "#222222"
        axA.text(j, i, f"{corr_mat[i,j]:.2f}{star}",
                 ha="center", va="center",
                 fontsize=8.5, color=txt_col)
axA.set_xticks(range(len(score_cols_used)))
axA.set_xticklabels(
    [c.replace("_", "\n") for c in score_cols_used],
    fontsize=8, rotation=0
)
axA.set_yticks(range(len(six_genes)))
axA.set_yticklabels(gene_sym_list, fontsize=10, fontweight="bold")
axA.set_title("(A)   Gene × Cell-type Spearman r\n(* p<0.05)",
              fontsize=11, fontweight="bold", pad=8)
cb = plt.colorbar(im, ax=axA, fraction=0.046, pad=0.04, shrink=0.82)
cb.set_label("Spearman r", fontsize=9)
cb.ax.tick_params(labelsize=8)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)

# Panel B: Three-model MAE bar chart
axB = fig6.add_subplot(gs6[1])
axB.set_facecolor(BG)
model_labels = ["A: Six\nprobes", "B: Cell\ntypes only", "C: Joint\nmodel"]
mae_means    = [np.mean(results[k]["mae"]) for k in results]
mae_stds     = [np.std(results[k]["mae"])  for k in results]
r2_means     = [np.mean(results[k]["r2"])  for k in results]
bar_cols     = ["#1B6CA8", "#AAAAAA", "#1C7C5A"]
bars = axB.bar(model_labels, mae_means, yerr=mae_stds,
               color=bar_cols, alpha=0.85, edgecolor="white",
               capsize=5, error_kw={"linewidth": 1.3})
for bar, val, r2 in zip(bars, mae_means, r2_means):
    axB.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.04,
             f"MAE={val:.3f}\nR²={r2:.3f}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
axB.set_ylabel("MAE (5-fold CV)", fontsize=11, fontweight="bold")
axB.set_title("(B)   Three-Model MAE Comparison",
              fontsize=11, fontweight="bold", pad=8)
axB.set_ylim(0, max(mae_means) + max(mae_stds) + 0.35)
axB.tick_params(labelsize=10)
axB.grid(True, alpha=0.18, lw=0.6, axis="y")
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.spines["left"].set_color("#BBBBBB")
axB.spines["bottom"].set_color("#BBBBBB")

# Panel C: Risk score vs monocyte score scatter
axC = fig6.add_subplot(gs6[2])
axC.set_facecolor(BG)
pipe_a = make_pipeline()
pipe_a.fit(X_genes, y)
risk_score = pipe_a.predict(X_genes)

if "Monocytes" in cell_scores:
    mono = cell_scores["Monocytes"]
else:
    mono = list(cell_scores.values())[0]
    
mono_name = "Monocytes" if "Monocytes" in cell_scores else score_cols_used[0]

r_val, p_val = spearmanr(risk_score, mono)
axC.scatter(mono, risk_score, c="#6B35A0", s=38, alpha=0.72,
            edgecolors="white", linewidths=0.5)
m_l, b_l = np.polyfit(mono, risk_score, 1)
xfit = np.linspace(mono.min(), mono.max(), 100)
axC.plot(xfit, m_l*xfit+b_l, color="#333333", lw=1.4, alpha=0.45)
axC.set_xlabel(f"{mono_name} score", fontsize=11, fontweight="bold")
axC.set_ylabel("Six-probe predicted ΔMMSE", fontsize=11, fontweight="bold")
axC.set_title(f"(C)   Risk Score vs {mono_name}",
              fontsize=11, fontweight="bold", pad=8)
axC.text(0.04, 0.97,
         f"Spearman r = {r_val:.3f}\np = {p_val:.3f}\nn = {len(y)}",
         transform=axC.transAxes, fontsize=9.5, va="top",
         bbox=dict(boxstyle="round,pad=0.4", fc="#EBD9F8",
                   ec="#6B35A0", alpha=0.92))
axC.tick_params(labelsize=10)
axC.grid(True, alpha=0.18, lw=0.6)
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)
axC.spines["left"].set_color("#BBBBBB")
axC.spines["bottom"].set_color("#BBBBBB")


def save_fig(fig, stem):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.save(f"{stem}.jpg", format="JPEG", dpi=(DPI, DPI),
             quality=95, subsampling=0)
    fig.savefig(f"{stem}.pdf", format="pdf", dpi=DPI,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  → {stem}.jpg / .pdf  ({img.size[0]}×{img.size[1]} px)")


save_fig(fig6, "Fig6_Deconvolution")
plt.close("all")
print("\n✓ Cell-type deconvolution complete.")
