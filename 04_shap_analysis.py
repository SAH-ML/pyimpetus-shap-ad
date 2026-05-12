"""
PyImpetus-SHAP Pipeline — Step 4: SHAP Analysis & Figures
==========================================================
Computes exact SHAP values using LinearExplainer, generates the
multi-panel SHAP importance and dependence figure, coefficient plot,
and pairwise correlation heatmap.

Inputs  : ADNI_Gene_Expression_Final_96_clean.csv
          six_gene_model.pkl
          six_gene_order.json
Outputs : Fig4_SHAP_Analysis.jpg / .pdf
          Fig5_Coeff_Heatmap.jpg  / .pdf
          shap_values.csv

Authors : Asif Hassan Syed et al., King Abdulaziz University
License : MIT
"""

import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
import io
import warnings

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"

DPI = 600

# ── Configuration ─────────────────────────────────────────────────
DATA_FILE  = "ADNI_Gene_Expression_Final_96_clean.csv"
MODEL_FILE = "six_gene_model.pkl"
GENE_FILE  = "six_gene_order.json"
TARGET     = "Delta_MMSE"

# Biological metadata for the six probes
GENE_META = {
    "11762936_x_at": {"symbol": "AQP7",   "role": "harmful",    "color": "#E24B4A"},
    "200024_PM_at":  {"symbol": "RPS5",   "role": "harmful",    "color": "#E24B4A"},
    "11762358_at":   {"symbol": "CHD2",   "role": "harmful",    "color": "#E24B4A"},
    "11763188_a_at": {"symbol": "SNX5",   "role": "protective", "color": "#1B6CA8"},
    "11757278_x_at": {"symbol": "ASS1",   "role": "harmful",    "color": "#E87040"},
    "11764118_at":   {"symbol": "Unchar", "role": "harmful",    "color": "#E87040"},
}

# ── Load data and model ───────────────────────────────────────────
print("Loading data and model …")
df = pd.read_csv(DATA_FILE)
with open(GENE_FILE) as f:
    six_genes = json.load(f)

X = df[six_genes]
y = df[TARGET].values
pipeline = joblib.load(MODEL_FILE)

scaler = pipeline.named_steps["standardscaler"]
en     = pipeline.named_steps["elasticnet"]
coef   = en.coef_

X_scaled = scaler.transform(X)
gene_symbols = [GENE_META[g]["symbol"] for g in six_genes]

# ── Compute SHAP values ───────────────────────────────────────────
print("Computing SHAP values (LinearExplainer — exact) …")
explainer   = shap.LinearExplainer(en, X_scaled,
                                    feature_perturbation="interventional")
shap_vals   = explainer.shap_values(X_scaled)     # shape (96, 6)
mean_abs_shap = np.abs(shap_vals).mean(axis=0)

# Save SHAP values
shap_df = pd.DataFrame(shap_vals, columns=gene_symbols)
shap_df.to_csv("shap_values.csv", index=False)
print("  → shap_values.csv saved")

print("\n  SHAP global importance:")
order = np.argsort(mean_abs_shap)[::-1]
for i in order:
    print(f"    {gene_symbols[i]:8s}  |SHAP| = {mean_abs_shap[i]:.3f}  "
          f"coef = {coef[i]:+.3f}")

# ── Helper: save fig as JPG and PDF ───────────────────────────────
def save_figure(fig, stem):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.save(f"{stem}.jpg", format="JPEG", dpi=(DPI, DPI),
             quality=95, subsampling=0)
    fig.savefig(f"{stem}.pdf", format="pdf", dpi=DPI,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  → {stem}.jpg / .pdf saved  ({img.size[0]}×{img.size[1]} px)")

# ════════════════════════════════════════════════════════════════
# FIGURE 4 — SHAP importance bar + 6 dependence plots
# ════════════════════════════════════════════════════════════════
print("\nGenerating Figure 4 — SHAP analysis …")
BG = "#F9FAFC"
fig4 = plt.figure(figsize=(18, 9), dpi=DPI)
fig4.patch.set_facecolor("white")
gs4 = gridspec.GridSpec(2, 4, figure=fig4,
                         hspace=0.42, wspace=0.38,
                         left=0.06, right=0.98,
                         top=0.93, bottom=0.09)

# Panel A: Global importance bar
axA = fig4.add_subplot(gs4[0, 0])
axA.set_facecolor(BG)
names_s = [gene_symbols[i] for i in order]
vals_s  = [mean_abs_shap[i] for i in order]
cols_s  = [GENE_META[six_genes[i]]["color"] for i in order]
bars = axA.barh(range(len(names_s)), vals_s, color=cols_s, alpha=0.85,
                edgecolor="white", linewidth=0.6, height=0.65)
for bar, val in zip(bars, vals_s):
    axA.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", ha="left",
             fontsize=9.5, fontweight="bold", color="#222222")
axA.set_yticks(range(len(names_s)))
axA.set_yticklabels(names_s, fontsize=11, fontweight="bold")
axA.set_xlabel("Mean |SHAP value|", fontsize=11, fontweight="bold")
axA.set_title("(A)   Global Feature Importance",
              fontsize=11, fontweight="bold", pad=8)
axA.set_xlim(0, max(vals_s) + 0.08)
axA.tick_params(labelsize=10)
axA.grid(True, alpha=0.18, lw=0.6, axis="x")
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)
axA.spines["left"].set_color("#BBBBBB")
axA.spines["bottom"].set_color("#BBBBBB")
p_harm = mpatches.Patch(fc="#E24B4A", alpha=0.85, label="Harmful probe")
p_prot = mpatches.Patch(fc="#1B6CA8", alpha=0.85, label="Protective — SNX5")
axA.legend(handles=[p_harm, p_prot], fontsize=8.5, loc="lower right",
           framealpha=0.9, edgecolor="#BBBBBB")

# Panels B–G: Dependence plots
panel_lbl = ["(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]
dep_pos   = [(0,1), (0,2), (0,3), (1,0), (1,1), (1,2)]

for idx in range(len(six_genes)):
    r, c = dep_pos[idx]
    ax   = fig4.add_subplot(gs4[r, c])
    ax.set_facecolor(BG)

    gene   = six_genes[idx]
    sym    = gene_symbols[idx]
    role   = GENE_META[gene]["role"]
    sv     = shap_vals[:, idx]
    xv     = X_scaled[:, idx]

    # Interaction: colour by the gene with strongest correlation
    interact_idx = (idx + 1) % len(six_genes)
    interact_v   = X_scaled[:, interact_idx]
    norm_i       = (interact_v - interact_v.min()) / (interact_v.max() - interact_v.min() + 1e-9)

    cmap = plt.cm.RdBu_r if role == "harmful" else plt.cm.RdBu
    sc = ax.scatter(xv, sv, c=norm_i, cmap=cmap, s=28, alpha=0.75,
                    edgecolors="none", zorder=4)

    # Trend line
    m_l, b_l, *_ = np.polyfit(xv, sv, 1), None
    m_l = np.polyfit(xv, sv, 1)
    x_sorted = np.sort(xv)
    ax.plot(x_sorted, np.polyval(m_l, x_sorted),
            color="#222222", lw=1.5, alpha=0.50, zorder=5)
    ax.axhline(0, color="#888888", lw=0.9, ls="--", alpha=0.55)

    if gene == "11763188_a_at":  # SNX5 threshold
        thresh = np.median(xv)
        ax.axvline(thresh, color="#1C7C5A", lw=1.4, ls=":",
                   alpha=0.85, label=f"Median = {thresh:.2f}")
        ax.legend(fontsize=7.5, loc="lower right",
                  framealpha=0.88, edgecolor="#BBBBBB")

    ax.set_xlabel(f"{sym} (scaled)", fontsize=10, fontweight="bold")
    ax.set_ylabel("SHAP value", fontsize=10, fontweight="bold")
    title_col = "#1C7C5A" if role == "protective" else "#C0392B"
    ax.set_title(f"{panel_lbl[idx]}   {sym}  [{role}]",
                 fontsize=10.5, fontweight="bold", pad=7, color=title_col)
    ax.text(0.04, 0.97,
            f"Coef: {coef[idx]:+.3f}\n|SHAP|: {mean_abs_shap[idx]:.3f}",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35",
                      fc="#E8F5E9" if role == "protective" else "#EBF2FF",
                      ec="#1C7C5A" if role == "protective" else "#1B4F8A",
                      alpha=0.92))
    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label(f"{gene_symbols[interact_idx]} expr.", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")

fig4.add_subplot(gs4[1, 3]).set_visible(False)
save_figure(fig4, "Fig4_SHAP_Analysis")
plt.close("all")

# ════════════════════════════════════════════════════════════════
# FIGURE 5 — Coefficient plot + Correlation heatmap
# ════════════════════════════════════════════════════════════════
print("\nGenerating Figure 5 — Coefficient plot and heatmap …")
corr_mat = X.corr().values

fig5 = plt.figure(figsize=(15, 7.2), dpi=DPI)
fig5.patch.set_facecolor("white")
fig5.subplots_adjust(left=0.07, right=0.97, top=0.91,
                     bottom=0.30, wspace=0.42)
gs5 = gridspec.GridSpec(1, 2, figure=fig5, wspace=0.42,
                         left=0.07, right=0.97, top=0.91, bottom=0.30)

# Panel A: Coefficient plot
axA5 = fig5.add_subplot(gs5[0])
axA5.set_facecolor(BG)
coef_order  = np.argsort(coef)
n_ord5      = [gene_symbols[i] for i in coef_order]
c_ord5      = [coef[i] for i in coef_order]
bar_cols5   = ["#E24B4A" if v < 0 else "#1B6CA8" for v in c_ord5]
bars5 = axA5.barh(range(len(n_ord5)), c_ord5,
                   color=bar_cols5, alpha=0.85,
                   edgecolor="white", linewidth=0.7, height=0.58)
for bar, val in zip(bars5, c_ord5):
    cy = bar.get_y() + bar.get_height()/2
    if val < 0:
        axA5.text(val - 0.022, cy, f"{val:+.3f}",
                  va="center", ha="right",
                  fontsize=10.5, fontweight="bold", color="#111111")
    else:
        axA5.text(val + 0.022, cy, f"{val:+.3f}",
                  va="center", ha="left",
                  fontsize=10.5, fontweight="bold", color="#111111")
axA5.axvline(0, color="#444444", lw=1.3, alpha=0.70)
axA5.set_yticks(range(len(n_ord5)))
axA5.set_yticklabels(n_ord5, fontsize=12, fontweight="bold")
axA5.set_xlabel("Elastic Net coefficient", fontsize=12, fontweight="bold",
                labelpad=10)
axA5.set_title("(A)   Elastic Net Coefficient Plot",
               fontsize=12, fontweight="bold", pad=10)
axA5.set_xlim(-0.82, 0.70)
axA5.tick_params(labelsize=10)
axA5.grid(True, alpha=0.18, lw=0.6, axis="x")
axA5.spines["top"].set_visible(False)
axA5.spines["right"].set_visible(False)
axA5.spines["left"].set_color("#BBBBBB")
axA5.spines["bottom"].set_color("#BBBBBB")
ph = mpatches.Patch(fc="#E24B4A", alpha=0.85, label="Harmful (negative coef.)")
pp = mpatches.Patch(fc="#1B6CA8", alpha=0.85, label="Protective — SNX5 (positive coef.)")
axA5.legend(handles=[ph, pp], fontsize=9.5, loc="lower right",
            framealpha=0.92, edgecolor="#BBBBBB")
fig5.text(0.185, 0.17,
          "\u2190  Higher expression \u2192 faster decline",
          ha="center", va="top", fontsize=9.5, fontweight="bold",
          color="#C0392B",
          bbox=dict(boxstyle="round,pad=0.40", fc="#FCEBEB",
                    ec="#E24B4A", alpha=0.92, linewidth=1.2))
fig5.text(0.385, 0.17,
          "Higher expression \u2192 slower decline  \u2192",
          ha="center", va="top", fontsize=9.5, fontweight="bold",
          color="#1B6CA8",
          bbox=dict(boxstyle="round,pad=0.40", fc="#E8F5E9",
                    ec="#1B6CA8", alpha=0.92, linewidth=1.2))

# Panel B: Correlation heatmap (lower triangle)
axB5 = fig5.add_subplot(gs5[1])
axB5.set_facecolor(BG)
n_g   = len(six_genes)
norm5 = TwoSlopeNorm(vmin=-0.40, vcenter=0, vmax=0.40)
cmap5 = plt.cm.RdBu_r

for i in range(n_g):
    for j in range(n_g):
        if j > i:
            axB5.add_patch(plt.Rectangle(
                [j-0.5, i-0.5], 1, 1, fc="#EEEEEE", ec="white", lw=1.5))
            continue
        val = corr_mat[i, j]
        axB5.add_patch(plt.Rectangle(
            [j-0.5, i-0.5], 1, 1, fc=cmap5(norm5(val)), ec="white", lw=1.5))
        txt_col = "white" if abs(val) > 0.22 else "#222222"
        disp = "1.00" if i == j else f"{val:+.2f}"
        axB5.text(j, i, disp, ha="center", va="center",
                  fontsize=10 if i != j else 11,
                  fontweight="bold" if i == j else "normal",
                  color=txt_col)

axB5.set_xlim(-0.5, n_g-0.5)
axB5.set_ylim(-0.5, n_g-0.5)
axB5.set_xticks(range(n_g))
axB5.set_xticklabels(gene_symbols, fontsize=11, fontweight="bold",
                     rotation=30, ha="right")
axB5.set_yticks(range(n_g))
axB5.set_yticklabels(gene_symbols, fontsize=11, fontweight="bold")
axB5.tick_params(length=0)
axB5.set_title("(B)   Pairwise Pearson Correlation Heatmap",
               fontsize=12, fontweight="bold", pad=10)
axB5.invert_yaxis()
axB5.spines["top"].set_visible(False)
axB5.spines["right"].set_visible(False)
axB5.spines["left"].set_color("#BBBBBB")
axB5.spines["bottom"].set_color("#BBBBBB")
sm5 = plt.cm.ScalarMappable(cmap=cmap5, norm=norm5)
sm5.set_array([])
cb5 = plt.colorbar(sm5, ax=axB5, fraction=0.046, pad=0.04, shrink=0.82)
cb5.set_label("Pearson r", fontsize=10, fontweight="bold")
cb5.ax.tick_params(labelsize=9)
fig5.text(0.735, 0.17,
          "Harmful module (AQP7, RPS5, Unchar, ASS1, CHD2):"
          "  positively inter-correlated  (r = 0.12 – 0.34)",
          ha="center", va="top", fontsize=9.0, fontweight="bold",
          color="#C0392B",
          bbox=dict(boxstyle="round,pad=0.40", fc="#FCEBEB",
                    ec="#E24B4A", alpha=0.92, linewidth=1.2))
fig5.text(0.735, 0.09,
          "Protective module (SNX5):"
          "  negatively correlated with harmful probes"
          "  (r = \u22120.08 to \u22120.18)",
          ha="center", va="top", fontsize=9.0, fontweight="bold",
          color="#1B6CA8",
          bbox=dict(boxstyle="round,pad=0.40", fc="#E8F5E9",
                    ec="#1B6CA8", alpha=0.92, linewidth=1.2))

save_figure(fig5, "Fig5_Coeff_Heatmap")
plt.close("all")

print("\n✓ SHAP analysis complete.")
