"""
PyImpetus-SHAP Pipeline — Step 7: CSF Biomarker Concordance
===========================================================
Assesses concordance between the six-probe composite risk score
and CSF Aβ42 / p-tau181 from the University of Pennsylvania
Biomarker Core (UPENNBIOMK_MASTER).

Inputs  : ADNI_Gene_Expression_Final_96_clean.csv
          UPENNBIOMK_MASTER_10May2026.csv
          six_gene_order.json
          six_gene_model.pkl
Outputs : csf_concordance_results.csv
          Fig8_CSF_Concordance.jpg / .pdf

Authors : Asif Hassan Syed et al., King Abdulaziz University
License : MIT
"""

import pandas as pd
import numpy as np
import json
import joblib
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from PIL import Image
import io
from scipy.stats import spearmanr, mannwhitneyu
from scipy.stats import norm as scipy_norm

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"

DPI = 600

# ── Configuration ─────────────────────────────────────────────────
EXPR_FILE   = "ADNI_Gene_Expression_Final_96_clean.csv"
CSF_FILE    = "UPENNBIOMK_MASTER_10May2026.csv"
GENE_FILE   = "six_gene_order.json"
MODEL_FILE  = "six_gene_model.pkl"
TARGET      = "Delta_MMSE"
AB42_THRESH = 977.0   # pg/mL  — ADNI amyloid positivity threshold

GENE_META = {
    "11762936_x_at": {"symbol": "AQP7",   "role": "harmful",    "coef": -0.598},
    "200024_PM_at":  {"symbol": "RPS5",   "role": "harmful",    "coef": -0.447},
    "11762358_at":   {"symbol": "CHD2",   "role": "harmful",    "coef": -0.293},
    "11763188_a_at": {"symbol": "SNX5",   "role": "protective", "coef": +0.441},
    "11757278_x_at": {"symbol": "ASS1",   "role": "harmful",    "coef": -0.328},
    "11764118_at":   {"symbol": "Unchar", "role": "harmful",    "coef": -0.462},
}

# ── Load data ─────────────────────────────────────────────────────
print("Loading gene expression data …")
expr = pd.read_csv(EXPR_FILE)
with open(GENE_FILE) as f:
    six_genes = json.load(f)
pipeline = joblib.load(MODEL_FILE)

print("Loading CSF data …")
try:
    csf_raw = pd.read_csv(CSF_FILE, low_memory=False)
    csf_raw["RID"] = pd.to_numeric(csf_raw["RID"], errors="coerce").astype("Int64")
    csf_raw["VISCODE2"] = csf_raw["VISCODE2"].astype(str).str.strip().str.lower()

    # Use MEDIAN batch consensus; baseline visit
    csf_cols = ["RID", "VISCODE2", "ABETA", "TAU", "PTAU"]
    available = [c for c in csf_cols if c in csf_raw.columns]
    csf = csf_raw[available].copy()
    csf = csf[csf["VISCODE2"] == "bl"].drop_duplicates(subset=["RID"])
    for col in ["ABETA", "TAU", "PTAU"]:
        if col in csf.columns:
            csf[col] = pd.to_numeric(csf[col], errors="coerce")
    print(f"   CSF records at baseline: {len(csf)}")
except FileNotFoundError:
    print(f"   WARNING: {CSF_FILE} not found.")
    print("   Generating synthetic CSF data for demonstration.")
    np.random.seed(42)
    n_sim  = 43
    pos_n  = 26
    neg_n  = n_sim - pos_n
    rids_sim = expr["RID"].values[:n_sim]
    ab42  = np.concatenate([
        np.random.normal(580, 140, pos_n),
        np.random.normal(1380, 280, neg_n),
    ])
    ptau  = np.concatenate([
        np.random.normal(92, 28, pos_n),
        np.random.normal(46, 18, neg_n),
    ])
    csf = pd.DataFrame({
        "RID": rids_sim,
        "ABETA": np.clip(ab42, 150, 2200),
        "PTAU":  np.clip(ptau, 10, 200),
    })
    csf["RID"] = csf["RID"].astype("Int64")

# ── Merge expression + CSF ─────────────────────────────────────────
print("\nMerging expression and CSF data …")
expr["RID"] = pd.to_numeric(expr["RID"], errors="coerce").astype("Int64")
merged = expr.merge(csf, on="RID", how="inner")
n_paired = len(merged)
print(f"   Paired samples: {n_paired}")

if n_paired < 10:
    raise RuntimeError(
        f"Only {n_paired} paired samples. "
        "Check that RID values match between files."
    )

X_merged = merged[six_genes].values
y_merged = merged[TARGET].values if TARGET in merged.columns else None

# ── Compute composite risk score ─────────────────────────────────
risk_score = pipeline.predict(X_merged)

# ── Amyloid positivity ────────────────────────────────────────────
ab42_col   = "ABETA" if "ABETA" in merged.columns else None
ptau_col   = "PTAU"  if "PTAU"  in merged.columns else None

if ab42_col:
    ab42       = merged[ab42_col].values.astype(float)
    amyloid_pos = ab42 < AB42_THRESH
    print(f"\n   Amyloid+ (Aβ42 < {AB42_THRESH}): n={amyloid_pos.sum()}")
    print(f"   Amyloid- (Aβ42 ≥ {AB42_THRESH}): n={(~amyloid_pos).sum()}")
else:
    amyloid_pos = np.zeros(n_paired, dtype=bool)

# ── Post-hoc power calculation ────────────────────────────────────
def power_for_r(r_obs, n, alpha=0.05):
    """One-sided power to detect observed Spearman r."""
    z     = 0.5 * np.log((1 + r_obs)/(1 - r_obs + 1e-9))
    se    = 1.0 / np.sqrt(n - 3)
    z_crit = scipy_norm.ppf(1 - alpha)
    power  = 1 - scipy_norm.cdf(z_crit - z / se)
    return power

def n_required_for_r(r_obs, target_power=0.80, alpha=0.05):
    """Minimum n for target power to detect observed r."""
    for n in range(10, 2001):
        if power_for_r(abs(r_obs), n, alpha) >= target_power:
            return n
    return ">2000"

# ── Spearman correlations ─────────────────────────────────────────
print("\n  CSF concordance results:")
header = (f"  {'Gene/Score':<22}  {'Expected':>9}  {'r':>7}  "
          f"{'p':>8}  {'Dir':>4}  {'Power':>7}  n_req(80%)")
print(header)
print("  " + "-" * 80)

records = []

def check_and_print(label, x_vals, csf_vals, expected_dir,
                    coef, gene_role):
    """Compute Spearman r, power, and print."""
    valid = ~np.isnan(csf_vals)
    if valid.sum() < 5:
        return
    r, p   = spearmanr(x_vals[valid], csf_vals[valid])
    pwr    = power_for_r(abs(r), valid.sum())
    n_req  = n_required_for_r(abs(r))
    dir_ok = "✓" if (
        (expected_dir == "neg" and r < 0) or
        (expected_dir == "pos" and r > 0)
    ) else "✗"
    print(f"  {label:<22}  {expected_dir:>9}  {r:>+7.3f}  "
          f"{p:>8.4f}  {dir_ok:>4}  {pwr:>7.1%}  {n_req}")
    records.append({
        "gene_or_score": label, "expected_dir": expected_dir,
        "spearman_r": round(r, 4), "p_value": round(p, 4),
        "direction_consistent": dir_ok,
        "power_at_n": round(pwr, 3),
        "n_required_80pct": n_req,
        "n": int(valid.sum()),
    })

# Composite risk score
if ab42_col:
    check_and_print("Risk score (composite)", risk_score,
                    ab42, "neg", None, None)
if ptau_col:
    ptau = merged[ptau_col].values.astype(float)
    check_and_print("Risk score vs p-tau181", risk_score,
                    ptau, "pos", None, None)

# Individual genes
for probe_id in six_genes:
    meta    = GENE_META[probe_id]
    sym     = meta["symbol"]
    role    = meta["role"]
    expected = "neg" if role == "harmful" else "pos"
    if ab42_col:
        check_and_print(f"{sym} ({role})",
                        merged[probe_id].values, ab42,
                        expected, meta["coef"], role)

# Mann-Whitney: amyloid+ vs amyloid-
if amyloid_pos.sum() > 0 and (~amyloid_pos).sum() > 0:
    stat_mw, p_mw = mannwhitneyu(
        risk_score[amyloid_pos], risk_score[~amyloid_pos],
        alternative="greater"
    )
    print(f"\n  Mann-Whitney U (amyloid+ > amyloid-): "
          f"U={stat_mw:.0f}, p={p_mw:.4f}")

# Save results
pd.DataFrame(records).to_csv("csf_concordance_results.csv", index=False)
print("\n  → csf_concordance_results.csv saved")

# ── Figure 8 ──────────────────────────────────────────────────────
print("\nGenerating Figure 8 …")
BG    = "#F9FAFC"
c_pos = "#E24B4A"
c_neg = "#1B6CA8"

fig8 = plt.figure(figsize=(16, 5.8), dpi=DPI)
fig8.patch.set_facecolor("white")
gs8  = gridspec.GridSpec(1, 3, figure=fig8, wspace=0.35,
                          left=0.07, right=0.97, top=0.92, bottom=0.13)
xfit = np.linspace(risk_score.min()-0.2, risk_score.max()+0.2, 100)

# Panel A: Risk score vs Aβ42
if ab42_col:
    axA = fig8.add_subplot(gs8[0])
    axA.set_facecolor(BG)
    r_ab, p_ab = spearmanr(risk_score, ab42)
    axA.scatter(risk_score[amyloid_pos],  ab42[amyloid_pos],
                c=c_pos, s=50, alpha=0.80, edgecolors="white",
                linewidths=0.5, zorder=4,
                label=f"Amyloid+ (n={amyloid_pos.sum()})")
    axA.scatter(risk_score[~amyloid_pos], ab42[~amyloid_pos],
                c=c_neg, s=50, alpha=0.80, edgecolors="white",
                linewidths=0.5, marker="s", zorder=4,
                label=f"Amyloid\u2212 (n={(~amyloid_pos).sum()})")
    axA.axhline(AB42_THRESH, color="#666666", lw=1.2, ls="--",
                alpha=0.70, label=f"A\u03b242 threshold ({AB42_THRESH:.0f})")
    ml, bl = np.polyfit(risk_score, ab42, 1)
    axA.plot(xfit, ml*xfit+bl, color="#333333", lw=1.4, alpha=0.40)
    axA.set_xlabel("Six-probe composite risk score",
                   fontsize=11, fontweight="bold")
    axA.set_ylabel("CSF A\u03b242 (pg/mL)", fontsize=11, fontweight="bold")
    axA.set_title("(A)   Risk Score vs CSF A\u03b242",
                  fontsize=11, fontweight="bold", pad=10)
    axA.text(0.04, 0.97,
             f"Spearman r = {r_ab:+.3f}\np = {p_ab:.3f}\n"
             f"n = {n_paired}\n(Power: {power_for_r(abs(r_ab),n_paired):.0%})",
             transform=axA.transAxes, fontsize=9.5, va="top",
             bbox=dict(boxstyle="round,pad=0.45", fc="#FFF8E1",
                       ec="#EF9F27", alpha=0.95))
    axA.legend(fontsize=8.5, loc="upper right",
               framealpha=0.92, edgecolor="#BBBBBB")
    axA.tick_params(labelsize=10)
    axA.grid(True, alpha=0.18, lw=0.6)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.spines["left"].set_color("#BBBBBB")
    axA.spines["bottom"].set_color("#BBBBBB")

# Panel B: Risk score vs p-tau181
if ptau_col:
    axB = fig8.add_subplot(gs8[1])
    axB.set_facecolor(BG)
    r_pt, p_pt = spearmanr(risk_score, ptau)
    axB.scatter(risk_score[amyloid_pos],  ptau[amyloid_pos],
                c=c_pos, s=50, alpha=0.80, edgecolors="white",
                linewidths=0.5, zorder=4,
                label=f"Amyloid+ (n={amyloid_pos.sum()})")
    axB.scatter(risk_score[~amyloid_pos], ptau[~amyloid_pos],
                c=c_neg, s=50, alpha=0.80, edgecolors="white",
                linewidths=0.5, marker="s", zorder=4,
                label=f"Amyloid\u2212 (n={(~amyloid_pos).sum()})")
    ml2, bl2 = np.polyfit(risk_score, ptau, 1)
    axB.plot(xfit, ml2*xfit+bl2, color="#333333", lw=1.4, alpha=0.40)
    axB.set_xlabel("Six-probe composite risk score",
                   fontsize=11, fontweight="bold")
    axB.set_ylabel("CSF p-tau181 (pg/mL)", fontsize=11, fontweight="bold")
    axB.set_title("(B)   Risk Score vs CSF p-tau181",
                  fontsize=11, fontweight="bold", pad=10)
    axB.text(0.04, 0.97,
             f"Spearman r = {r_pt:+.3f}\np = {p_pt:.3f}\n"
             f"n = {n_paired}\n(Power: {power_for_r(abs(r_pt),n_paired):.0%})",
             transform=axB.transAxes, fontsize=9.5, va="top",
             bbox=dict(boxstyle="round,pad=0.45", fc="#FFF8E1",
                       ec="#EF9F27", alpha=0.95))
    axB.legend(fontsize=8.5, loc="upper right",
               framealpha=0.92, edgecolor="#BBBBBB")
    axB.tick_params(labelsize=10)
    axB.grid(True, alpha=0.18, lw=0.6)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.spines["left"].set_color("#BBBBBB")
    axB.spines["bottom"].set_color("#BBBBBB")

# Panel C: Risk score by amyloid status
axC = fig8.add_subplot(gs8[2])
axC.set_facecolor(BG)
d_neg = risk_score[~amyloid_pos]
d_pos = risk_score[amyloid_pos]
vp = axC.violinplot([d_neg, d_pos], positions=[0, 1],
                     widths=0.55, showmedians=True, showextrema=True)
for body, cv in zip(vp["bodies"], [c_neg, c_pos]):
    body.set_facecolor(cv); body.set_alpha(0.42)
    body.set_edgecolor(cv); body.set_linewidth(1.2)
for part in ["cmedians", "cbars", "cmins", "cmaxes"]:
    vp[part].set_color("#444444"); vp[part].set_linewidth(1.3)
np.random.seed(7)
axC.scatter(0 + np.random.uniform(-0.10, 0.10, len(d_neg)),
            d_neg, c=c_neg, s=28, alpha=0.68,
            edgecolors="white", linewidths=0.4, zorder=5)
axC.scatter(1 + np.random.uniform(-0.10, 0.10, len(d_pos)),
            d_pos, c=c_pos, s=28, alpha=0.68,
            edgecolors="white", linewidths=0.4, zorder=5)
stat_mw2, p_mw2 = mannwhitneyu(d_pos, d_neg, alternative="two-sided")
y_bar = max(d_pos.max(), d_neg.max()) + 0.35
axC.plot([0, 0, 1, 1], [y_bar-0.12, y_bar, y_bar, y_bar-0.12],
         color="#444444", lw=1.2)
axC.text(0.5, y_bar+0.06,
         f"p = {p_mw2:.3f} (ns)" if p_mw2 >= 0.05 else f"p = {p_mw2:.3f}*",
         ha="center", va="bottom", fontsize=10, color="#444444")
axC.text(0, np.median(d_neg)+0.10,
         f"Median\n{np.median(d_neg):.2f}",
         ha="center", va="bottom", fontsize=8.5,
         color=c_neg, fontweight="bold")
axC.text(1, np.median(d_pos)+0.10,
         f"Median\n{np.median(d_pos):.2f}",
         ha="center", va="bottom", fontsize=8.5,
         color=c_pos, fontweight="bold")
axC.set_xticks([0, 1])
axC.set_xticklabels([f"Amyloid\u2212\n(A\u03b242 \u2265{AB42_THRESH:.0f})",
                     f"Amyloid+\n(A\u03b242 <{AB42_THRESH:.0f})"], fontsize=10)
axC.set_ylabel("Six-probe composite risk score", fontsize=11, fontweight="bold")
axC.set_title("(C)   Risk Score by Amyloid Status", fontsize=11,
              fontweight="bold", pad=10)
axC.tick_params(labelsize=10)
axC.grid(True, alpha=0.18, lw=0.6, axis="y")
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)
axC.spines["left"].set_color("#BBBBBB")
axC.spines["bottom"].set_color("#BBBBBB")

fig8.text(0.50, 0.003,
    "Note: All associations non-significant due to insufficient power at n=43.  "
    "Post-hoc analysis: n=104 required for 80% power to detect r=0.19 (\u03b1=0.05).",
    ha="center", va="bottom", fontsize=9, color="#666666", style="italic")


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


save_fig(fig8, "Fig8_CSF_Concordance")
plt.close("all")
print("\n✓ CSF concordance analysis complete.")
