"""
PyImpetus-SHAP — Real-Time Clinical Decision Support Tool
=========================================================
Streamlit web application for live demonstration of the
six-probe Alzheimer disease cognitive decline predictor.

Deploy free at: https://streamlit.io/cloud
No server, no installation needed by the user.

Authors : Asif Hassan Syed et al., King Abdulaziz University
License : MIT
"""

import streamlit as st
import numpy as np
import json
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title  = "PyImpetus-SHAP | AD Cognitive Decline Predictor",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem; font-weight: 800; color: #1B4F8A;
        text-align: center; padding-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem; color: #555555;
        text-align: center; padding-bottom: 1.2rem;
    }
    .risk-high     { background:#FDECEA; border:2px solid #E24B4A;
                     border-radius:10px; padding:14px; text-align:center; }
    .risk-moderate { background:#FFF8E1; border:2px solid #EF9F27;
                     border-radius:10px; padding:14px; text-align:center; }
    .risk-low      { background:#E8F5E9; border:2px solid #1C7C5A;
                     border-radius:10px; padding:14px; text-align:center; }
    .metric-box    { background:#EBF2FF; border:1px solid #1B4F8A;
                     border-radius:8px; padding:10px; text-align:center; }
    .disclaimer    { background:#FFF9E6; border-left:4px solid #EF9F27;
                     padding:10px 14px; font-size:0.85rem; color:#555; }
    .gene-harmful  { color:#C0392B; font-weight:700; }
    .gene-protect  { color:#1C7C5A; font-weight:700; }
    div[data-testid="stSlider"] > label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Gene metadata ─────────────────────────────────────────────────
GENE_ORDER = [
    "11762936_x_at",
    "200024_PM_at",
    "11762358_at",
    "11763188_a_at",
    "11757278_x_at",
    "11764118_at",
]

GENE_META = {
    "11762936_x_at": {
        "symbol":   "AQP7",
        "name":     "Aquaporin 7",
        "role":     "harmful",
        "coef":     -0.598,
        "min_val":  2.5, "max_val": 5.5, "default": 3.75,
        "step":     0.01,
        "info":     "Glycerol/water channel; glymphatic & metabolic regulation. "
                    "Higher expression → faster cognitive decline.",
    },
    "200024_PM_at": {
        "symbol":   "RPS5",
        "name":     "Ribosomal Protein S5",
        "role":     "harmful",
        "coef":     -0.447,
        "min_val":  9.5, "max_val": 13.0, "default": 11.25,
        "step":     0.01,
        "info":     "40S ribosomal protein; tau mRNA translation & "
                    "ribosomal stress in AD.",
    },
    "11762358_at": {
        "symbol":   "CHD2",
        "name":     "Chromodomain Helicase DNA-Binding Protein 2",
        "role":     "harmful",
        "coef":     -0.293,
        "min_val":  6.0, "max_val": 11.0, "default": 8.25,
        "step":     0.01,
        "info":     "Epigenetic chromatin remodeller; dysregulated in "
                    "microglial AD pathology.",
    },
    "11763188_a_at": {
        "symbol":   "SNX5",
        "name":     "Sorting Nexin 5 (retromer component)",
        "role":     "protective",
        "coef":     +0.441,
        "min_val":  6.0, "max_val": 10.0, "default": 8.30,
        "step":     0.01,
        "info":     "Retromer trafficking of APP/BACE1. "
                    "Higher expression → slower decline (PROTECTIVE). "
                    "Replicated in AddNeuroMed p=0.002.",
    },
    "11757278_x_at": {
        "symbol":   "ASS1",
        "name":     "Argininosuccinate Synthase 1",
        "role":     "harmful",
        "coef":     -0.328,
        "min_val":  5.0, "max_val": 10.0, "default": 7.50,
        "step":     0.01,
        "info":     "Urea cycle & nitric oxide synthesis; arginine "
                    "metabolism dysregulated in AD.",
    },
    "11764118_at": {
        "symbol":   "Unchar",
        "name":     "Uncharacterised transcript (chr12q15)",
        "role":     "harmful",
        "coef":     -0.462,
        "min_val":  2.0, "max_val": 7.0, "default": 4.00,
        "step":     0.01,
        "info":     "Putative lncRNA; no HGNC symbol. "
                    "Biological function unknown — requires RNA-seq validation.",
    },
}

COHORT_MEANS = {p: GENE_META[p]["default"] for p in GENE_ORDER}
COHORT_SDS   = {p: (GENE_META[p]["max_val"] - GENE_META[p]["min_val"]) / 6
                for p in GENE_ORDER}

# ── Load model ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "six_gene_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()


def predict_from_expression(expr_values: list) -> dict:
    """
    Run the pipeline and return prediction + SHAP contributions.
    Falls back to manual linear computation if pkl not loaded.
    """
    X = np.array(expr_values, dtype=float).reshape(1, -1)

if model is not None:
    delta_mmse = float(model.predict(X)[0])
    scaler     = model[0]
    X_scaled   = scaler.transform(X)[0]
    else:
        # Manual standardisation using cohort parameters
        X_scaled   = np.array([
            (expr_values[i] - COHORT_MEANS[p]) / max(COHORT_SDS[p], 0.01)
            for i, p in enumerate(GENE_ORDER)
        ])
        coefs      = np.array([GENE_META[p]["coef"] for p in GENE_ORDER])
        delta_mmse = float(np.dot(X_scaled, coefs))

    coefs  = np.array([GENE_META[p]["coef"] for p in GENE_ORDER])
    contribs = X_scaled * coefs

    return {
        "delta_mmse": delta_mmse,
        "contributions": {
            GENE_META[p]["symbol"]: float(contribs[i])
            for i, p in enumerate(GENE_ORDER)
        },
        "scaled_values": {
            GENE_META[p]["symbol"]: float(X_scaled[i])
            for i, p in enumerate(GENE_ORDER)
        },
    }


def get_risk(delta_mmse: float) -> dict:
    if delta_mmse <= -2.0:
        return {"label": "HIGH RISK",    "code": "high",
                "color": "#E24B4A", "emoji": "🔴",
                "action": "Priority clinical review in 3 months. "
                           "Consider trial enrolment and carer support."}
    if delta_mmse < 0.0:
        return {"label": "MODERATE RISK", "code": "moderate",
                "color": "#EF9F27", "emoji": "🟡",
                "action": "Standard monitoring. Review in 6 months. "
                           "Consider cognitive rehabilitation referral."}
    return {"label": "LOW RISK",     "code": "low",
            "color": "#1C7C5A", "emoji": "🟢",
            "action": "Annual review appropriate. "
                       "Reinforce lifestyle protective factors."}


def shap_bar_chart(contributions: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F9FAFC")
    genes  = list(contributions.keys())
    vals   = list(contributions.values())
    colors = ["#E24B4A" if v < 0 else "#1B6CA8" for v in vals]
    bars   = ax.barh(genes, vals, color=colors, alpha=0.85,
                     edgecolor="white", height=0.55)
    for bar, val in zip(bars, vals):
        cx = bar.get_y() + bar.get_height() / 2
        if val < 0:
            ax.text(val - 0.005, cx, f"{val:.3f}",
                    va="center", ha="right", fontsize=9, fontweight="bold")
        else:
            ax.text(val + 0.005, cx, f"{val:.3f}",
                    va="center", ha="left",  fontsize=9, fontweight="bold")
    ax.axvline(0, color="#444444", lw=1.2, alpha=0.7)
    ax.set_xlabel("SHAP contribution to ΔMMSE prediction", fontsize=9)
    ax.set_title("Per-probe SHAP contributions", fontsize=10,
                 fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.grid(True, alpha=0.18, lw=0.5, axis="x")
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 PyImpetus-SHAP</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Six-Probe Blood Transcriptomic Predictor for '
    'Alzheimer Disease Cognitive Decline (12-month ΔMMSE)<br>'
    '<small>King Abdulaziz University · ADNI-GO · '
    'LOOCV MAE=1.39 · R²=0.25</small>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Gene Expression Input")
    st.markdown(
        "Enter log₂ RMA-normalised expression values for each probe.  \n"
        "**Default = cohort mean** (ADNI-GO, n=96)."
    )
    st.markdown("---")

    expr_vals = []
    for probe_id in GENE_ORDER:
        meta = GENE_META[probe_id]
        sym  = meta["symbol"]
        role = meta["role"]
        icon = "🔴" if role == "harmful" else "🟢"

        with st.expander(f"{icon} {sym} — {meta['name']}", expanded=True):
            st.caption(meta["info"])
            val = st.slider(
                label       = f"{sym} expression (log₂ AU)",
                min_value   = float(meta["min_val"]),
                max_value   = float(meta["max_val"]),
                value       = float(meta["default"]),
                step        = meta["step"],
                key         = f"slider_{probe_id}",
            )
            dev = val - meta["default"]
            if abs(dev) < 0.05:
                st.caption("📍 At cohort mean (neutral)")
            elif (dev > 0 and role == "harmful") or \
                 (dev < 0 and role == "protective"):
                st.markdown(
                    f"<span style='color:#E24B4A'>⚠️ {abs(dev):.2f} "
                    f"above mean → increases risk</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span style='color:#1C7C5A'>✅ {abs(dev):.2f} "
                    f"{'above' if dev>0 else 'below'} mean → reduces risk</span>",
                    unsafe_allow_html=True,
                )
            expr_vals.append(val)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Cognitive Trajectory",
                             type="primary", use_container_width=True)
    reset_btn   = st.button("↺ Reset to cohort means",
                             use_container_width=True)

    if reset_btn:
        for probe_id in GENE_ORDER:
            st.session_state[f"slider_{probe_id}"] = \
                float(GENE_META[probe_id]["default"])
        st.rerun()

# ── Main content ──────────────────────────────────────────────────
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown("### 📊 Model Information")

    info_cols = st.columns(3)
    with info_cols[0]:
        st.markdown(
            '<div class="metric-box"><b>LOOCV MAE</b><br>'
            '<span style="font-size:1.6rem;color:#1B4F8A">1.388</span>'
            '</div>', unsafe_allow_html=True)
    with info_cols[1]:
        st.markdown(
            '<div class="metric-box"><b>LOOCV R²</b><br>'
            '<span style="font-size:1.6rem;color:#1B4F8A">0.247</span>'
            '</div>', unsafe_allow_html=True)
    with info_cols[2]:
        st.markdown(
            '<div class="metric-box"><b>Training n</b><br>'
            '<span style="font-size:1.6rem;color:#1B4F8A">96</span>'
            '</div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("#### Six-Probe Panel Summary")
    for probe_id in GENE_ORDER:
        meta = GENE_META[probe_id]
        sym  = meta["symbol"]
        role = meta["role"]
        coef = meta["coef"]
        tag  = "🔴 Harmful"  if role == "harmful" else "🟢 Protective"
        st.markdown(
            f"**{sym}** &nbsp;|&nbsp; Coef: `{coef:+.3f}` &nbsp;|&nbsp; {tag}  \n"
            f"<small>{meta['name']}</small>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.markdown("---")
    st.markdown("#### 📖 How to interpret")
    st.markdown("""
- **ΔMMSE < −2** → High Risk (rapid 12-month decline)
- **−2 < ΔMMSE < 0** → Moderate Risk (mild decline)
- **ΔMMSE ≥ 0** → Low Risk (stable or improving)

SHAP bars show each probe's contribution:
- **Red bars pointing left** → gene increases risk score
- **Blue bars pointing right** → gene (SNX5) decreases risk score
""")

with col2:
    st.markdown("### 🎯 Prediction Result")

    if predict_btn or "last_result" in st.session_state:
        result = predict_from_expression(expr_vals)
        st.session_state["last_result"] = result
        risk   = get_risk(result["delta_mmse"])

        # Risk category card
        cls = f"risk-{risk['code']}"
        st.markdown(
            f'<div class="{cls}">'
            f'<span style="font-size:2rem">{risk["emoji"]}</span><br>'
            f'<span style="font-size:1.6rem;font-weight:800;'
            f'color:{risk["color"]}">{risk["label"]}</span><br><br>'
            f'<b>Predicted ΔMMSE:</b> '
            f'<span style="font-size:1.4rem;font-weight:700;'
            f'color:{risk["color"]}">{result["delta_mmse"]:+.3f}</span><br>'
            f'<small>(12-month change in MMSE score)</small><br><br>'
            f'<i>{risk["action"]}</i>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        # SHAP bar chart
        fig_shap = shap_bar_chart(result["contributions"])
        st.pyplot(fig_shap, use_container_width=True)
        plt.close("all")

        # Detailed contributions table
        st.markdown("#### Detailed SHAP Contributions")
        rows = []
        for probe_id in GENE_ORDER:
            meta   = GENE_META[probe_id]
            sym    = meta["symbol"]
            raw    = expr_vals[GENE_ORDER.index(probe_id)]
            scaled = result["scaled_values"][sym]
            contrib = result["contributions"][sym]
            rows.append({
                "Gene":        sym,
                "Role":        meta["role"].capitalize(),
                "Raw value":   f"{raw:.3f}",
                "Scaled (z)":  f"{scaled:+.3f}",
                "SHAP contrib":f"{contrib:+.4f}",
                "Direction":   "↑ Risk" if contrib < 0 else "↓ Risk",
            })

        import pandas as pd
        df_disp = pd.DataFrame(rows)
        st.dataframe(df_disp, use_container_width=True, hide_index=True)

    else:
        st.info(
            "👈 Adjust the gene expression sliders in the sidebar "
            "and click **Predict Cognitive Trajectory** to see the result."
        )

# ── Disclaimer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="disclaimer">'
    '<b>⚠️ Research Prototype Disclaimer:</b> '
    'This tool is for research demonstration only. '
    'It is not validated for clinical use and has not received regulatory approval. '
    'The model was trained on ADNI-GO (n=96, North American cohort). '
    'External longitudinal validation in diverse populations is required '
    'before any clinical deployment. '
    'Do not use this tool to make clinical decisions.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    "<br><center><small>Syed AH et al. | King Abdulaziz University | "
    "BioData Mining (submitted 2026) | "
    "<a href='https://github.com/SAH-ML/pyimpetus-shap-ad' "
    "target='_blank'>GitHub</a></small></center>",
    unsafe_allow_html=True,
)
