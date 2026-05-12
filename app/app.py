"""
PyImpetus-SHAP Pipeline — Clinical Decision Support Tool
=========================================================
Flask REST API serving six-probe Alzheimer disease cognitive
decline predictions with per-probe SHAP contributions and
risk stratification.

Usage:
    python app.py

Endpoints:
    GET  /health           → model status check
    GET  /gene_info        → probe metadata and typical ranges
    POST /predict          → prediction + risk + SHAP contributions

Authors : Asif Hassan Syed et al., King Abdulaziz University
License : MIT
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)   # allow the HTML frontend (any origin) to call the API

# ── Load pipeline once at startup ────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "six_gene_model.pkl")
pipeline   = joblib.load(MODEL_PATH)

# ── Gene order — MUST match the order used during model training ──
#    Loaded from six_gene_order.json at startup (see below)
import json
with open(os.path.join(os.path.dirname(__file__), "six_gene_order.json")) as f:
    GENE_ORDER = json.load(f)

# ── Gene metadata: symbol, name, role, EN coefficient, typical range ─
GENE_META = {
    "11762936_x_at": {
        "symbol":        "AQP7",
        "name":          "Aquaporin 7",
        "role":          "harmful",
        "coefficient":   -0.598,
        "typical_range": [3.25, 4.25],
        "module":        "Metabolic",
    },
    "200024_PM_at": {
        "symbol":        "RPS5",
        "name":          "Ribosomal Protein S5",
        "role":          "harmful",
        "coefficient":   -0.447,
        "typical_range": [10.7, 11.8],
        "module":        "Ribosomal",
    },
    "11762358_at": {
        "symbol":        "CHD2",
        "name":          "Chromodomain Helicase DNA Binding Protein 2",
        "role":          "harmful",
        "coefficient":   -0.293,
        "typical_range": [7.5, 9.0],
        "module":        "Epigenetic",
    },
    "11763188_a_at": {
        "symbol":        "SNX5",
        "name":          "Sorting Nexin 5 (retromer component)",
        "role":          "protective",
        "coefficient":   +0.441,
        "typical_range": [7.8, 8.8],
        "module":        "Retromer",
    },
    "11757278_x_at": {
        "symbol":        "ASS1",
        "name":          "Argininosuccinate Synthase 1",
        "role":          "harmful",
        "coefficient":   -0.328,
        "typical_range": [6.5, 8.5],
        "module":        "Metabolic",
    },
    "11764118_at": {
        "symbol":        "Unchar",
        "name":          "Uncharacterised transcript (chr12q15)",
        "role":          "harmful",
        "coefficient":   -0.462,
        "typical_range": [3.0, 5.0],
        "module":        "Non-coding",
    },
}

# Elastic Net coefficients in GENE_ORDER sequence
COEFS = [GENE_META[p]["coefficient"] for p in GENE_ORDER]


# ── Risk stratification ───────────────────────────────────────────
def get_risk_category(delta_mmse: float) -> dict:
    """Convert predicted ΔMMSE to a clinical risk category."""
    if delta_mmse <= -2.0:
        return {
            "category":   "High Risk",
            "code":       "HIGH",
            "color":      "#E24B4A",
            "description": (
                "Predicted rapid cognitive decline (ΔMMSE ≤ −2 over 12 months). "
                "Consider prioritising for clinical trial enrolment, carer support "
                "planning, and intensive neuropsychological monitoring."
            ),
            "monitoring": "Recommended review: 3 months",
            "threshold":  "ΔMMSE ≤ −2.0",
        }
    if delta_mmse < 0.0:
        return {
            "category":   "Moderate Risk",
            "code":       "MODERATE",
            "color":      "#EF9F27",
            "description": (
                "Predicted mild cognitive decline (−2 < ΔMMSE < 0 over 12 months). "
                "Recommend standard monitoring protocol, cognitive rehabilitation "
                "referral, and lifestyle intervention discussion."
            ),
            "monitoring": "Recommended review: 6 months",
            "threshold":  "−2.0 < ΔMMSE < 0",
        }
    return {
        "category":   "Low Risk",
        "code":       "LOW",
        "color":      "#378ADD",
        "description": (
            "Predicted cognitive stability or improvement over 12 months. "
            "Standard annual follow-up is appropriate."
        ),
        "monitoring": "Recommended review: 12 months (annual)",
        "threshold":  "ΔMMSE ≥ 0",
    }


# ── Per-probe SHAP contributions ─────────────────────────────────
def compute_contributions(X_raw: np.ndarray) -> list:
    """
    Approximate per-probe SHAP contribution as coef × scaled_value.
    Uses the pipeline's StandardScaler for exact standardisation.
    """
    scaler   = pipeline.named_steps["standardscaler"]
    X_scaled = scaler.transform(X_raw)[0]

    contributions = []
    for i, probe_id in enumerate(GENE_ORDER):
        meta    = GENE_META[probe_id]
        scaled  = float(X_scaled[i])
        contrib = float(scaled * COEFS[i])
        contributions.append({
            "probe_id":   probe_id,
            "symbol":     meta["symbol"],
            "name":       meta["name"],
            "role":       meta["role"],
            "module":     meta["module"],
            "coefficient":         COEFS[i],
            "raw_value":           round(float(X_raw[0][i]), 4),
            "scaled_value":        round(scaled, 4),
            "contribution":        round(contrib, 4),
            "direction": (
                "increasing risk" if contrib < 0 else "decreasing risk"
            ),
        })
    return contributions


# ── POST /predict ─────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received."}), 400
    if "expression" not in data:
        return jsonify({"error": "Missing key: expression"}), 400

    expr = data["expression"]
    if len(expr) != 6:
        return jsonify({
            "error": f"Expected 6 values in probe order {GENE_ORDER}, "
                     f"received {len(expr)}."
        }), 400

    try:
        expr = [float(v) for v in expr]
    except (ValueError, TypeError):
        return jsonify({"error": "All expression values must be numeric."}), 400

    # Calibration warnings
    warnings_list = []
    for i, (probe_id, val) in enumerate(zip(GENE_ORDER, expr)):
        lo, hi = GENE_META[probe_id]["typical_range"]
        if val < lo - 2 or val > hi + 2:
            sym = GENE_META[probe_id]["symbol"]
            warnings_list.append(
                f"{sym} ({probe_id}): value {val:.3f} is outside "
                f"expected range [{lo}, {hi}]. Verify assay calibration."
            )

    # Prediction — pipeline handles scaling internally
    X_raw      = np.array(expr, dtype=float).reshape(1, -1)
    delta_mmse = float(pipeline.predict(X_raw)[0])

    # Composite risk score (for display)
    scaler     = pipeline.named_steps["standardscaler"]
    X_scaled   = scaler.transform(X_raw)[0]
    risk_score = float(np.dot(X_scaled, COEFS))

    risk         = get_risk_category(delta_mmse)
    contributions = compute_contributions(X_raw)

    return jsonify({
        "predicted_delta_mmse":   round(delta_mmse, 3),
        "composite_risk_score":   round(risk_score, 3),
        "risk_stratification":    risk,
        "gene_contributions":     contributions,
        "thresholds": {
            "high":     "ΔMMSE ≤ −2.0 (rapid decline)",
            "moderate": "−2.0 < ΔMMSE < 0 (mild decline)",
            "low":      "ΔMMSE ≥ 0 (stable or improving)",
        },
        "calibration_warnings":   warnings_list,
        "gene_order_used":        GENE_ORDER,
        "model_info": {
            "training_cohort":            "ADNI-GO (n=96)",
            "platform":                   "Affymetrix HG-U219",
            "outcome":                    "12-month ΔMMSE (month 60 − month 48)",
            "loocv_mae":                  1.388,
            "loocv_rmse":                 2.004,
            "loocv_r2":                   0.247,
            "cross_platform_replication": "SNX5 replicated in AddNeuroMed GSE63060 (p=0.002)",
        },
        "disclaimer": (
            "Research prototype only. Not validated for clinical use or "
            "regulatory approval. External longitudinal validation required."
        ),
    })


# ── GET /gene_info ────────────────────────────────────────────────
@app.route("/gene_info", methods=["GET"])
def gene_info():
    """Return gene metadata for the frontend to build input sliders."""
    return jsonify({
        "gene_order":  GENE_ORDER,
        "genes":       GENE_META,
        "input_units": "log2 RMA-normalised expression (Affymetrix HG-U219)",
    })


# ── GET /health ───────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": True,
        "gene_count":   len(GENE_ORDER),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
