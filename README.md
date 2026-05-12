# PyImpetus-SHAP: A Reproducible Markov Blanket Pipeline for Sparse Blood Transcriptomic Biomarker Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Platform: Affymetrix HG-U219](https://img.shields.io/badge/Platform-Affymetrix%20HG--U219-orange)](https://adni.loni.usc.edu/)
[![ADNI Data](https://img.shields.io/badge/Data-ADNI--GO-green)](https://adni.loni.usc.edu/)

> **Applied to Alzheimer Disease Cognitive Decline Prediction**  
> Asif Hassan Syed et al. — King Abdulaziz University, Department of Computer Science  
> *Submitted to: BioData Mining (BioMed Central / Springer)*

---

## Overview

This repository contains the complete, reproducible implementation of the **PyImpetus-SHAP pipeline** — an end-to-end interpretable machine learning workflow for sparse blood transcriptomic biomarker discovery. The pipeline is applied to predicting 12-month cognitive decline (ΔMMSE) in 96 ADNI-GO participants using whole-blood Affymetrix HG-U219 microarray data (49,410 probes).

### Pipeline summary

```
49,410 Affymetrix probes
       ↓
Step 1 — Data preparation     (01_data_preparation.py)
       ↓
Step 2 — PyImpetus Markov Blanket feature selection   (02_feature_selection.py)
       ↓  4 experiments × 2 thresholds × 2 seeds → cross-threshold intersection → 6 probes
Step 3 — Elastic Net regression + LOOCV + subgroup analysis   (03_model_training.py)
       ↓  LOOCV MAE = 1.388 | R² = 0.247
Step 4 — SHAP LinearExplainer + figures   (04_shap_analysis.py)
       ↓
Step 5 — Prototype clinical REST API   (app/app.py + app/index.html)
```

### Key results

| Metric | Full 49,410-probe EN | Six-probe EN (LOOCV) |
|---|---|---|
| MAE | 1.623 ± 0.366 | **1.388** |
| RMSE | 2.113 ± 0.476 | **2.004** |
| R² | 0.098 ± 0.185 | **0.247** |

### The six-probe core panel

| Probe ID | Gene | Role | EN Coefficient |
|---|---|---|---|
| 11762936_x_at | AQP7 | Harmful | −0.598 |
| 200024_PM_at | RPS5 | Harmful | −0.447 |
| 11764118_at | Unchar (chr12q15) | Harmful | −0.462 |
| 11757278_x_at | ASS1 | Harmful | −0.328 |
| 11762358_at | CHD2 | Harmful | −0.293 |
| 11763188_a_at | **SNX5** | **Protective** | **+0.441** |

SNX5 significantly replicated in AddNeuroMed (GSE63060, n=329, Illumina HumanHT-12): Spearman r = −0.170, **p = 0.002**.

---

## Repository structure

```
pyimpetus-shap-ad/
│
├── 01_data_preparation.py       # ADNI-GO dataset construction (Step 1)
├── 02_feature_selection.py      # PyImpetus Markov Blanket experiments (Step 2)
├── 03_model_training.py         # Elastic Net training, LOOCV, subgroups (Step 3)
├── 04_shap_analysis.py          # SHAP values, Fig 4 & Fig 5 (Step 4)
│
├── six_gene_order.json          # Probe order for model input (CRITICAL)
├── six_gene_model.pkl           # Trained pipeline (StandardScaler + ElasticNet)
│
├── app/
│   ├── app.py                   # Flask REST API backend
│   ├── index.html               # HTML/JS clinical decision support frontend
│   └── requirements_app.txt     # App-specific dependencies
│
├── requirements.txt             # Full analysis dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

> **Note on data files:** Raw ADNI expression and clinical CSV files are not included in this repository. Access requires ADNI registration and execution of the Data Use Agreement at [adni.loni.usc.edu](https://adni.loni.usc.edu/).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/[SAH-ML]/pyimpetus-shap-ad.git
cd pyimpetus-shap-ad
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux / Mac
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the pipeline

Run the four steps in order. Each script is self-contained and saves its outputs to the working directory.

### Step 1 — Data preparation

```bash
python 01_data_preparation.py
```

Place all ADNI CSV files in the same directory before running. Output: `ADNI_Gene_Expression_Final_96_clean.csv`

### Step 2 — Feature selection (four experiments)

```bash
python 02_feature_selection.py
```

Runs all four PyImpetus experiments automatically. Outputs:
- `selected_features_p0.1_seed42.json`
- `selected_features_p0.1_seed123.json`
- `selected_features_p0.05_seed42.json`
- `selected_features_p0.05_seed123.json`
- `six_gene_order.json`
- `feature_stability_report.txt`

### Step 3 — Model training and evaluation

```bash
python 03_model_training.py
```

Outputs: `six_gene_model.pkl`, `loocv_results.csv`, `cv_results.csv`, `subgroup_results.csv`

### Step 4 — SHAP analysis and figures

```bash
python 04_shap_analysis.py
```

Outputs: `Fig4_SHAP_Analysis.jpg/.pdf`, `Fig5_Coeff_Heatmap.jpg/.pdf`, `shap_values.csv`

---

## Clinical decision support tool (prototype)

### Start the backend

```bash
cd app
pip install -r requirements_app.txt
python app.py
# → Running on http://0.0.0.0:5000
```

### Open the frontend

Open `app/index.html` in any browser. Enter the six probe expression values using the sliders and click **Predict cognitive trajectory**.

### API reference

**POST** `/predict`

Request body:
```json
{
    "expression": [3.85, 11.20, 8.10, 8.20, 7.40, 4.10]
}
```
Values must be in the probe order defined in `six_gene_order.json`:
`[11762936_x_at, 200024_PM_at, 11762358_at, 11763188_a_at, 11757278_x_at, 11764118_at]`

Response includes: `predicted_delta_mmse`, `composite_risk_score`, `risk_stratification` (High/Moderate/Low with clinical recommendation), `gene_contributions` (per-probe SHAP), `calibration_warnings`.

**GET** `/gene_info` — Returns probe metadata and typical expression ranges.

**GET** `/health` — Returns model load status.

> **Disclaimer:** This prototype is for research demonstration only. It is not validated for clinical use or regulatory approval. External longitudinal validation is required before any clinical deployment.

---

## Input data requirements

| File | Source | Description |
|---|---|---|
| `ADNI_Gene_Expression_Profile.csv` | ADNI | Affymetrix HG-U219 RMA-normalised expression |
| `All_Subjects_MMSE_12Mar2026.csv` | ADNI | MMSE scores across all visits |
| `All_Subjects_FHQ_05Mar2026.csv` | ADNI | Family history questionnaire |
| `All_Subjects_APOERES_05Mar2026.csv` | ADNI | APOE genotyping |
| `DXSUM.csv` | ADNI | Diagnosis at each visit |
| `All_Subjects_CDR_12Mar2026.csv` | ADNI | Clinical Dementia Rating |
| `ADAS.csv` | ADNI | ADAS-Cog 13-item total |
| `All_Subjects_PTDEMOG_05Mar2026.csv` | ADNI | Demographics |

All files are available from [adni.loni.usc.edu](https://adni.loni.usc.edu/) upon registration.

---

## Cross-platform validation

The AddNeuroMed dataset (GSE63060, n=329) used for cross-platform validation is publicly available from NCBI GEO:
[https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63060](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63060)

---

## Citation

If you use this code or pipeline in your research, please cite:

```
Syed AH, et al. PyImpetus-SHAP: A Reproducible Markov Blanket Pipeline for
Sparse Blood Transcriptomic Biomarker Discovery with Cell-type Deconvolution
and Cross-Platform Validation, Applied to Alzheimer Disease Cognitive Decline.
BioData Mining. 2026. [DOI to be assigned]
```

---

## Acknowledgements

Data collection and sharing for this project were funded by the Alzheimer Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904). This work was supported by the Deanship of Scientific Research (DSR), King Abdulaziz University, through the Indexed Publications Program (IPP) 2026.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
