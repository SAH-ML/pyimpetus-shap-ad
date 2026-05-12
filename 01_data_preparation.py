"""
PyImpetus-SHAP Pipeline — Step 1: Data Preparation
====================================================
Constructs the final analytical dataset (n=96) for predicting
12-month MMSE change (ΔMMSE) from whole-blood Affymetrix HG-U219
microarray data in ADNI-GO participants.

Inputs  : ADNI_Gene_Expression_Profile.csv, All_Subjects_MMSE_12Mar2026.csv,
           All_Subjects_FHQ_05Mar2026.csv, All_Subjects_APOERES_05Mar2026.csv,
           DXSUM.csv, All_Subjects_CDR_12Mar2026.csv, ADAS.csv,
           All_Subjects_PTDEMOG_05Mar2026.csv
Output  : ADNI_Gene_Expression_Final_96_clean.csv

Authors : Asif Hassan Syed et al., King Abdulaziz University
License : MIT
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────
GENE_FILE   = "ADNI_Gene_Expression_Profile.csv"
MMSE_FILE   = "All_Subjects_MMSE_12Mar2026.csv"
FHQ_FILE    = "All_Subjects_FHQ_05Mar2026.csv"
APOE_FILE   = "All_Subjects_APOERES_05Mar2026.csv"
DX_FILE     = "DXSUM.csv"
CDR_FILE    = "All_Subjects_CDR_12Mar2026.csv"
ADAS_FILE   = "ADAS.csv"
DEMO_FILE   = "All_Subjects_PTDEMOG_05Mar2026.csv"
OUTPUT_FILE = "ADNI_Gene_Expression_Final_96_clean.csv"

DELTA_MONTHS = 12   # prediction horizon
TOLERANCE    = 3    # ±3 months tolerance for future MMSE search

VISIT_TO_MONTHS = {
    "bl": 0, "m03": 3, "m06": 6, "m12": 12, "m18": 18,
    "m24": 24, "m36": 36, "m48": 48, "m60": 60,
    "v02": 2, "v03": 3, "v04": 4, "v05": 5, "v06": 6, "v11": 11,
}

# ── Step 1: Load and reshape gene expression data ─────────────────
print("[1/7] Loading gene expression data …")
df_raw = pd.read_csv(GENE_FILE, on_bad_lines="skip")
df_raw = df_raw.set_index("Phase").T.reset_index()
df_raw.rename(columns={"index": "SampleID"}, inplace=True)
df_raw = df_raw[~df_raw["SampleID"].isin(["Unnamed: 1", "Unnamed: 2"])]

# Deduplicate column names
cols, counts, unique_cols = pd.Series(df_raw.columns), {}, []
for item in cols:
    counts[item] = counts.get(item, 0)
    suffix = f".{counts[item]}" if counts[item] > 0 else ""
    unique_cols.append(f"{item}{suffix}")
    counts[item] += 1
df_raw.columns = unique_cols

df_raw["RID"] = (
    pd.to_numeric(
        df_raw["SubjectID"].astype(str).str.split("_").str[-1],
        errors="coerce",
    ).astype("Int64")
)
df_raw.rename(columns={"Visit": "VISCODE2"}, inplace=True)

META = ["Phase", "VISCODE2", "SubjectID", "260/280", "260/230",
        "RIN", "Affy Plate", "YearofCollection", "RID"]
meta_present = [c for c in META if c in df_raw.columns]
expr_cols = [c for c in df_raw.columns
             if c not in meta_present and c != "ProbeSet"]

gene_df = df_raw[meta_present + expr_cols].copy()
for col in expr_cols + ["260/280", "260/230", "RIN", "Affy Plate", "YearofCollection"]:
    if col in gene_df.columns:
        gene_df[col] = pd.to_numeric(gene_df[col], errors="coerce")

print(f"   Gene expression shape after reshape: {gene_df.shape}")

# ── Step 2: Align MMSE scores and compute ΔMMSE ───────────────────
print("[2/7] Merging MMSE scores and computing ΔMMSE …")
mmse = pd.read_csv(MMSE_FILE)
mmse = mmse[["RID", "VISCODE2", "MMSCORE"]].drop_duplicates()
mmse["RID"] = pd.to_numeric(mmse["RID"], errors="coerce").astype("Int64")
mmse["VISCODE2"] = mmse["VISCODE2"].astype(str).str.strip()
mmse["months"] = mmse["VISCODE2"].map(VISIT_TO_MONTHS)
mmse = mmse.dropna(subset=["months"])
mmse["months"] = mmse["months"].astype(int)

gene_df["months_gene"] = gene_df["VISCODE2"].map(VISIT_TO_MONTHS)
gene_df = gene_df.dropna(subset=["months_gene"])
gene_df["months_gene"] = gene_df["months_gene"].astype(int)

# Merge current MMSE
gene_df = gene_df.merge(
    mmse[["RID", "VISCODE2", "MMSCORE"]],
    on=["RID", "VISCODE2"], how="left"
)
gene_df.rename(columns={"MMSCORE": "MMSE_current"}, inplace=True)


def add_future_mmse(df, mmse_tab, delta, tol=TOLERANCE):
    """Find the closest future MMSE within ±tol months of delta horizon."""
    fut_scores, fut_months = [], []
    for _, row in df.iterrows():
        rid, gmo = row["RID"], row["months_gene"]
        fut = mmse_tab[
            (mmse_tab["RID"] == rid) & (mmse_tab["months"] > gmo)
        ].copy()
        if fut.empty:
            fut_scores.append(np.nan)
            fut_months.append(np.nan)
            continue
        target = gmo + delta
        fut["dist"] = (fut["months"] - target).abs()
        best = fut.loc[fut["dist"].idxmin()]
        if abs(best["months"] - target) <= tol:
            fut_scores.append(best["MMSCORE"])
            fut_months.append(best["months"])
        else:
            fut_scores.append(np.nan)
            fut_months.append(np.nan)
    df[f"MMSE_{delta}m_future"] = fut_scores
    df[f"future_months_{delta}m"] = fut_months
    return df


gene_df = add_future_mmse(gene_df, mmse, DELTA_MONTHS)
gene_df["Delta_MMSE"] = (
    gene_df[f"MMSE_{DELTA_MONTHS}m_future"] - gene_df["MMSE_current"]
)
gene_df = gene_df.dropna(
    subset=[f"MMSE_{DELTA_MONTHS}m_future", "MMSE_current"]
).copy()
print(f"   Subjects with paired MMSE: {gene_df.shape[0]}")

# ── Step 3: Remove duplicate RIDs ─────────────────────────────────
print("[3/7] Removing duplicate RIDs …")
gene_df = gene_df.drop_duplicates(subset=["RID"], keep="first")
print(f"   Unique subjects after deduplication: {gene_df.shape[0]}")

main_df = gene_df.copy()

# ── Step 4: Merge auxiliary tables ────────────────────────────────
print("[4/7] Merging clinical and demographic tables …")

# 4.1 Family history
fhq = pd.read_csv(FHQ_FILE)
fhq = fhq[["RID", "FHQMOM", "FHQDAD", "FHQSIB"]].copy()
fhq["RID"] = pd.to_numeric(fhq["RID"], errors="coerce").astype("Int64")
fhq.dropna(subset=["RID"], inplace=True)
for c in ["FHQMOM", "FHQDAD", "FHQSIB"]:
    fhq[c] = pd.to_numeric(fhq[c], errors="coerce")
fhq["FH_any"] = (
    (fhq["FHQMOM"] > 0) | (fhq["FHQDAD"] > 0) | (fhq["FHQSIB"] > 0)
).astype(int)
fhq_agg = fhq.groupby("RID")["FH_any"].max().reset_index()
main_df = main_df.merge(fhq_agg, on="RID", how="left")

# 4.2 APOE genotyping
apoe = pd.read_csv(APOE_FILE)
apoe = apoe[["RID", "GENOTYPE"]].drop_duplicates()
apoe["RID"] = pd.to_numeric(apoe["RID"], errors="coerce").astype("Int64")
apoe.dropna(subset=["RID"], inplace=True)


def count_e4(gt):
    if pd.isna(gt):
        return np.nan
    try:
        a1, a2 = map(int, gt.split("/"))
        return (a1 == 4) + (a2 == 4)
    except Exception:
        return np.nan


apoe["APOE4_count"] = apoe["GENOTYPE"].apply(count_e4)
apoe["APOE4"] = (apoe["APOE4_count"] > 0).astype(int)
apoe = apoe.dropna(subset=["APOE4_count"])[["RID", "APOE4", "APOE4_count"]]
main_df = main_df.merge(apoe, on="RID", how="left")

# 4.3 Diagnosis
dx = pd.read_csv(DX_FILE)
dx = dx[["RID", "VISCODE2", "DIAGNOSIS"]].drop_duplicates()
dx["RID"] = pd.to_numeric(dx["RID"], errors="coerce").astype("Int64")
dx["VISCODE2"] = dx["VISCODE2"].astype(str).str.strip()
dx.dropna(subset=["RID", "VISCODE2"], inplace=True)
main_df = main_df.merge(dx, on=["RID", "VISCODE2"], how="left")
main_df = pd.get_dummies(main_df, columns=["DIAGNOSIS"], prefix="DX")

# 4.4 CDR sum of boxes
cdr = pd.read_csv(CDR_FILE)
cdr = cdr[["RID", "VISCODE2", "CDRSB"]].drop_duplicates()
cdr["RID"] = pd.to_numeric(cdr["RID"], errors="coerce").astype("Int64")
cdr["VISCODE2"] = cdr["VISCODE2"].astype(str).str.strip()
cdr.dropna(subset=["RID", "VISCODE2"], inplace=True)
main_df = main_df.merge(cdr, on=["RID", "VISCODE2"], how="left")

# 4.5 ADAS-Cog
adas = pd.read_csv(ADAS_FILE)
adas = adas[["RID", "VISCODE2", "TOTAL13"]].drop_duplicates()
adas["RID"] = pd.to_numeric(adas["RID"], errors="coerce").astype("Int64")
adas["VISCODE2"] = adas["VISCODE2"].astype(str).str.strip()
adas.dropna(subset=["RID", "VISCODE2"], inplace=True)
main_df = main_df.merge(adas, on=["RID", "VISCODE2"], how="left")

# 4.6 Demographics
demo = pd.read_csv(DEMO_FILE)
demo = demo[["RID", "VISCODE2", "PTGENDER", "PTEDUCAT", "PTDOB", "VISDATE"]].copy()
demo["RID"] = pd.to_numeric(demo["RID"], errors="coerce").astype("Int64")
demo = demo[demo["VISCODE2"] == "bl"]
demo = demo.groupby("RID").first().reset_index()
demo["DOB"] = pd.to_datetime(
    "15/" + demo["PTDOB"], format="%d/%m/%Y", errors="coerce"
)
demo["BL_DATE"] = pd.to_datetime(demo["VISDATE"], errors="coerce")
demo["AGE_BL"] = (demo["BL_DATE"] - demo["DOB"]).dt.days / 365.25
demo = demo.dropna(subset=["AGE_BL"])
demo["PTGENDER"] = demo["PTGENDER"].round(0).astype("Int64")
demo["PTEDUCAT"] = demo["PTEDUCAT"].round(0).astype("Int64")
main_df = main_df.merge(
    demo[["RID", "PTGENDER", "PTEDUCAT", "AGE_BL"]], on="RID", how="left"
)
main_df["AGE_AT_VISIT"] = (
    main_df["AGE_BL"] + main_df["months_gene"] / 12
)
main_df["AGE_AT_VISIT"] = main_df["AGE_AT_VISIT"].round(0).astype("Int64")
main_df["AGE_BL"] = main_df["AGE_BL"].round(0).astype("Int64")

# ── Step 5: Drop empty columns ────────────────────────────────────
print("[5/7] Dropping empty columns …")
empty_cols = [c for c in main_df.columns if main_df[c].isnull().all()]
main_df.drop(columns=empty_cols, errors="ignore", inplace=True)
print(f"   Dropped {len(empty_cols)} empty columns: {empty_cols}")

# ── Step 6: Impute missing clinical values ─────────────────────────
print("[6/7] Imputing missing clinical values …")
for col in ["TOTAL13", "CDRSB", "RIN"]:
    n_miss = main_df[col].isnull().sum() if col in main_df.columns else 0
    if n_miss > 0:
        main_df[col] = main_df[col].fillna(main_df[col].median())
        print(f"   Imputed {n_miss} missing value(s) in {col} with median.")

print("   Missing after imputation:")
check_cols = [c for c in ["TOTAL13", "CDRSB", "RIN"] if c in main_df.columns]
print(f"   {main_df[check_cols].isnull().sum().to_dict()}")

# ── Step 7: Save ──────────────────────────────────────────────────
print("[7/7] Saving final dataset …")
main_df.to_csv(OUTPUT_FILE, index=False)
print(f"   Saved: {OUTPUT_FILE}")
print(f"   Final shape: {main_df.shape}")
print(f"   Target (Delta_MMSE) range: {main_df['Delta_MMSE'].min():.1f} to {main_df['Delta_MMSE'].max():.1f}")
print("\n✓ Data preparation complete.")
