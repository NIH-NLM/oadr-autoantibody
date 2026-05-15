"""
Shared data loaders for OADR autoantibody work.

Two feature panels are exposed:

    Panel A (legacy, 9 features) — matches what the existing CNN/Ridge/RF
    notebooks were trained on. Columns:
        MIAA, GAD65, IA2IC, ICA, ZNT8, 8-12, 13-17, >18, Sex
    Target: log(C_Peptide_AUC_4Hrs).
    Studies: SDY524, SDY569, SDY797, SDY1737  (N=154 in principle; 150
    after intersecting tidy + c-peptide subject IDs).

    Panel B (Jeff extended) — adds BMI, height, weight, exact age (years),
    disease duration (years since T1D diagnosis), one-hot race/ethnicity/
    cohort. Studies: SDY524, SDY569, SDY1737 only (no Jeff data for
    SDY797/SDY1625). Per-study autoantibody composition varies.

The per-study source CSVs have inconsistent column names (SDY524 uses
'ImmPort Accession' and 'IA_2ic'; SDY569 has typo 'Subject_IDel'; ages
are coded as '18-30'/'>30' in newer tidy files vs '>18' in cleaned
files). All of that is normalized here so notebooks can stay clean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Resolve repo root from this file's location: modules/oadr_data.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA = _REPO_ROOT / "data"

PANEL_A_FEATURES = [
    "MIAA", "GAD65", "IA2IC", "ICA", "ZNT8",
    "8-12", "13-17", ">18", "Sex",
]
PANEL_A_TARGET = "log_auc"
PANEL_A_RAW_TARGET = "C_Peptide_AUC_4Hrs"

PANEL_A_STUDIES = ["SDY524", "SDY569", "SDY797", "SDY1737"]
PANEL_B_STUDIES = ["SDY524", "SDY569", "SDY1737"]


def _normalize_property(p: str) -> str:
    """Map per-study property labels to canonical names."""
    p = p.strip()
    if p.upper() == "IA_2IC":
        return "IA2IC"
    return p.upper() if p.upper() in {"GAD65", "IA2IC", "MIAA", "ICA", "ZNT8"} else p


def _normalize_age_group(a: str) -> str:
    """Map age-group strings into the legacy {'8-12', '13-17', '>18'} schema."""
    if a in ("18-30", ">30", ">18"):
        return ">18"
    return a


def _read_cpeptide(study: str) -> pd.DataFrame:
    """Load c-peptide AUC tidy file; normalize columns to (Subject_ID, C_Peptide_AUC_4Hrs)."""
    df = pd.read_csv(_DATA / f"{study}_cpeptide_auc_tidy.csv")
    rename = {
        "ImmPort Accession": "Subject_ID",
        "Subject_IDel": "Subject_ID",  # SDY569 typo
        "C_Peptide_AUC": "C_Peptide_AUC_4Hrs",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df[["Subject_ID", "C_Peptide_AUC_4Hrs"]]


def load_panel_a(study: str) -> pd.DataFrame:
    """Build the 9-feature panel for one study.

    Returns a DataFrame with columns:
        Subject_ID, Study, <PANEL_A_FEATURES>, C_Peptide_AUC_4Hrs, log_auc
    Antibodies absent from the study are filled with 0.0 (matching the
    convention of the existing cleaned/* CSVs).
    """
    if study not in PANEL_A_STUDIES:
        raise ValueError(f"Unknown study {study!r} for Panel A")

    feat = pd.read_csv(_DATA / f"{study}_tidy.csv")
    feat = feat.rename(columns={"Accession": "Subject_ID"})
    feat["Property"] = feat["Property"].map(_normalize_property)
    feat["Age_Group"] = feat["Age_Group"].astype(str).map(_normalize_age_group)

    wide = feat.pivot_table(
        index=["Subject_ID", "Sex"],
        columns="Property",
        values="Value",
    ).reset_index()
    wide["Sex"] = wide["Sex"].map({"Male": 0, "Female": 1}).astype(float)

    age = (
        pd.get_dummies(feat[["Subject_ID", "Age_Group"]], columns=["Age_Group"])
        .groupby("Subject_ID")
        .max()
        .reset_index()
    )
    for col in ("Age_Group_8-12", "Age_Group_13-17", "Age_Group_>18"):
        if col not in age.columns:
            age[col] = 0
    age = age.rename(columns={
        "Age_Group_8-12": "8-12",
        "Age_Group_13-17": "13-17",
        "Age_Group_>18": ">18",
    })
    wide = wide.merge(age[["Subject_ID", "8-12", "13-17", ">18"]], on="Subject_ID", how="left")

    cpep = _read_cpeptide(study)
    df = wide.merge(cpep, on="Subject_ID", how="inner")

    for ab in ("MIAA", "GAD65", "IA2IC", "ICA", "ZNT8"):
        if ab not in df.columns:
            df[ab] = 0.0
    df[PANEL_A_FEATURES] = df[PANEL_A_FEATURES].fillna(0.0).astype(float)

    df["log_auc"] = np.log(df["C_Peptide_AUC_4Hrs"])
    df["Study"] = study
    return df[["Subject_ID", "Study"] + PANEL_A_FEATURES + [PANEL_A_RAW_TARGET, PANEL_A_TARGET]]


def load_panel_a_all() -> pd.DataFrame:
    """Concatenate Panel A across all 4 studies."""
    return pd.concat([load_panel_a(s) for s in PANEL_A_STUDIES], ignore_index=True)


# ---------- Panel B (Jeff extended) ----------

# Jeff CSVs use lowercase column names; map back to PANEL_A naming for AAs.
_JEFF_AA_MAP = {
    "gad65": "GAD65",
    "ia_2ic": "IA2IC",
    "miaa": "MIAA",
    "zn_t8": "ZNT8",
}
_JEFF_STUDY_FILES = {
    "SDY524": ("aa_524.csv", "demo_524.csv"),
    "SDY569": ("aa_569.csv", "demo_569.csv"),
    "SDY1737": ("aa_1737.csv", "demo_1737.csv"),
}


def _parse_date(s):
    return pd.to_datetime(s, errors="coerce")


def load_panel_b(study: str) -> pd.DataFrame:
    """Build the Jeff-extended feature panel for one study.

    Columns returned:
        Subject_ID, Study, Sex, age_years, disease_duration_years,
        bmi, height_cm, weight_kg,
        race, ethnicity, cohort_group  (raw categoricals — caller one-hots),
        <available autoantibodies in PANEL_A_FEATURES>,
        ICA  (filled 0 — Jeff files do not measure it),
        C_Peptide_AUC_4Hrs, log_auc.
    """
    if study not in PANEL_B_STUDIES:
        raise ValueError(f"Study {study!r} has no Jeff-panel data")

    aa_file, demo_file = _JEFF_STUDY_FILES[study]
    aa = pd.read_csv(_DATA / "Jeff" / aa_file)
    demo = pd.read_csv(_DATA / "Jeff" / demo_file)

    aa = aa.rename(columns={"accession": "Subject_ID", **_JEFF_AA_MAP})
    for ab in ("GAD65", "IA2IC", "MIAA", "ZNT8", "ICA"):
        if ab not in aa.columns:
            aa[ab] = 0.0

    # Date column varies: 'date_of_screening_visit' (524, 1737) vs 'numeric_date_drawn' (569).
    if "date_of_screening_visit" in aa.columns:
        aa["assay_date"] = _parse_date(aa["date_of_screening_visit"])
    elif "numeric_date_drawn" in aa.columns:
        aa["assay_date"] = _parse_date(aa["numeric_date_drawn"])
    else:
        aa["assay_date"] = pd.NaT

    keep_aa = ["Subject_ID", "GAD65", "IA2IC", "MIAA", "ZNT8", "ICA",
               "baseline_height_cm", "baseline_weight_kg", "baseline_bmi_kg_m_2",
               "assay_date"]
    aa = aa[[c for c in keep_aa if c in aa.columns]]
    aa = aa.rename(columns={
        "baseline_height_cm": "height_cm",
        "baseline_weight_kg": "weight_kg",
        "baseline_bmi_kg_m_2": "bmi",
    })

    demo = demo.rename(columns={"accession": "Subject_ID"})
    demo["Sex"] = demo["sex"].map({"Male": 0, "Female": 1}).astype(float)
    demo["t1d_dx_date"] = _parse_date(demo["date_of_t1dm_diagnosis"])
    demo["day_0_date"] = _parse_date(demo["day_0_date"])
    demo["age_years"] = (demo["day_0_date"] - pd.to_datetime(
        demo["year_of_birth"].astype("Int64").astype(str) + "-07-01",
        errors="coerce")).dt.days / 365.25
    demo["disease_duration_years"] = (demo["day_0_date"] - demo["t1d_dx_date"]).dt.days / 365.25

    keep_demo = ["Subject_ID", "Sex", "age_years", "disease_duration_years",
                 "race", "ethnicity", "cohort_group"]
    demo = demo[[c for c in keep_demo if c in demo.columns]]

    cpep = _read_cpeptide(study)

    df = aa.merge(demo, on="Subject_ID", how="inner").merge(cpep, on="Subject_ID", how="inner")
    df["log_auc"] = np.log(df["C_Peptide_AUC_4Hrs"])
    df["Study"] = study

    cols = (["Subject_ID", "Study", "Sex", "age_years", "disease_duration_years",
             "bmi", "height_cm", "weight_kg",
             "race", "ethnicity", "cohort_group",
             "GAD65", "IA2IC", "MIAA", "ZNT8", "ICA",
             "C_Peptide_AUC_4Hrs", "log_auc"])
    return df[[c for c in cols if c in df.columns]]


def load_panel_b_all(impute_bmi: bool = True) -> pd.DataFrame:
    """Concatenate Panel B across the 3 Jeff-covered studies.

    If impute_bmi=True, fill BMI/height/weight missingness with study-median and
    add a 'bmi_missing' flag column.
    """
    df = pd.concat([load_panel_b(s) for s in PANEL_B_STUDIES], ignore_index=True)
    if impute_bmi:
        df["bmi_missing"] = df["bmi"].isna().astype(int)
        for col in ("bmi", "height_cm", "weight_kg"):
            df[col] = df.groupby("Study")[col].transform(lambda s: s.fillna(s.median()))
            df[col] = df[col].fillna(df[col].median())
    return df


def panel_b_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Turn a Panel B frame into (X, y, feature_names) with categoricals one-hot encoded.

    Note on ``cohort_group``: the raw column mixes study-specific labels (e.g.
    SDY569's internal arm codes ``1`` and ``2``) and age bins (``Adult``,
    ``Pediatric``) that are redundant with the continuous ``age_years``
    column.  Rather than one-hot encode all of them, we collapse the column
    into a single binary ``received_active_treatment`` flag derived from
    the SDY524 (AbATE) trial arms: ``hOKT3`` (the active anti-CD3 antibody
    arm) → 1, ``Control`` → 0, every other label (other studies, missing) → 0.
    This preserves the one biologically interpretable signal in the column
    (active treatment vs placebo) without leaving the design matrix with
    five fragmented ``cohort_group`` dummies, three of which are non-
    interpretable across studies.
    """
    y = df["log_auc"].astype(float)
    base_cont = ["Sex", "age_years", "disease_duration_years",
                 "bmi", "height_cm", "weight_kg",
                 "GAD65", "IA2IC", "MIAA", "ZNT8", "ICA"]
    if "bmi_missing" in df.columns:
        base_cont.append("bmi_missing")
    cont = df[[c for c in base_cont if c in df.columns]].astype(float)

    # Collapse cohort_group to a single binary "received active treatment".
    if "cohort_group" in df.columns:
        received_active = (df["cohort_group"].astype(str) == "hOKT3").astype(float)
        cont = pd.concat([cont, received_active.rename("received_active_treatment")], axis=1)

    # One-hot encode only race and ethnicity now; cohort_group is dropped.
    cat_df = df[[c for c in ("race", "ethnicity") if c in df.columns]].copy()
    for c in cat_df.columns:
        cat_df[c] = cat_df[c].astype(str).fillna("MISSING")
    cats = pd.get_dummies(cat_df, drop_first=True).astype(float)
    X = pd.concat([cont, cats], axis=1)
    return X, y, list(X.columns)


def panel_a_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_names) for Panel A — features are already numeric."""
    return df[PANEL_A_FEATURES].astype(float), df[PANEL_A_TARGET].astype(float), list(PANEL_A_FEATURES)
