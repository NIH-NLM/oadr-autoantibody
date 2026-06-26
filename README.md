# Predicting residual beta-cell function in Type 1 Diabetes, under federation

Analytical work for the **federated prediction of pancreatic beta-cell
function** from autoantibody and clinical profiles, across four independent
ImmPort clinical-study cohorts (SDY524, SDY569, SDY797, SDY1737). Presented at
BioITWorld 2026 — talk deck: [`slides/2026BioITWorld_Federated_Learning_v9.pdf`](slides/2026BioITWorld_Federated_Learning_v9.pdf),
markdown rendering aligned to the notebooks: [`SLIDES.md`](SLIDES.md).

## The biological premise

Type 1 Diabetes is an autoimmune disease in which the immune system
progressively destroys the insulin-producing **beta cells** of the pancreas.
The standard quantitative readout of how many functional beta cells remain is
the **C-peptide** released during a four-hour mixed-meal tolerance test;
C-peptide is cleaved off one-to-one with insulin and is not confounded by
injected insulin, so the area under its concentration curve (**C-peptide AUC**)
is the field-standard measure of *endogenous* beta-cell output.

The five autoantibodies in the panel — GAD65, IA-2 (IA2IC), insulin (MIAA),
ICA, ZnT8 — are immune signatures of the destruction process. The scientific
question: **how well does the autoantibody profile predict the beta-cell
function that survives the attack?**

> **Read the [Limitations](#limitations) before the results.** The short answer
> is that the autoantibody panel alone is *not* predictive; what predictive
> power exists comes largely from body-size / age demographics, which are a
> known confound of C-peptide AUC.

## Privacy-preserving federation

Each cohort comes from a different institution and cannot be centralised (IRB,
data-use governance). The institutions share **only model parameters** trained
locally; a coordinator aggregates them. No subject-level data ever moves.

This repository simulates that faithfully: every model is fit **per study**, and
only coefficient vectors (or, for random forest, the trained forest) are
combined. There is no pooling — not in fitting, not in feature scaling (each
institution min–max scales its own data), and not in scoring.

## The two-stage architecture

**Stage 1 — per-study notebooks** (`Ridge_<study>`, `Lasso_<study>`,
`RandomForest_<study>`). Each institution fits on its own subjects, LASSO
determines the features, and each writes its model artifact for the four
selected features (weight_kg, GAD65, received_active_treatment, Sex):

- LASSO / Ridge → `vectors/<study>_<method>_sel_vector.csv` (coefficients)
- Random Forest → `models/<study>_rf_sel.pkl` (the trained forest)

*(In production the LASSO-determined features become the OMOP-selected feature
set, harmonised across institutions.)*

**Stage 2 — federated notebooks** (`<method>_<study>_federated_w_SDY_coefficients`).
Each reads the **other** institutions' artifacts from disk, applies them to this
institution's own data, and produces the presentation graphics: solo vs
federated, with R² (bootstrap 95% CI) and MSE. Only artifacts cross
institutions; subject-level data never moves.

```
per-study fit ──▶ vectors/*.csv, models/*.pkl ──▶ federated notebook ──▶ figures/*.pdf
   (Stage 1)         (only parameters cross)            (Stage 2)
```

## Treatment provenance (transitive closure)

`received_active_treatment` is **not** taken from a free-text column. It is
derived by transitive closure over the ImmPort arm files in `data/arms/`:
subject → arm (`*_arm_2_subject.txt`) → treatment, where the control arm is
identified by name (placebo / control) and the treatment arm is the active-drug
arm. This corrected a real error: SDY569's extended data coded arms as `1`/`2`,
which the naive `=="hOKT3"` rule mapped to **all-untreated**; the closure
recovers its 6 treated / 4 control split. SDY1737 has only age-group arms
(Adult/Pediatric) and so has no determinable treatment.

## Key results

**A-priori power — what effect is detectable as institutions federate** (the
honest power framing; we do *not* report post-hoc power from observed R²):

![a-priori power vs N](figures/apriori_power_vs_N.png)

**LASSO feature selection (federated, Panel B excl. SDY1737)** keeps four
features — weight_kg, GAD65 (negative: more autoantibody → less C-peptide),
received_active_treatment, Sex:

![LASSO aggregation Panel B](figures/federated_lasso_coef_panelB.png)

**Federation from a single institution's view** — the small institution
(SDY569, N=10) borrowing the larger study's coefficients. Note the wide
confidence interval: this is a *transfer* result on few subjects, not a
within-institution power gain:

![Ridge SDY569 federated](figures/Ridge_SDY569_federated.png)

## Reproducing

```bash
conda env create -f environment.yml
conda activate oadr-autoantibody
jupyter lab
```

Run order: Stage-1 per-study notebooks first (they write `vectors/` and
`models/`), then the Stage-2 `*_federated_w_SDY_coefficients` notebooks (they
read those artifacts and write `figures/`). All notebooks are generated
deterministically by the scripts in `bin/notebook_builders/` (fixed seed 42).

## Notebook map

| Notebook(s) | Role |
|---|---|
| `Ridge_/Lasso_/RandomForest_SDY{524,569,797,1737}` | Stage 1 — per-institution fit, write artifacts |
| `Federated_Lasso` | federated LASSO + ADMM, feature selection |
| `{LASSO,Ridge,RandomForest}_SDY{524,569}_federated_w_SDY_coefficients` | Stage 2 — apply partners' artifacts, presentation graphics |
| `Apply_Coefficients_SDY{797,1737}` | reduced-feature analysis for the studies that cannot fully join |
| `Predicting_Residual_Beta-cell_function` | k-axis sweep across all methods (backup) |
| `c-peptide_baseline_to_tidy` | data preparation |

Superseded / exploratory notebooks are in `archive/ipynb/`.

## Limitations

These are real and material; a reviewer should see them stated plainly.

1. **`weight_kg` is a body-size / age confound, not beta-cell biology.** It is
   the dominant feature (coefficient ≈ +0.97) and correlates with C-peptide AUC
   at r = 0.67–0.93 — as do height and age. **C-peptide AUC is the standard
   AUC-mean concentration and is *not* body-size normalized** (the ImmPort
   ADCPEP files record weight as a separate column). Age-at-onset is the
   best-known predictor of residual C-peptide, and weight/height proxy it in
   pediatric cohorts. So the extended panel's predictive power comes from
   demographics, not the autoantibody profile — and this is why the model fails
   to transfer to SDY1737 (where weight r ≈ 0). The SDY1737 weights were
   cross-checked against the authentic ImmPort PHEXMSTR physical-exam file and
   match exactly, so its null weight–C-peptide relationship is a genuine
   population difference, not a data error.

2. **No post-hoc power.** Computing "achieved power" from an observed R² is
   circular and was removed. Power is reported only a-priori (detectable effect
   vs N, above).

3. **Small N — wide confidence intervals.** SDY569 has 10 subjects and 4
   predictors; its federated R² point estimate of ~0.50 carries a 95% bootstrap
   CI of roughly [−0.2, +0.8]. Read the interval, not the point.

4. **"Federation gain" is transfer, not power.** With only a small institution's
   own subjects to score on, the federated number shows the *larger* study's
   model fitting the *smaller* study — not that more N raised power within the
   small institution.

5. **`received_active_treatment` conflates different drugs** (anti-CD3 hOKT3 /
   teplizumab vs anti-CD2 alefacept) as one binary. Valid for the SDY524+SDY569
   pair (both anti-CD3); a simplification if generalised.

6. **Researcher degrees of freedom.** The L1 penalty (α = 0.008) and the cohort
   (excluding SDY1737) were chosen after observing they work; some optimism is
   baked in. No external validation cohort exists.

7. **Assay heterogeneity.** SDY797's autoantibodies are binary (positive/
   negative) rather than continuous titres, and GAD65's sign differs across
   studies — both limit clean coefficient transfer.

## Layout

```
src/oadr_data.py             shared loader (per-study quirks, panels, treatment closure)
data/arms/                   ImmPort arm files for treatment provenance
ipynb/                       active notebooks
archive/ipynb/               superseded notebooks
bin/notebook_builders/       deterministic notebook generators
vectors/ , models/           Stage-1 artifacts (only parameters)
figures/ , results/          Stage-2 outputs
slides/                      BioITWorld 2026 deck (v9 PDF) + SLIDES.md rendering
environment.yml              conda environment
```
