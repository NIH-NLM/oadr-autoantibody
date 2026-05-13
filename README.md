# Predicting residual beta-cell function in Type 1 Diabetes, under federation

This repository contains the analytical work behind a BioITWorld talk on
**federated prediction of pancreatic beta-cell function** from autoantibody
profiles, across four independent clinical study cohorts (SDY524, SDY569,
SDY797, SDY1737) plus a fifth (SDY1625) used only for the unsupervised
reconstruction comparison.

## The biological premise

Type 1 Diabetes is an autoimmune disease in which the patient's own immune
system progressively destroys the insulin-producing **beta cells** of the
pancreatic islets of Langerhans. The natural history of the disease is the
slow disappearance of those cells; in long-standing Type 1 Diabetes there are
essentially none left.

The most informative quantitative readout of how many functional beta cells a
patient still has is the **C-peptide** they release during a standardised
four-hour mixed-meal tolerance test. C-peptide is the short fragment that
gets cleaved off when proinsulin matures into insulin, so it is released into
the bloodstream in exact one-to-one proportion with insulin. Unlike insulin
itself it is not removed by the liver in the first pass and is not confounded
by injected insulin treatment, which makes the area under its four-hour
blood-concentration curve (C-peptide AUC) the field-standard measurement of
*endogenous* beta-cell output in a treated Type 1 Diabetes patient. Higher
C-peptide AUC means more surviving beta-cell mass.

The five autoantibodies in the input panel — GAD65, IA-2 (IA2IC), Insulin
(MIAA), ICA, ZnT8 — are the immune signatures of the destruction process.
Their levels rise during the immune attack on the beta cells. The natural
scientific question is therefore: **how well does the autoantibody profile
predict the beta-cell function that survives that attack?** That is what every
model in this repository is trying to answer.

## The methodological challenge: studies cannot be pooled

Each cohort comes from a different institutional clinical study with its own
data-use governance. In a real deployment those data sets cannot be moved
between institutions: privacy regulations, IRB constraints, and institutional
policies forbid it. The methodological question is therefore not *"what is the
best model when we put all the data in one place?"* but **"what is the best
model when each institution trains on its own subjects and only model
parameters are exchanged with a central coordinator?"** That is federated
learning, and it is the only framing under which the work could be deployed
in practice.

The four institutions cannot agree to centralise their data, but they can
agree to share **only** model parameters after each one has trained on its
own subjects, and to accept a central coordinator that aggregates those
parameters by taking the median value across institutions for each parameter.
This is the protocol used everywhere in this repository.

## The three arguments the talk makes

1. **Federation across more institutions improves prediction.** Adding more
   participating studies to the federation lowers the prediction error of
   every method, even when each study only trains once. The plot of
   prediction error against number of participating institutions is the
   talk's first central figure.

2. **Iterating federation across multiple rounds extracts further benefit
   until the improvement plateaus.** When each institution warm-starts its
   local model from the previous round's aggregated parameters and continues
   training, then re-federates, the prediction error continues to fall until
   the round-over-round improvement is too small to matter. The plot of
   prediction error against round number is the talk's second central
   figure.

3. **Federation lifts the cohort-size ceiling on what effects can be
   detected at all.** The most fundamental statistical message: at a fixed
   biological effect size, the probability of detecting that effect grows
   rapidly with cohort size. A single institution alone (the smallest is
   SDY569 with N = 10 subjects) cannot even fit a 9-feature regression. The
   minimum detectable effect size shrinks from R² ≥ 0.91 at N = 10 (only
   near-deterministic effects detectable) to R² ≥ 0.10 at N = 150 (a
   biologically plausible effect size that single-institution studies would
   have missed entirely). **We are not constrained by single-institution
   cohort size when we have federation.** This is the same reasoning that
   motivates multi-site clinical trials, except expressed as a computational
   protocol rather than as a centralised database.

Five prediction methods are compared, evaluated under the same federation
protocol:

| Method | Family | Interpretable? |
|---|---|---|
| Simple linear regression | Linear, no regularisation | yes (coefficients) |
| Lasso regression¹ | Linear, automatic feature selection via L1 penalty | yes (non-zero coefficients) |
| Random Forest (used here for *regression*, not classification)² | Tree ensemble, non-linear | yes (feature importance) |
| Convolutional neural network | Non-linear deep model, no pretraining | no (latent layers) |
| Convolutional network with autoencoder pretraining | Non-linear, two-stage training: unsupervised feature learning then a small supervised head | no (latent layers) |

¹ **Lasso** is an acronym for *Least Absolute Shrinkage and Selection
Operator*. It is ordinary linear regression with an additional penalty on
the absolute value of the coefficients, which has the side-effect of
pushing weak features to exactly zero. Features that end up non-zero are
the ones the data identifies as useful predictors, which makes lasso
doubly informative: it predicts *and* it selects features for you.

² **Random Forest** is most commonly associated with classification tasks,
where a forest of decision trees votes on the class label. To extract a
formal p-value from a Random Forest typically requires permuting the
labels and re-fitting the forest hundreds or thousands of times. **In this
repository the Random Forest is not used for classification or for
p-values.** It is used as a *regression* model: each leaf of each tree
predicts a continuous C-peptide value, and the forest's prediction is the
average over its trees. Random Forest is included because it provides a
non-linear but still interpretable alternative to the convolutional
networks: its built-in *feature importance* ranking tells us which
features the trees split on most often when reducing prediction error,
which is the analogue of "which features matter" for the linear methods.

## What we learned

The federated experiment matrix in
[`ipynb/federated_analysis_simulation_3x3.ipynb`](ipynb/federated_analysis_simulation_3x3.ipynb)
produces a small number of clean findings:

1. **For interpretable methods (simple linear, lasso, Random Forest),
   federation across more institutions sharply reduces prediction error.**
   Simple linear regression's prediction error falls from 1.03 at a single
   institution (averaged across all four choices of which one) to 0.33 at
   four institutions — a three-fold improvement. Lasso falls from 0.45 to
   0.24; Random Forest from 0.37 to 0.25. All three interpretable methods
   reach roughly the same federated prediction error and converge to a
   similar floor.
2. **For the naive convolutional network — no autoencoder — federation
   *hurts* rather than helps.** Single-institution prediction error is 0.36
   (the average of four solo runs) but the federated four-institution
   number is 0.41. Naive median aggregation of the weights of
   independently-trained neural networks across institutions degrades the
   model because the per-institution networks settle into incompatible
   parameterisations. This is a key finding of the talk.
3. **The autoencoder pretraining is what makes neural-network federation
   work at all.** With an autoencoder learned per institution and
   federated *before* the supervised head is trained, the predictive head
   inherits a stable representation that all institutions share. The
   resulting federated model holds at ~0.31 across cohort sizes and
   continues to improve with iteration.
4. **Multi-round federation extracts further benefit, but the curve
   matters.** The plain convolutional network does drop sharply between
   rounds 1 and 2 (warm-starting fixes the random-initialisation problem),
   then degrades from rounds 3 onward. The autoencoder + head shows clean
   monotonic improvement and plateaus at round 4 (mean prediction error
   0.245), matching lasso's single-round federated number to two decimal
   places.
5. **Statistical power scales with cohort size, and federation is the
   device that delivers cohort size without moving data.** At the medium
   effect size R² = 0.15 (Cohen's f² ≈ 0.18), power along the
   smallest-first federation path goes from *cannot fit the model*
   (N = 10, alone) to 17 % (N = 26), to 65 % (N = 75), to 96 % (N = 150).
   The minimum detectable effect size shrinks from R² ≥ 0.91 alone to
   R² ≥ 0.10 fully federated. Single-institution studies would miss
   biologically plausible effects that the federation can decisively
   detect.

The fundamental message of the talk is the combination of arguments 1, 3,
and 5: **federation lifts the cohort-size ceiling on what we can learn
about residual β-cell function from autoantibody profiles, and it does so
without any institution acquiring more subjects or sharing any subject
data.** The interpretable models (linear, lasso, Random Forest) all
benefit substantially; the deep models benefit only when an autoencoder
provides a shared representation across institutions; iteration buys
further accuracy until a plateau is reached.

## How to read this repository

Four notebooks, all under `ipynb/`:

| Notebook | What it answers |
|---|---|
| [`LinearRegression_vs_CNN_HeadToHead.ipynb`](ipynb/LinearRegression_vs_CNN_HeadToHead.ipynb) | The centralised head-to-head: under pooled cross-validation, can a convolutional network beat simple linear regression on the same 9 features? Adds an autoencoder-pretrained variant as a probe for non-linearity. |
| [`LASSO_ElasticNet_Autoantibody_CPeptide.ipynb`](ipynb/LASSO_ElasticNet_Autoantibody_CPeptide.ipynb) | The interpretable-linear iteration: lasso and elastic-net regression with the extended feature panel (Panel B), with selected-feature interpretation. |
| [`EffectSizes_and_PowerAnalysis_Extended.ipynb`](ipynb/EffectSizes_and_PowerAnalysis_Extended.ipynb) | Effect sizes per feature, per-feature partial η², and *a-priori* statistical power calculations. Quantifies the minimum detectable effect size at each candidate cohort size, which sets up the federation power argument extended in the federated notebook below. |
| [`federated_analysis_simulation_3x3.ipynb`](ipynb/federated_analysis_simulation_3x3.ipynb) | The federated story end to end: per-study autoencoder reconstruction, then C-peptide prediction under the federation protocol described above. Tests all three arguments numerically (more institutions, more rounds, more statistical power) for all five prediction methods. Produces slide-ready summary figures including the power-vs-cohort-size headline plot. |

The shared loader [`src/oadr_data.py`](src/oadr_data.py) handles per-study
column-naming quirks (one study uses `Subject_IDel` rather than `Subject_ID`;
another uses `ImmPort Accession`) and exposes two feature panels:

| Panel | Studies | Features | Effective N |
|---|---|---|---|
| **A** (legacy) | SDY524, SDY569, SDY797, SDY1737 | 5 autoantibodies + age-group indicators + sex (9 total) | 150 |
| **B** (extended) | SDY524, SDY569, SDY1737 | Panel A + BMI, height, weight, exact age, disease duration, race, ethnicity, treatment cohort (23 total) | 98 |

Older exploration that is not on the talk's critical path lives under
`archive/`. Generated CSV outputs land in `results/`, figures in `figures/`
as both PDF (vector, for archival) and PNG at 220 dots per inch (for direct
import into a presentation deck).

**On retention.** Some of what is currently under `archive/` could be
removed from the working tree with `git rm` instead; the files would still
be recoverable from git history. The decision is purely aesthetic. The
`archive/` directory keeps historical context visible in the repository,
which is useful for an academic audit trail; using `git rm` keeps the
working tree minimal at the cost of requiring git-log spelunking to find
deleted files. Either is fine.

## Running the notebooks

Tested kernel: `~/miniforge3/envs/springer-verlag/bin/python` (TensorFlow
2.20, scikit-learn 1.7, statsmodels 0.14.6).

```bash
jupyter notebook ipynb/
```

A full run of the federated notebook from a cold state takes thirty to
sixty minutes (dominated by the multi-round convolutional methods). The
notebook checks for cached results CSVs and skips the heavy computation
if they are present, so reading the results / re-rendering the figures
takes seconds. Delete the corresponding files in `results/` to force a
re-run.

## Next step

The federated comparison in this repository is implemented in plain Python
inside Jupyter, so the protocol is fully visible. The next deliverable is to
port the same per-institution local training plus median aggregation plus
multi-round iteration into a Nextflow workflow that runs on an actual
federated platform across the four institutions. The notebook code is meant
to map directly onto that workflow: each per-institution training step
becomes a Nextflow process; the aggregation step becomes a single coordinator
process; the round loop becomes Nextflow's iteration construct.

## License

See [LICENSE](LICENSE).
