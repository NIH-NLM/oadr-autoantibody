# Predicting residual beta-cell function in Type 1 Diabetes, under federation

This repository contains the analytical work behind a 
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
patient still has is the **C-peptide** released during a standardised
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

## This analysis - privacy preserving federated analysis

Each cohort comes from a different institutional clinical study with its own
data-use governance. In a real deployment those data sets cannot be moved
between institutions: privacy regulations, IRB constraints, and institutional
policies forbid it. The methodological question is therefore not *"what is the
best model when we put all the data in one place?"* but **"what is the best
model when each institution trains on its own subjects and only model
parameters are exchanged with a central coordinator?"** That is federated
learning, and it is the only framing under which the work could be deployed
in practice.

In this analysis, the four institutions cannot agree to centralise their data, but they can
agree to share **only** model parameters after each one has trained on its
own subjects, and to accept a central coordinator that aggregates those
parameters by taking the median value across institutions for each parameter.
This is the protocol used everywhere in this repository.

## The two arguments this analysis makes

1. **Federation across more institutions improves prediction.** Adding more
   participating studies to the federation lowers the prediction error of
   every method, even when each study only trains once. The plot of
   prediction error against number of participating institutions is one of
   the talk's two central figures.

2. **Iterating federation across multiple rounds extracts further benefit
   until the improvement plateaus.** When each institution warm-starts its
   local model from the previous round's aggregated parameters and continues
   training, then re-federates, the prediction error continues to fall until
   the round-over-round improvement is too small to matter. The plot of
   prediction error against round number is the talk's second central
   figure.

3. **Explainable Analysis in using Artificial Intelligence**. 

The federated analysis and iteration  arguments are evaluated against different prediction methods.
Convolution neural networks with and without autoencoding represented a black box method that is *not*
interpretable, we do not know the *why* of the specific parameter behind a better result.  We do learn
that there is a latent variable, but we do not know which one.   Other methods, linear regression, LASSO regre

| Method | Family | Interpretable? |
|---|---|---|
| Simple linear regression | Linear, no regularisation | yes (coefficients) |
| Lasso regression | Linear, automatic feature selection via L1 penalty | yes (non-zero coefficients) |
| Convolutional neural network | Non-linear, no pretraining | no (latent layers) |
| Convolutional network with autoencoder pretraining | Non-linear, two-stage training: unsupervised feature learning then a small supervised head | no (latent layers) |

## How to read this repository

Four notebooks, all under `ipynb/`:

| Notebook | What it answers |
|---|---|
| [`LinearRegression_vs_CNN_HeadToHead.ipynb`](ipynb/LinearRegression_vs_CNN_HeadToHead.ipynb) | The centralised head-to-head: under pooled cross-validation, can a convolutional network beat simple linear regression on the same 9 features? Adds an autoencoder-pretrained variant as a probe for non-linearity. |
| [`LASSO_ElasticNet_Autoantibody_CPeptide.ipynb`](ipynb/LASSO_ElasticNet_Autoantibody_CPeptide.ipynb) | The interpretable-linear iteration: lasso and elastic-net regression with the extended feature panel (Panel B), with selected-feature interpretation. |
| [`EffectSizes_and_PowerAnalysis_Extended.ipynb`](ipynb/EffectSizes_and_PowerAnalysis_Extended.ipynb) | Effect sizes per feature and frequentist power calculations on top of the linear/lasso results. |
| [`federated_analysis_simulation_3x3.ipynb`](ipynb/federated_analysis_simulation_3x3.ipynb) | The federated story end to end: per-study autoencoder reconstruction, then C-peptide prediction under the federation protocol described above, with both arguments tested numerically and a slide-ready summary figure. |

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
