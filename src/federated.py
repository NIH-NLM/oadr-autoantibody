"""
Federated-learning helpers for the OADR autoantibody work.

This module provides the per-method aggregation rules and the cross-validation
runners used across the per-figure notebooks in `ipynb/`. The goal is one
source of truth for the federated protocols so each notebook can stay short,
focused on its specific figure, and import the heavy lifting from here.

Modules used by the talk:

  ADMM-Lasso       — consensus optimisation for federated L1 regression
                     (Boyd et al. 2011, F&T Machine Learning 3(1))
  FedAvg-by-N      — weighted mean of per-institution coefficients for OLS
  Union of forests — per-institution Random Forests; predictions averaged
                     across all institutions' trees
  Median federation — naive baseline; included for diagnostic comparison

Cross-validation throughout: StratifiedKFold(n_splits=5) on the Study label,
so every fold contains every participating institution in known proportion.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import f as fdist, ncf


# =========================================================================
# Power calculation (a-priori F-test power for multiple regression)
# =========================================================================

def calc_power(n: int, k: int, f2: float, alpha: float = 0.05) -> float:
    """
    A-priori statistical power for an F-test in multiple regression.

    Parameters
    ----------
    n : sample size
    k : number of predictors
    f2 : Cohen's f-squared effect size (f2 = R^2 / (1 - R^2))
    alpha : Type I error rate (default 0.05)

    Returns
    -------
    power : probability of rejecting H0 under the alternative, in [0, 1].
            NaN if the design can't fit the model (n <= k + 1) or f2 <= 0.
    """
    if n <= k + 1 or f2 <= 0:
        return float("nan")
    F_crit = fdist.ppf(1 - alpha, k, n - k - 1)
    nc = f2 * n
    return float(1 - ncf.cdf(F_crit, k, n - k - 1, nc))


def f2_from_r2(r2: float) -> float:
    """Convert R^2 to Cohen's f-squared. Returns 0 for r2 outside (0, 1)."""
    return r2 / (1 - r2) if 0 < r2 < 1 else 0.0


# =========================================================================
# Soft-thresholding and ADMM-Lasso (Boyd 2011 §8.2 consensus form)
# =========================================================================

def soft_threshold(v: np.ndarray, lam: float) -> np.ndarray:
    """Element-wise soft-thresholding: sign(v) * max(|v| - lam, 0)."""
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)


def admm_lasso(X_list, y_list, alpha, rho_init=1.0, max_iter=10000, tol=1e-7,
               adaptive_rho=True, mu=10.0, tau=2.0):
    """
    Consensus ADMM-Lasso with an intercept column.

    Solves the sklearn-Lasso objective in a federated setting:
        (1/(2 N_total)) * ||y - X beta||^2 + alpha * ||beta_non_intercept||_1

    Each institution k holds (X_k, y_k); only the consensus coefficient vector
    and dual variables are exchanged across the network. Subject-level data
    never leaves the institution.

    Parameters
    ----------
    X_list : list of arrays
        Per-institution feature matrices. Must be feature-scaled (e.g. by
        MinMaxScaler fit on the global training set). Do NOT pre-pend an
        intercept column — this function adds it.
    y_list : list of arrays
        Per-institution target vectors.
    alpha : float
        L1 penalty strength, sklearn convention.
    rho_init : float
        Initial augmented-Lagrangian penalty.
    max_iter, tol : convergence controls.
    adaptive_rho, mu, tau : adaptive-rho schedule (Boyd 2011 §3.4.1).
        rho is doubled when primal residual > mu * dual residual; halved when
        dual > mu * primal. Helps when per-institution sizes are heterogeneous.

    Returns
    -------
    intercept : float
    coefs : array of shape (n_features,)
    n_iter : int — number of iterations to convergence (or max_iter).
    """
    K = len(X_list)
    Xa_list = [np.column_stack([np.ones(len(X)), X]) for X in X_list]
    p = Xa_list[0].shape[1]
    N = sum(len(y) for y in y_list)

    betas = [np.zeros(p) for _ in range(K)]
    us = [np.zeros(p) for _ in range(K)]
    z = np.zeros(p)
    rho = rho_init

    Xty = [(1.0 / N) * (Xa.T @ y) for Xa, y in zip(Xa_list, y_list)]
    XtX = [(1.0 / N) * (Xa.T @ Xa) for Xa in Xa_list]

    def compute_A_inv(rho_val):
        return [np.linalg.inv(XtX[k] + rho_val * np.eye(p)) for k in range(K)]

    A_inv = compute_A_inv(rho)

    for t in range(max_iter):
        # Local update at each institution
        for k in range(K):
            betas[k] = A_inv[k] @ (Xty[k] + rho * (z - us[k]))

        # Coordinator update: soft-threshold all but the intercept
        mean_term = np.mean([betas[k] + us[k] for k in range(K)], axis=0)
        z_new = mean_term.copy()
        z_new[1:] = soft_threshold(mean_term[1:], alpha / (rho * K))

        # Dual update at each institution
        for k in range(K):
            us[k] = us[k] + betas[k] - z_new

        primal = np.sqrt(np.mean([np.sum((betas[k] - z_new) ** 2) for k in range(K)]))
        dual = rho * np.linalg.norm(z_new - z)
        z = z_new

        if primal < tol and dual < tol:
            return z[0], z[1:], t + 1

        if adaptive_rho and (t + 1) % 20 == 0:
            if primal > mu * dual:
                rho *= tau
                for k in range(K): us[k] /= tau
                A_inv = compute_A_inv(rho)
            elif dual > mu * primal:
                rho /= tau
                for k in range(K): us[k] *= tau
                A_inv = compute_A_inv(rho)

    return z[0], z[1:], max_iter


# =========================================================================
# Scatter helper: solo-vs-federated out-of-fold predictions
# =========================================================================

def federated_scatter_oof(X, y, study_labels, method, alpha=None,
                          rng_seed=42, n_splits=5):
    """
    Out-of-fold predictions for the solo-vs-federated scatter figures.

    For each StratifiedKFold-by-Study split:
      - oof_solo: SDY569 subjects predicted by a model trained ONLY on
        SDY569's training-fold subjects.
      - oof_fed: every subject predicted by a model trained on ALL
        institutions' training-fold subjects, federated by the method's
        proper aggregation rule.

    Parameters
    ----------
    X : array, shape (n, p)
    y : array, shape (n,)
    study_labels : array of str, shape (n,)
    method : {"mlr_fedavg", "lasso_admm", "rf_union"}
    alpha : float, optional — L1 strength for Lasso ADMM (ignored otherwise)

    Returns
    -------
    oof_solo, oof_fed : arrays of shape (n,) with NaN at non-test positions.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
    oof_solo = np.full_like(y, np.nan, dtype=float)
    oof_fed = np.full_like(y, np.nan, dtype=float)

    for tr, te in skf.split(X, study_labels):
        # SDY569 alone
        tr569 = tr[study_labels[tr] == "SDY569"]
        te569 = te[study_labels[te] == "SDY569"]
        if len(tr569) >= 2 and len(te569) >= 1:
            sc = MinMaxScaler().fit(X[tr569])
            Xs = sc.transform(X[tr569])
            if method == "mlr_fedavg":
                m = LinearRegression().fit(Xs, y[tr569])
            elif method == "lasso_admm":
                m = Lasso(alpha=alpha or 0.008, max_iter=20000).fit(Xs, y[tr569])
            elif method == "rf_union":
                m = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                          n_jobs=1, random_state=42).fit(Xs, y[tr569])
            else:
                raise ValueError(f"Unknown method: {method}")
            oof_solo[te569] = m.predict(sc.transform(X[te569]))

        # All institutions federated
        local_models, scalers = {}, {}
        X_list, y_list, sord = [], [], []
        for s in sorted(set(study_labels[tr])):
            idx = tr[study_labels[tr] == s]
            if len(idx) < 2: continue
            sc = MinMaxScaler().fit(X[idx])
            Xs = sc.transform(X[idx])
            scalers[s] = sc
            if method == "mlr_fedavg":
                m = LinearRegression().fit(Xs, y[idx])
                local_models[s] = (m.coef_, m.intercept_, len(idx))
            elif method == "lasso_admm":
                X_list.append(Xs); y_list.append(y[idx]); sord.append(s)
            elif method == "rf_union":
                m = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                          n_jobs=1, random_state=42).fit(Xs, y[idx])
                local_models[s] = m

        if method == "mlr_fedavg":
            coefs = np.stack([local_models[s][0] for s in local_models])
            ints = np.array([local_models[s][1] for s in local_models])
            sizes = np.array([local_models[s][2] for s in local_models])
            agg_c = np.average(coefs, axis=0, weights=sizes)
            agg_i = np.average(ints, weights=sizes)
            for i in te:
                if study_labels[i] in scalers:
                    xs = scalers[study_labels[i]].transform(X[i:i+1])[0]
                    oof_fed[i] = agg_c @ xs + agg_i
        elif method == "lasso_admm":
            sc_g = MinMaxScaler().fit(X[tr])
            X_g = [sc_g.transform(X[tr[study_labels[tr] == s]]) for s in sord]
            y_g = [y[tr[study_labels[tr] == s]] for s in sord]
            intc, coef, _ = admm_lasso(X_g, y_g, alpha=alpha or 0.008)
            for i in te:
                xs = sc_g.transform(X[i:i+1])[0]
                oof_fed[i] = coef @ xs + intc
        elif method == "rf_union":
            for i in te:
                preds = []
                for s_k, m in local_models.items():
                    xs = scalers[s_k].transform(X[i:i+1])
                    preds.append(m.predict(xs)[0])
                if preds:
                    oof_fed[i] = np.mean(preds)

    return oof_solo, oof_fed


def federated_lasso_median_oof(X, y, study_labels, alpha=0.05,
                                rng_seed=42, n_splits=5):
    """
    Out-of-fold predictions for the diagnostic Lasso-with-naive-median figure.

    Same protocol as federated_scatter_oof for method="lasso_admm" but with
    median-of-coefficients aggregation instead of ADMM. Kept as a separate
    function so the diagnostic figure can illustrate the bug it surfaces.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
    oof_solo = np.full_like(y, np.nan, dtype=float)
    oof_fed = np.full_like(y, np.nan, dtype=float)
    for tr, te in skf.split(X, study_labels):
        tr569 = tr[study_labels[tr] == "SDY569"]
        te569 = te[study_labels[te] == "SDY569"]
        if len(tr569) >= 2 and len(te569) >= 1:
            sc = MinMaxScaler().fit(X[tr569])
            m = Lasso(alpha=alpha, max_iter=20000).fit(sc.transform(X[tr569]), y[tr569])
            oof_solo[te569] = m.predict(sc.transform(X[te569]))
        local_coefs, local_ints, scalers = {}, {}, {}
        for s in sorted(set(study_labels[tr])):
            idx = tr[study_labels[tr] == s]
            if len(idx) < 2: continue
            sc = MinMaxScaler().fit(X[idx])
            m = Lasso(alpha=alpha, max_iter=20000).fit(sc.transform(X[idx]), y[idx])
            local_coefs[s] = m.coef_; local_ints[s] = m.intercept_; scalers[s] = sc
        coefs = np.stack(list(local_coefs.values()))
        med_coef = np.median(coefs, axis=0)
        med_int = np.median(list(local_ints.values()))
        for i in te:
            if study_labels[i] in scalers:
                xs = scalers[study_labels[i]].transform(X[i:i+1])[0]
                oof_fed[i] = med_coef @ xs + med_int
    return oof_solo, oof_fed


# =========================================================================
# Two-panel scatter figure (solo vs federated)
# =========================================================================

def metrics(y, oof):
    """Return (mse, r2, n) on the non-NaN positions of `oof`."""
    mask = ~np.isnan(oof)
    if mask.sum() < 2:
        return float("nan"), float("nan"), int(mask.sum())
    mse = float(np.mean((y[mask] - oof[mask]) ** 2))
    rss = float(np.sum((y[mask] - oof[mask]) ** 2))
    tss = float(np.sum((y[mask] - y[mask].mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    return mse, r2, int(mask.sum())


def plot_scatter_pair(y, study_labels, oof_solo, oof_fed,
                       suptitle, out_path_png,
                       study_colors=None,
                       figsize=(11.5, 5)):
    """
    Two-panel scatter: left = SDY569 alone, right = all institutions federated.

    Each point is a held-out subject prediction. Title of each panel shows
    MSE and R² on the non-NaN positions. Also saves a PDF alongside the PNG.
    """
    import matplotlib.pyplot as plt
    if study_colors is None:
        study_colors = {"SDY524": "#1f77b4", "SDY569": "#d62728",
                        "SDY797": "#2ca02c", "SDY1737": "#9467bd"}

    is569 = (study_labels == "SDY569")
    mse_solo, r2_solo, _ = metrics(y[is569], oof_solo[is569])
    mse_fed, r2_fed, _ = metrics(y, oof_fed)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    ax = axes[0]
    mask = is569 & ~np.isnan(oof_solo)
    ax.scatter(y[mask], oof_solo[mask], c=study_colors["SDY569"], s=65,
                edgecolor="white", linewidth=0.6,
                label=f"SDY569 (N={mask.sum()})")
    if mask.sum() > 0:
        lo = min(y[mask].min(), oof_solo[mask].min())
        hi = max(y[mask].max(), oof_solo[mask].max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
    ax.set_xlabel("Observed log(C-peptide AUC)")
    ax.set_ylabel("Predicted log(C-peptide AUC)")
    ax.set_title(f"SDY569 alone  (N = {mask.sum()})\n"
                  f"MSE = {mse_solo:.3f}    R² = {r2_solo:+.3f}",
                  fontweight="bold")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.25)

    ax = axes[1]
    for s in sorted(set(study_labels)):
        mask = (study_labels == s) & ~np.isnan(oof_fed)
        if mask.sum() == 0: continue
        ax.scatter(y[mask], oof_fed[mask], c=study_colors.get(s, "grey"),
                    s=65, edgecolor="white", linewidth=0.6, alpha=0.85,
                    label=f"{s} (N={mask.sum()})")
    all_mask = ~np.isnan(oof_fed)
    if all_mask.sum() > 0:
        lo = min(y[all_mask].min(), oof_fed[all_mask].min())
        hi = max(y[all_mask].max(), oof_fed[all_mask].max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
    ax.set_xlabel("Observed log(C-peptide AUC)")
    ax.set_ylabel("Predicted log(C-peptide AUC)")
    ax.set_title(f"All institutions federated  (N = {all_mask.sum()})\n"
                  f"MSE = {mse_fed:.3f}    R² = {r2_fed:+.3f}",
                  fontweight="bold")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.25)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.savefig(out_path_png, dpi=220)
    fig.savefig(str(out_path_png).replace(".png", ".pdf"))
    return mse_solo, r2_solo, mse_fed, r2_fed
