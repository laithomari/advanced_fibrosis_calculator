#!/usr/bin/env python3
"""
8-Feature Advanced Fibrosis Prediction Model — Analysis Pipeline
=================================================================
L2-regularized logistic regression for predicting advanced fibrosis (F>=3)
in MASLD patients with indeterminate FIB-4 scores (1.3-2.67).

Pipeline:
  1. Data loading and feature engineering
  2. Train/validation split (hybrid training: Low + High + 65% Indeterminate)
  3. Recursive feature elimination (RFE) with 5-fold CV
  4. Final model training (8 selected features, class_weight=None)
  5. Threshold determination (rule-out at 90% sensitivity, rule-in at 90% specificity)
  6. Cohort evaluation (Internal, Asian External, NHANES)
  7. Threshold sweep, risk stratification, subgroup analyses
  8. Head-to-head comparison with APRI, NFS, SAFE
  9. Sensitivity analysis (indeterminate-only training)
  10. Optimism-corrected AUROC (200 bootstrap iterations)
  11. NHANES LSM threshold sensitivity

Features (7 inputs + 1 derived):
  age, bmi, ast (log), alt (log), ggt (log), platelets, diabetes, ast/alt ratio

Usage:
    python model_analysis.py --data_dir /path/to/data --output_dir /path/to/output

Data requirements:
  - data_dir/df2.csv                              (NAFLD DB2 cohort)
  - data_dir/NHANES_with_fib4_nonmissing.csv      (NHANES cohort)
  - data_dir/external_data/livefbr/discovery_set.csv   (Asian cohort)
  - data_dir/external_data/livefbr/validation_set1.csv (Asian cohort)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 3473

# All 11 candidate features considered during RFE
ALL_MODEL_FEATURES = [
    "age", "gender", "bmi",
    "ast_log", "alt_log", "ggt_log",
    "platelets", "hba1c_log",
    "diabetes", "cholesterol", "ast_alt_ratio",
]

# Target number of features selected by RFE (plateau point)
TARGET_N_FEATURES = 8


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def bootstrap_auc(y_true, y_pred, n_boot=1000, seed=RANDOM_STATE):
    """Compute AUROC with 95% CI via bootstrap (percentile method)."""
    auc_point = roc_auc_score(y_true, y_pred)
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) > 1:
            aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    return auc_point, np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def metrics_at_threshold(y_true, y_pred, threshold):
    """Compute classification metrics at a given probability threshold."""
    y_class = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_class, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def full_evaluate(y_true, y_pred, thresholds):
    """Full evaluation: AUROC, AUPRC, Brier, calibration, classification."""
    auc_val, auc_lo, auc_hi = bootstrap_auc(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall, precision)
    brier = brier_score_loss(y_true, y_pred)
    mean_pred = float(y_pred.mean())
    mean_obs = float(y_true.mean())
    oe_ratio = mean_obs / mean_pred if mean_pred > 0 else float("nan")

    classification = {}
    for name, th in thresholds.items():
        classification[name] = metrics_at_threshold(y_true, y_pred, th)

    return {
        "n": len(y_true),
        "events": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "auroc": float(auc_val),
        "auroc_lo": float(auc_lo),
        "auroc_hi": float(auc_hi),
        "auprc": float(auprc),
        "brier": float(brier),
        "mean_pred": mean_pred,
        "mean_obs": mean_obs,
        "oe_ratio": float(oe_ratio),
        "classification": classification,
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_db2(data_dir):
    """Load and prepare the NAFLD DB2 cohort."""
    df = pd.read_csv(os.path.join(data_dir, "df2.csv"))
    df["fib4"] = (df["age"] * df["ast"]) / ((df["platelets"] / 1000) * np.sqrt(df["alt"]))
    df["fib4_cat"] = pd.cut(df["fib4"], bins=[0, 1.3, 2.67, float("inf")], labels=["Low", "Indet", "High"])
    df["target"] = (df["fib_stage"] >= 3).astype(int)
    df["ast_log"] = np.log1p(df["ast"])
    df["alt_log"] = np.log1p(df["alt"])
    df["ggt_log"] = np.log1p(df["ggt"])
    df["hba1c_log"] = np.log1p(df["hba1c"])
    df["ast_alt_ratio"] = df["ast"] / df["alt"]
    df["diabetes"] = ((df["hba1c"] >= 6.5) | (df["glucose"] >= 126)).astype(int)
    return df


def load_nhanes(data_dir):
    """Load and prepare the NHANES cohort."""
    nh = pd.read_csv(os.path.join(data_dir, "NHANES_with_fib4_nonmissing.csv"))
    nh["cholesterol"] = nh["total_cholesterol"]
    # Platelet unit harmonization
    if nh["platelets"].mean() <= 500:
        nh["platelets"] = nh["platelets"] * 1000
    nh["ast_log"] = np.log1p(nh["ast"])
    nh["alt_log"] = np.log1p(nh["alt"])
    nh["ggt_log"] = np.log1p(nh["ggt"])
    nh["hba1c"] = nh["a1c"]
    nh["hba1c_log"] = np.log1p(nh["a1c"])
    nh["ast_alt_ratio"] = nh["ast"] / nh["alt"]
    nh["diabetes"] = ((nh["diabetes"] == 1) | (nh["glucose"] >= 126) | (nh["a1c"] >= 6.5)).astype(int)
    nh["target"] = (nh["median_stiffness"] >= 10).astype(int)
    nh["fib4_cat"] = pd.cut(nh["fib_4"], bins=[0, 1.3, 2.67, float("inf")], labels=["Low", "Indet", "High"])
    # Gender column harmonization
    if "gender" not in nh.columns and "sex" in nh.columns:
        nh["gender"] = nh["sex"]
    if "gender" not in nh.columns and "female" in nh.columns:
        nh["gender"] = 1 - nh["female"]
    if "albumin" not in nh.columns and "serum_albumin" in nh.columns:
        nh["albumin"] = nh["serum_albumin"]
    return nh


def load_asian(data_dir):
    """Load and prepare the Asian (LiveFbr) external validation cohort."""
    def clean_one(df_raw):
        out = pd.DataFrame()
        out["age"] = df_raw["Age"]
        out["gender"] = df_raw["Sex"]
        out["bmi"] = df_raw["BMI"]
        out["ast"] = df_raw["AST"]
        out["alt"] = df_raw["ALT"]
        out["ggt"] = df_raw["GGT"]
        out["platelets"] = df_raw["PLT"] * 1000
        out["hba1c"] = df_raw["HbA1c"]
        out["glucose"] = df_raw["FBG"] * 18.0
        out["cholesterol"] = df_raw["TC"] * 38.67
        out["diabetes"] = ((df_raw["DM.IFG"] == 1) | (out["glucose"] >= 126) | (out["hba1c"] >= 6.5)).astype(int)
        out["target"] = (df_raw["group"] == "S34").astype(int)
        out["fib4"] = (out["age"] * out["ast"]) / ((out["platelets"] / 1000) * np.sqrt(out["alt"]))
        out["fib4_cat"] = pd.cut(out["fib4"], bins=[0, 1.3, 2.67, float("inf")], labels=["Low", "Indet", "High"])
        out["ast_log"] = np.log1p(out["ast"])
        out["alt_log"] = np.log1p(out["alt"])
        out["ggt_log"] = np.log1p(out["ggt"])
        out["hba1c_log"] = np.log1p(out["hba1c"])
        out["ast_alt_ratio"] = out["ast"] / out["alt"]
        out["albumin"] = np.nan
        return out

    disc = pd.read_csv(os.path.join(data_dir, "external_data", "livefbr", "discovery_set.csv"))
    val1 = pd.read_csv(os.path.join(data_dir, "external_data", "livefbr", "validation_set1.csv"))
    return pd.concat([clean_one(disc), clean_one(val1)], ignore_index=True)


# ============================================================
# COMPETING SCORES
# ============================================================

def calc_apri(df):
    """APRI score."""
    plts = df["platelets"].copy()
    if plts.mean() > 500:
        plts = plts / 1000
    return ((df["ast"] / 40.0) / plts) * 100


def calc_nfs(df):
    """NAFLD Fibrosis Score."""
    plts = df["platelets"].copy()
    if plts.mean() > 500:
        plts = plts / 1000
    return (-1.675 + 0.037 * df["age"] + 0.094 * df["bmi"] + 1.13 * df["diabetes"]
            + 0.99 * (df["ast"] / df["alt"]) - 0.013 * plts - 0.66 * df["albumin"])


def calc_safe(df):
    """SAFE score."""
    plts = df["platelets"].copy()
    if plts.mean() > 500:
        plts = plts / 1000
    female = df.get("female", None)
    if female is None:
        if "gender" in df.columns:
            prop = df["gender"].mean()
            female = df["gender"] if prop > 0.5 else (1 - df["gender"])
        else:
            female = pd.Series(0, index=df.index)
    return (-7.981 + 0.046 * df["age"] - 0.800 * female + 0.158 * df["bmi"]
            + 0.015 * df["ast"] + 0.004 * df["alt"] + 1.643 * df["diabetes"]
            - 0.011 * plts + 0.926 * np.log(df["ggt"].clip(lower=1)))


# ============================================================
# MAIN PIPELINE
# ============================================================

def main(data_dir, output_dir):
    results = {}
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("8-FEATURE FIBROSIS RISK STRATIFICATION MODEL")
    print("=" * 70)

    # ── 1. Load data ──
    print("\n[1] Loading data...")
    df = load_db2(data_dir)
    nh = load_nhanes(data_dir)
    asian = load_asian(data_dir)
    print(f"    DB2: {len(df)}, NHANES: {len(nh)}, Asian: {len(asian)}")

    # ── 2. Train/validation split ──
    print("\n[2] Splitting data...")
    df_indet = df[df["fib4_cat"] == "Indet"]
    indet_train, indet_val = train_test_split(
        df_indet, test_size=0.35,
        stratify=df_indet["target"],
        random_state=RANDOM_STATE,
    )
    df_train = pd.concat([
        df[df["fib4_cat"] == "Low"],
        df[df["fib4_cat"] == "High"],
        indet_train,
    ])
    asian_indet = asian[asian["fib4_cat"] == "Indet"].dropna(subset=["target"])
    nh_indet = nh[nh["fib4_cat"] == "Indet"].dropna(subset=["target"])

    print(f"    Training: {len(df_train)} (Low + High + 65% Indet)")
    print(f"    Internal Validation: {len(indet_val)} (35% Indet)")
    print(f"    Asian External: {len(asian_indet)}")
    print(f"    NHANES: {len(nh_indet)}")

    results["cohort_sizes"] = {
        "training": len(df_train),
        "indet_train": len(indet_train),
        "internal_validation": len(indet_val),
        "asian_external": len(asian_indet),
        "nhanes": len(nh_indet),
    }

    # ── 3. Preprocessing ──
    print("\n[3] Preprocessing (median imputation + standardization)...")
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_train = df_train[ALL_MODEL_FEATURES].values
    y_train = df_train["target"].values
    X_train_scaled = sc.fit_transform(imp.fit_transform(X_train))

    scaler_means = dict(zip(ALL_MODEL_FEATURES, sc.mean_))
    scaler_stds = dict(zip(ALL_MODEL_FEATURES, sc.scale_))

    # ── 4. Recursive Feature Elimination ──
    print("\n[4] Feature selection (RFE with 5-fold CV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base = LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced",
        max_iter=5000, random_state=RANDOM_STATE,
    )

    cv_results = []
    for n in range(4, len(ALL_MODEL_FEATURES) + 1):
        rfe = RFE(base, n_features_to_select=n, step=1)
        rfe.fit(X_train_scaled, y_train)
        auc_cv = cross_val_score(
            base, X_train_scaled[:, rfe.support_], y_train,
            cv=cv, scoring="roc_auc",
        ).mean()
        cv_results.append((n, auc_cv, rfe.support_.copy()))
        print(f"    {n} features: CV-AUROC = {auc_cv:.4f}")

    results["rfe_results"] = []
    for n, auc_cv, support in cv_results:
        feats = [f for f, s in zip(ALL_MODEL_FEATURES, support) if s]
        results["rfe_results"].append({
            "n_features": n, "cv_auroc": float(auc_cv), "features": feats,
        })

    # Select 8-feature model
    target_result = [r for r in cv_results if r[0] == TARGET_N_FEATURES][0]
    best_n, best_auc_cv, best_support = target_result
    selected_features = [f for f, s in zip(ALL_MODEL_FEATURES, best_support) if s]

    print(f"\n    Selected: {best_n} features (CV-AUROC = {best_auc_cv:.4f})")
    print(f"    Features: {selected_features}")

    results["selected_features"] = selected_features
    results["cv_auroc"] = float(best_auc_cv)

    # ── 5. Train final model ──
    print("\n[5] Training final model (class_weight=None)...")
    X_train_sel = X_train_scaled[:, best_support]
    model = LogisticRegression(
        penalty="l2", solver="lbfgs", C=1.0,
        class_weight=None, max_iter=5000, random_state=RANDOM_STATE,
    )
    model.fit(X_train_sel, y_train)

    coefficients = dict(zip(selected_features, [float(c) for c in model.coef_[0]]))
    intercept = float(model.intercept_[0])

    print(f"    Intercept: {intercept:.4f}")
    for f, c in coefficients.items():
        print(f"    {f}: {c:+.4f}")

    results["model"] = {
        "intercept": intercept,
        "coefficients": coefficients,
        "features": selected_features,
        "n_features": best_n,
    }

    results["standardization"] = {}
    for f in selected_features:
        results["standardization"][f] = {
            "mean": float(scaler_means[f]),
            "std": float(scaler_stds[f]),
        }

    # ── 6. Threshold determination ──
    print("\n[6] Determining thresholds on training indeterminate subset...")
    X_ti = sc.transform(imp.transform(indet_train[ALL_MODEL_FEATURES].values))[:, best_support]
    p_ti = model.predict_proba(X_ti)[:, 1]
    y_ti = indet_train["target"].values

    fpr_t, tpr_t, ths_t = roc_curve(y_ti, p_ti)
    th_youden = float(ths_t[np.argmax(tpr_t - fpr_t)])
    th_sens90 = float(ths_t[np.argmin(np.abs(tpr_t - 0.90))])
    th_spec90 = float(ths_t[np.argmin(np.abs((1 - fpr_t) - 0.90))])

    thresholds = {"rule_out": th_sens90, "youden": th_youden, "rule_in": th_spec90}
    print(f"    Rule-out (90% sens): {th_sens90:.4f}")
    print(f"    Youden optimal:      {th_youden:.4f}")
    print(f"    Rule-in (90% spec):  {th_spec90:.4f}")
    results["thresholds"] = thresholds

    # ── Helper ──
    def get_preds(data):
        X = sc.transform(imp.transform(data[ALL_MODEL_FEATURES].values))[:, best_support]
        return model.predict_proba(X)[:, 1]

    # ── 7. Evaluate all cohorts ──
    print("\n[7] Evaluating on all cohorts...")
    y_val = indet_val["target"].values
    p_val = get_preds(indet_val)
    y_asian = asian_indet["target"].values
    p_asian = get_preds(asian_indet)
    y_nh = nh_indet["target"].values
    p_nh = get_preds(nh_indet)

    cohort_list = [
        ("Internal Validation (DB2)", indet_val, y_val, p_val),
        ("Asian External (LiveFbr)", asian_indet, y_asian, p_asian),
        ("NHANES Generalizability", nh_indet, y_nh, p_nh),
    ]

    results["cohorts"] = {}
    for name, data, y_true, y_pred in cohort_list:
        res = full_evaluate(y_true, y_pred, thresholds)
        results["cohorts"][name] = res
        print(f"    {name}: AUROC={res['auroc']:.3f} "
              f"({res['auroc_lo']:.3f}-{res['auroc_hi']:.3f}), "
              f"N={res['n']}, Events={res['events']}, O/E={res['oe_ratio']:.2f}")

    # ── 8. Threshold sweep ──
    print("\n[8] Threshold sweep (internal validation)...")
    results["threshold_sweep"] = []
    for th in np.arange(0.10, 0.75, 0.05):
        m = metrics_at_threshold(y_val, p_val, th)
        results["threshold_sweep"].append(m)
        print(f"    {th:.2f}: Sens={m['sensitivity']:.1%}, Spec={m['specificity']:.1%}, "
              f"PPV={m['ppv']:.1%}, NPV={m['npv']:.1%}")

    # ── 9. Risk stratification ──
    print("\n[9] Risk stratification...")
    results["risk_stratification"] = {}
    for name, y_true, y_pred in [
        ("Internal Validation", y_val, p_val),
        ("Asian External", y_asian, p_asian),
        ("NHANES", y_nh, p_nh),
    ]:
        low = y_pred < th_sens90
        high = y_pred >= th_spec90
        mid = ~low & ~high
        strat = {
            "low_risk": {"n": int(low.sum()), "pct": float(low.mean()),
                         "event_rate": float(y_true[low].mean()) if low.sum() > 0 else 0},
            "intermediate": {"n": int(mid.sum()), "pct": float(mid.mean()),
                             "event_rate": float(y_true[mid].mean()) if mid.sum() > 0 else 0},
            "high_risk": {"n": int(high.sum()), "pct": float(high.mean()),
                          "event_rate": float(y_true[high].mean()) if high.sum() > 0 else 0},
            "classified_pct": float((low | high).mean()),
        }
        results["risk_stratification"][name] = strat
        print(f"    {name}: Low={strat['low_risk']['n']} ({strat['low_risk']['event_rate']:.1%}), "
              f"Int={strat['intermediate']['n']} ({strat['intermediate']['event_rate']:.1%}), "
              f"High={strat['high_risk']['n']} ({strat['high_risk']['event_rate']:.1%}), "
              f"Classified={strat['classified_pct']:.1%}")

    # ── 10. Subgroup analyses ──
    print("\n[10] Subgroup analyses...")
    results["subgroups"] = {}

    def subgroup_auc(y_true_all, p_pred_all, mask):
        y = y_true_all[mask]
        p = p_pred_all[mask]
        if len(y) < 15 or len(np.unique(y)) < 2:
            return None
        a, lo, hi = bootstrap_auc(y, p, n_boot=500)
        return {"n": len(y), "events": int(y.sum()), "auroc": float(a),
                "auroc_lo": float(lo), "auroc_hi": float(hi)}

    for cohort_name, data, y_true, y_pred in cohort_list:
        results["subgroups"][cohort_name] = {}
        for label, cond in [("<65 years", data["age"] < 65), (">=65 years", data["age"] >= 65)]:
            mask = cond.values if hasattr(cond, "values") else cond
            r = subgroup_auc(y_true, y_pred, mask)
            if r:
                results["subgroups"][cohort_name][f"age_{label}"] = r
                print(f"    {cohort_name} | Age {label}: AUROC={r['auroc']:.3f}")
        for label, cond in [("<30", data["bmi"] < 30), (">=30", data["bmi"] >= 30)]:
            mask = cond.values if hasattr(cond, "values") else cond
            r = subgroup_auc(y_true, y_pred, mask)
            if r:
                results["subgroups"][cohort_name][f"bmi_{label}"] = r
                print(f"    {cohort_name} | BMI {label}: AUROC={r['auroc']:.3f}")
        for label, cond in [("No", data["diabetes"] == 0), ("Yes", data["diabetes"] == 1)]:
            mask = cond.values if hasattr(cond, "values") else cond
            r = subgroup_auc(y_true, y_pred, mask)
            if r:
                results["subgroups"][cohort_name][f"diabetes_{label}"] = r
                print(f"    {cohort_name} | Diabetes {label}: AUROC={r['auroc']:.3f}")

    # ── 11. VCTE comparison ──
    print("\n[11] VCTE comparison...")
    lsm_col = None
    for col in ["median_lsm", "vcte", "lsm", "liver_stiffness", "elastography"]:
        if col in indet_val.columns:
            lsm_col = col
            break

    if lsm_col and indet_val[lsm_col].notna().sum() > 20:
        vcte_mask = indet_val[lsm_col].notna()
        y_vcte = indet_val.loc[vcte_mask, "target"].values
        p_vcte_model = p_val[vcte_mask.values]
        p_vcte_lsm = indet_val.loc[vcte_mask, lsm_col].values

        if len(np.unique(y_vcte)) > 1:
            auc_m, lo_m, hi_m = bootstrap_auc(y_vcte, p_vcte_model)
            auc_l, lo_l, hi_l = bootstrap_auc(y_vcte, p_vcte_lsm)
            z = (auc_m - auc_l) / max(0.001, np.sqrt(
                ((hi_m - lo_m) / 3.92) ** 2 + ((hi_l - lo_l) / 3.92) ** 2))
            p_delong = 2 * (1 - norm.cdf(abs(z)))
            results["vcte_comparison"] = {
                "n": int(vcte_mask.sum()),
                "model_auroc": float(auc_m), "model_lo": float(lo_m), "model_hi": float(hi_m),
                "vcte_auroc": float(auc_l), "vcte_lo": float(lo_l), "vcte_hi": float(hi_l),
                "delong_p": float(p_delong),
            }
            print(f"    Model: {auc_m:.3f} ({lo_m:.3f}-{hi_m:.3f})")
            print(f"    VCTE:  {auc_l:.3f} ({lo_l:.3f}-{hi_l:.3f}), p={p_delong:.3f}")
        else:
            results["vcte_comparison"] = None
            print("    Insufficient outcome variation in VCTE subset")
    else:
        results["vcte_comparison"] = None
        print("    VCTE/LSM data not available in validation set")

    # ── 12. Head-to-head with APRI, NFS, SAFE ──
    print("\n[12] Head-to-head comparison...")
    results["head_to_head"] = {}

    for cohort_name, data, y_true, our_preds in cohort_list:
        results["head_to_head"][cohort_name] = {}
        auc_ours, lo_ours, hi_ours = bootstrap_auc(y_true, our_preds)
        results["head_to_head"][cohort_name]["our_model"] = {
            "auroc": float(auc_ours), "lo": float(lo_ours), "hi": float(hi_ours)}
        print(f"\n    {cohort_name}:")
        print(f"      Our Model: {auc_ours:.3f} ({lo_ours:.3f}-{hi_ours:.3f})")

        for score_name, calc_fn in [("apri", calc_apri), ("nfs", calc_nfs), ("safe", calc_safe)]:
            try:
                if score_name == "nfs":
                    if "albumin" not in data.columns or data["albumin"].notna().sum() <= 20:
                        continue
                scores = calc_fn(data)
                valid = scores.notna()
                if valid.sum() > 20 and len(np.unique(y_true[valid])) > 1:
                    a, lo, hi = bootstrap_auc(y_true[valid], scores.values[valid])
                    results["head_to_head"][cohort_name][score_name] = {
                        "auroc": float(a), "lo": float(lo), "hi": float(hi), "n": int(valid.sum())}
                    print(f"      {score_name.upper():10s}: {a:.3f} ({lo:.3f}-{hi:.3f})")
            except Exception:
                pass

    # ── 13. Sensitivity analysis: indeterminate-only training ──
    print("\n[13] Sensitivity analysis: indeterminate-only training...")
    imp2 = SimpleImputer(strategy="median")
    sc2 = StandardScaler()
    X_train2 = indet_train[ALL_MODEL_FEATURES].values
    y_train2 = indet_train["target"].values
    X_train2_scaled = sc2.fit_transform(imp2.fit_transform(X_train2))

    cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base2 = LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced",
        max_iter=5000, random_state=RANDOM_STATE,
    )
    cv_results2 = []
    for n in range(4, len(ALL_MODEL_FEATURES) + 1):
        rfe2 = RFE(base2, n_features_to_select=n, step=1)
        rfe2.fit(X_train2_scaled, y_train2)
        auc_cv2 = cross_val_score(
            base2, X_train2_scaled[:, rfe2.support_], y_train2,
            cv=cv2, scoring="roc_auc",
        ).mean()
        cv_results2.append((n, auc_cv2, rfe2.support_.copy()))

    best_n2, _, best_support2 = max(cv_results2, key=lambda x: x[1])
    selected_features2 = [f for f, s in zip(ALL_MODEL_FEATURES, best_support2) if s]

    model2 = LogisticRegression(
        penalty="l2", solver="lbfgs", C=1.0,
        class_weight=None, max_iter=5000, random_state=RANDOM_STATE,
    )
    model2.fit(X_train2_scaled[:, best_support2], y_train2)

    def get_preds2(data):
        X = sc2.transform(imp2.transform(data[ALL_MODEL_FEATURES].values))[:, best_support2]
        return model2.predict_proba(X)[:, 1]

    results["sensitivity_indet_only"] = {
        "features": selected_features2, "n_features": best_n2, "cohorts": {},
    }
    print(f"    Indet-only selected {best_n2} features: {selected_features2}")

    for cohort_name, data, y_true, our_preds in cohort_list:
        p2 = get_preds2(data)
        a_hyb, lo_hyb, hi_hyb = bootstrap_auc(y_true, our_preds)
        a_ind, lo_ind, hi_ind = bootstrap_auc(y_true, p2)
        results["sensitivity_indet_only"]["cohorts"][cohort_name] = {
            "hybrid_auroc": float(a_hyb), "hybrid_lo": float(lo_hyb), "hybrid_hi": float(hi_hyb),
            "indet_auroc": float(a_ind), "indet_lo": float(lo_ind), "indet_hi": float(hi_ind),
        }
        print(f"    {cohort_name}: Hybrid={a_hyb:.3f}, Indet-only={a_ind:.3f}")

    # ── 14. Optimism-corrected AUROC ──
    print("\n[14] Optimism-corrected AUROC (200 bootstrap iterations)...")
    n_boot_opt = 200
    rng = np.random.RandomState(RANDOM_STATE)
    optimisms = []
    for i in range(n_boot_opt):
        boot_idx = rng.choice(len(df_train), len(df_train), replace=True)
        boot_train = df_train.iloc[boot_idx]
        X_b = boot_train[ALL_MODEL_FEATURES].values
        y_b = boot_train["target"].values
        imp_b = SimpleImputer(strategy="median")
        sc_b = StandardScaler()
        X_b_scaled = sc_b.fit_transform(imp_b.fit_transform(X_b))
        base_b = LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            max_iter=5000, random_state=RANDOM_STATE,
        )
        rfe_b = RFE(base_b, n_features_to_select=best_n, step=1)
        rfe_b.fit(X_b_scaled, y_b)
        model_b = LogisticRegression(
            penalty="l2", solver="lbfgs", C=1.0,
            class_weight=None, max_iter=5000, random_state=RANDOM_STATE,
        )
        model_b.fit(X_b_scaled[:, rfe_b.support_], y_b)
        p_app = model_b.predict_proba(X_b_scaled[:, rfe_b.support_])[:, 1]
        if len(np.unique(y_b)) < 2:
            continue
        auc_app = roc_auc_score(y_b, p_app)
        X_val_b = sc_b.transform(imp_b.transform(indet_val[ALL_MODEL_FEATURES].values))[:, rfe_b.support_]
        p_test = model_b.predict_proba(X_val_b)[:, 1]
        auc_test = roc_auc_score(y_val, p_test)
        optimisms.append(auc_app - auc_test)
        if (i + 1) % 50 == 0:
            print(f"      {i + 1}/{n_boot_opt}...")

    apparent_auc = float(roc_auc_score(y_train, model.predict_proba(X_train_sel)[:, 1]))
    mean_opt = float(np.mean(optimisms))
    corrected = apparent_auc - mean_opt
    results["optimism_correction"] = {
        "apparent_auroc": apparent_auc,
        "mean_optimism": mean_opt,
        "corrected_auroc": corrected,
        "n_iterations": len(optimisms),
    }
    print(f"    Apparent: {apparent_auc:.4f}, Optimism: {mean_opt:.4f}, Corrected: {corrected:.4f}")

    # ── 15. NHANES LSM threshold sensitivity ──
    print("\n[15] NHANES LSM threshold sensitivity...")
    results["nhanes_lsm_sensitivity"] = {}
    for lsm_thresh in [8, 10, 12]:
        y_nh_alt = (nh_indet["median_stiffness"] >= lsm_thresh).astype(int).values
        if len(np.unique(y_nh_alt)) < 2:
            continue
        res_nh = full_evaluate(y_nh_alt, p_nh, thresholds)
        results["nhanes_lsm_sensitivity"][f">={lsm_thresh}kPa"] = res_nh
        print(f"    LSM>={lsm_thresh}kPa: AUROC={res_nh['auroc']:.3f}, "
              f"N={res_nh['n']}, Events={res_nh['events']}")

    # ── Save results ──
    output_file = os.path.join(output_dir, "results.json")
    print(f"\n[16] Saving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="8-Feature Advanced Fibrosis Prediction Model — Analysis Pipeline"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to directory containing df2.csv, NHANES_with_fib4_nonmissing.csv, "
             "and external_data/livefbr/ subdirectory",
    )
    parser.add_argument(
        "--output_dir", default="./output",
        help="Path to directory for saving results.json (default: ./output)",
    )
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
