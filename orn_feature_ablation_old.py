from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

# ============================================================
# Purpose
# ============================================================
# This script is an additional feature-removal ablation experiment for the legacy thesis dataset data_v3.1.xlsx.
# It keeps the main pipeline logic unchanged:
#   - same target: ORN_label
#   - same full feature set: clinical + treatment
#   - patient-level hold-out split
#   - patient-level repeated StratifiedGroupKFold CV
#   - same preprocessing: median imputation for numeric variables and one-hot encoding for categorical variables
#
# Recommended use in the thesis:
#   Treat this as a small sensitivity analysis, not as the main model comparison.

# ============================================================
# Constants matching legacy / thesis main structured-data pipeline (data_v3.1.xlsx)
# ============================================================
RANDOM_STATE = 42
TARGET_COL = "ORN_label"
ID_COL = "ipatient"

DROP_NON_MODEL_COLS = [
    "orn_diagnosis_date",
    "censor_date",
    "reference_date_for_model",
]

EXPECTED_CATEGORICAL = [
    "性別",
    "Tumor location",
    "T_analyze",
    "N_analyze",
    "M_analyze",
    "吸菸史",
    "喝酒史",
    "嚼檳榔史",
]
FORCE_CATEGORICAL = ["reconstruction_max"]
EXPECTED_NUMERIC = [
    "年齡",
    "ECOG",
    "是否進行手術",
    "tooth_extraction_any",
    "mandible_resection_any",
    "maxilla_resection_any",
    "has_L",
    "has_M",
    "CTV_H_dose_max",
    "CTV_M_dose_max",
    "CTV_L_dose_max",
    "DM",
    "HTN",
    "ESRD",
    "Osteoporosis",
]

# Keep this feature definition identical to the old thesis main pipeline.
CLINICAL_FEATURES = [
    "年齡", "性別", "ECOG", "Tumor location", "T_analyze", "N_analyze", "M_analyze",
    "吸菸史", "喝酒史", "嚼檳榔史", "DM", "HTN", "ESRD", "Osteoporosis",
]

TREATMENT_ADDED_FEATURES = [
    "是否進行手術", "tooth_extraction_any", "mandible_resection_any", "maxilla_resection_any",
    "reconstruction_max", "has_L", "has_M", "CTV_H_dose_max", "CTV_M_dose_max", "CTV_L_dose_max",
]

FULL_FEATURES = CLINICAL_FEATURES + TREATMENT_ADDED_FEATURES

ABLATION_REMOVE_GROUPS = {
    "full_model": [],
    "without_resection_features": [
        "mandible_resection_any", "maxilla_resection_any",
    ],
    "without_reconstruction_features": [
        "reconstruction_max",
    ],
    "without_radiation_features": [
        "has_L", "has_M", "CTV_H_dose_max", "CTV_M_dose_max", "CTV_L_dose_max",
    ],
    "without_extraction_features": [
        "tooth_extraction_any",
    ],
}

@dataclass
class ModelSpec:
    estimator: object
    scale_numeric: bool
    param_dist: Dict[str, list]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def load_data(data_path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(data_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if ID_COL not in df.columns:
        raise ValueError(f"Missing patient ID column: {ID_COL}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    df[ID_COL] = df[ID_COL].astype(str)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feature_cols = [
        c for c in df.columns
        if c not in [ID_COL, TARGET_COL] + DROP_NON_MODEL_COLS
    ]
    categorical_cols = [c for c in feature_cols if c in EXPECTED_CATEGORICAL or c in FORCE_CATEGORICAL]
    numeric_cols = [c for c in feature_cols if c in EXPECTED_NUMERIC and c not in categorical_cols]

    remaining = [c for c in feature_cols if c not in categorical_cols + numeric_cols]
    for c in remaining:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    tmp = X.copy()
    tmp[TARGET_COL] = 0
    numeric_cols, categorical_cols = infer_feature_types(tmp)
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])


def split_xy_groups(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    present = [c for c in feature_cols if c in df.columns]
    if len(present) == 0:
        raise ValueError("No requested feature columns are present in the data.")
    data = df.dropna(subset=[TARGET_COL]).copy()
    X = data[present].copy()
    y = pd.to_numeric(data[TARGET_COL], errors="coerce").astype(int)
    groups = data[ID_COL].astype(str)
    return X, y, groups


def patient_level_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    patient_label_group = pd.DataFrame({"patient": groups.astype(str), "y": y.to_numpy()}).groupby("patient")["y"]
    nunique = patient_label_group.nunique()
    if (nunique > 1).any():
        bad = nunique[nunique > 1].index.tolist()
        raise ValueError(f"Same ipatient has conflicting ORN_label values. Please fix before modeling: {bad[:10]}")

    patient_df = patient_label_group.max().reset_index()
    train_ids, test_ids = train_test_split(
        patient_df["patient"],
        test_size=test_size,
        random_state=random_state,
        stratify=patient_df["y"],
    )
    train_set = set(train_ids)
    test_set = set(test_ids)
    train_mask = groups.astype(str).isin(train_set)
    test_mask = groups.astype(str).isin(test_set)
    return (
        X.loc[train_mask].copy(),
        X.loc[test_mask].copy(),
        y.loc[train_mask].copy(),
        y.loc[test_mask].copy(),
        groups.loc[train_mask].copy(),
        groups.loc[test_mask].copy(),
    )


def make_group_cv(y: pd.Series, groups: pd.Series, n_splits: int, random_state: int) -> StratifiedGroupKFold:
    patient_labels = pd.DataFrame({"patient": groups.astype(str), "y": y.to_numpy()}).groupby("patient")["y"].max()
    min_class_count = int(patient_labels.value_counts().min())
    safe_splits = max(2, min(n_splits, min_class_count))
    return StratifiedGroupKFold(n_splits=safe_splits, shuffle=True, random_state=random_state)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) == 0:
            continue
        ece += abs(np.mean(y_prob[mask]) - np.mean(y_true[mask])) * (np.sum(mask) / len(y_prob))
    return float(ece)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return {
        "roc_auc": safe_auc(y_true, y_prob),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "precision_ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": compute_ece(y_true, y_prob),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_model_specs(y_train: pd.Series, random_state: int) -> Dict[str, ModelSpec]:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    specs: Dict[str, ModelSpec] = {
        "LogisticRegression": ModelSpec(
            estimator=LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
            scale_numeric=True,
            param_dist={
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l2"],
            },
        ),
        "RandomForest": ModelSpec(
            estimator=RandomForestClassifier(class_weight="balanced", random_state=random_state),
            scale_numeric=False,
            param_dist={
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [3, 4, 5],
                "model__min_samples_split": [4, 8, 12],
                "model__min_samples_leaf": [2, 4, 6],
            },
        ),
        "SVM": ModelSpec(
            estimator=SVC(probability=True, class_weight="balanced", random_state=random_state),
            scale_numeric=True,
            param_dist={
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", 0.1, 0.01],
                "model__kernel": ["rbf"],
            },
        ),
    }

    if HAS_XGBOOST:
        specs["XGBoost"] = ModelSpec(
            estimator=XGBClassifier(
                eval_metric="logloss",
                random_state=random_state,
                scale_pos_weight=scale_pos_weight,
            ),
            scale_numeric=False,
            param_dist={
                "model__n_estimators": [200, 300],
                "model__max_depth": [3, 4],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 0.9],
                "model__colsample_bytree": [0.8, 0.9],
            },
        )
    return specs


def build_pipeline(model_spec: ModelSpec, X_for_schema: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(X_for_schema, scale_numeric=model_spec.scale_numeric)),
        ("model", clone(model_spec.estimator)),
    ])


def tune_pipeline(
    pipe: Pipeline,
    param_dist: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_iter: int,
    n_splits: int,
    random_state: int,
    n_jobs: int,
) -> Pipeline:
    inner_cv = make_group_cv(y_train, groups_train, n_splits=n_splits, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=min(n_iter, max(1, int(np.prod([len(v) for v in param_dist.values()])))),
        scoring="roc_auc",
        cv=inner_cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train, groups=groups_train)
    best_pipe = search.best_estimator_
    best_pipe.best_params_for_report_ = search.best_params_  # type: ignore[attr-defined]
    best_pipe.best_inner_cv_score_for_report_ = float(search.best_score_)  # type: ignore[attr-defined]
    return best_pipe


def repeated_group_cv_scores(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    fold_id = 0
    for repeat in range(n_repeats):
        cv = make_group_cv(y, groups, n_splits=n_splits, random_state=random_state + repeat)
        for train_idx, test_idx in cv.split(X, y, groups=groups):
            fold_id += 1
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model = clone(pipe)
            model.fit(X_tr, y_tr)
            p_tr = model.predict_proba(X_tr)[:, 1]
            p_te = model.predict_proba(X_te)[:, 1]
            train_metrics = evaluate_probs(y_tr.to_numpy(), p_tr)
            test_metrics = evaluate_probs(y_te.to_numpy(), p_te)
            rows.append({
                "repeat": repeat + 1,
                "fold": fold_id,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            })
    return pd.DataFrame(rows)


def features_after_removal(full_features: Sequence[str], remove_features: Sequence[str], df: pd.DataFrame) -> List[str]:
    remove_set = set(remove_features)
    return [c for c in full_features if c in df.columns and c not in remove_set]


def make_ablation_manifest(df: pd.DataFrame, ablation_settings: Dict[str, Sequence[str]], output_dir: Path) -> pd.DataFrame:
    rows = []
    for setting, remove_features in ablation_settings.items():
        kept_features = features_after_removal(FULL_FEATURES, remove_features, df)
        for f in FULL_FEATURES:
            rows.append({
                "ablation_setting": setting,
                "feature": f,
                "present_in_data": f in df.columns,
                "is_removed": f in set(remove_features),
                "is_kept_for_model": f in kept_features,
                "missing_count": int(df[f].isna().sum()) if f in df.columns else None,
                "n_unique": int(df[f].nunique(dropna=True)) if f in df.columns else None,
            })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "ablation_feature_manifest.csv", index=False, encoding="utf-8-sig")
    return out


def evaluate_ablation_setting(
    df: pd.DataFrame,
    setting_name: str,
    remove_features: Sequence[str],
    selected_models: Sequence[str],
    output_dir: Path,
    test_size: float,
    n_splits: int,
    n_repeats: int,
    n_iter: int,
    random_state: int,
    n_jobs: int,
    tune: bool,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    kept_features = features_after_removal(FULL_FEATURES, remove_features, df)
    X, y, groups = split_xy_groups(df, kept_features)
    X_train, X_test, y_train, y_test, groups_train, groups_test = patient_level_train_test_split(
        X, y, groups, test_size=test_size, random_state=random_state
    )

    setting_dir = output_dir / setting_name
    ensure_dir(setting_dir)
    pd.DataFrame({
        "split": ["train", "test"],
        "n_rows": [len(X_train), len(X_test)],
        "n_patients": [groups_train.nunique(), groups_test.nunique()],
        "n_orn": [int(y_train.sum()), int(y_test.sum())],
        "n_non_orn": [int((y_train == 0).sum()), int((y_test == 0).sum())],
    }).to_csv(setting_dir / "holdout_split_summary.csv", index=False, encoding="utf-8-sig")

    model_specs_all = build_model_specs(y_train, random_state=random_state)
    fitted_pipes: Dict[str, Pipeline] = {}
    rows = []

    for model_name in selected_models:
        if model_name not in model_specs_all:
            print(f"[Skip] {model_name} is not available. Available models: {sorted(model_specs_all.keys())}")
            continue

        print(f"[Run] {setting_name} | {model_name} | n_features={len(kept_features)}")
        spec = model_specs_all[model_name]
        base_pipe = build_pipeline(spec, X_train)

        if tune:
            pipe = tune_pipeline(
                base_pipe,
                spec.param_dist,
                X_train,
                y_train,
                groups_train,
                n_iter=n_iter,
                n_splits=n_splits,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            best_params = getattr(pipe, "best_params_for_report_", {})
            best_inner = getattr(pipe, "best_inner_cv_score_for_report_", np.nan)
        else:
            pipe = base_pipe
            best_params = {}
            best_inner = np.nan

        X_all = pd.concat([X_train, X_test], axis=0)
        y_all = pd.concat([y_train, y_test], axis=0)
        groups_all = pd.concat([groups_train, groups_test], axis=0)
        cv_df = repeated_group_cv_scores(
            pipe, X_all, y_all, groups_all,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        cv_df.to_csv(setting_dir / f"repeated_cv_scores_{model_name}.csv", index=False, encoding="utf-8-sig")

        fitted = clone(pipe)
        fitted.fit(X_train, y_train)
        y_prob = fitted.predict_proba(X_test)[:, 1]
        holdout_metrics = evaluate_probs(y_test.to_numpy(), y_prob)
        fitted_pipes[model_name] = fitted

        pd.DataFrame([holdout_metrics]).to_csv(setting_dir / f"holdout_metrics_{model_name}.csv", index=False, encoding="utf-8-sig")

        rows.append({
            "ablation_setting": setting_name,
            "model": model_name,
            "n_features": len(kept_features),
            "removed_features": json.dumps([f for f in remove_features if f in df.columns], ensure_ascii=False),
            "kept_features": json.dumps(kept_features, ensure_ascii=False),
            "tuned": bool(tune),
            "inner_cv_best_roc_auc": best_inner,
            "best_params": json.dumps(best_params, ensure_ascii=False),
            **{f"holdout_{k}": v for k, v in holdout_metrics.items()},
            "repeated_cv_roc_auc_mean": float(cv_df["test_roc_auc"].mean()),
            "repeated_cv_roc_auc_std": float(cv_df["test_roc_auc"].std(ddof=1)),
            "repeated_cv_pr_auc_mean": float(cv_df["test_pr_auc"].mean()),
            "repeated_cv_pr_auc_std": float(cv_df["test_pr_auc"].std(ddof=1)),
            "repeated_cv_brier_mean": float(cv_df["test_brier"].mean()),
            "repeated_cv_brier_std": float(cv_df["test_brier"].std(ddof=1)),
            "repeated_cv_ece_mean": float(cv_df["test_ece"].mean()),
            "repeated_cv_ece_std": float(cv_df["test_ece"].std(ddof=1)),
            "optimism_gap_train_minus_test_auc": float(cv_df["train_roc_auc"].mean() - cv_df["test_roc_auc"].mean()),
        })

    out = pd.DataFrame(rows)
    out.to_csv(setting_dir / "ablation_results_this_setting.csv", index=False, encoding="utf-8-sig")
    return out, fitted_pipes


def compute_topk_features_by_permutation(
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    test_size: float,
    n_splits: int,
    n_iter: int,
    random_state: int,
    n_jobs: int,
    tune: bool,
    k: int = 5,
    n_repeats: int = 30,
) -> List[str]:
    """Derive top-k raw features from hold-out permutation importance of the baseline full model.

    This is intended as a sensitivity check. It should not be described as causal feature selection.
    If you want a stricter design, pass manually chosen top features by editing ABLATION_REMOVE_GROUPS.
    """
    kept_features = features_after_removal(FULL_FEATURES, [], df)
    X, y, groups = split_xy_groups(df, kept_features)
    X_train, X_test, y_train, y_test, groups_train, _ = patient_level_train_test_split(
        X, y, groups, test_size=test_size, random_state=random_state
    )
    specs = build_model_specs(y_train, random_state=random_state)
    if model_name not in specs:
        raise ValueError(f"Cannot compute top-k features. Model not available: {model_name}")
    spec = specs[model_name]
    pipe = build_pipeline(spec, X_train)
    if tune:
        pipe = tune_pipeline(
            pipe,
            spec.param_dist,
            X_train,
            y_train,
            groups_train,
            n_iter=n_iter,
            n_splits=n_splits,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    pipe.fit(X_train, y_train)

    r = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(output_dir / f"baseline_permutation_importance_for_top{k}_{model_name}.csv", index=False, encoding="utf-8-sig")

    top = importance_df[importance_df["importance_mean"] > 0].head(k)["feature"].tolist()
    if len(top) == 0:
        top = importance_df.head(k)["feature"].tolist()
    save_text(output_dir / f"auto_top{k}_features.txt", "\n".join(top))
    return top


def summarize_best_results(all_results: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if all_results.empty:
        raise ValueError("No ablation results were generated.")

    best = (
        all_results.sort_values(["ablation_setting", "repeated_cv_roc_auc_mean"], ascending=[True, False])
        .groupby("ablation_setting", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    baseline_rows = best[best["ablation_setting"] == "full_model"]
    if len(baseline_rows) == 0:
        raise ValueError("Missing full_model baseline results.")
    baseline = baseline_rows.iloc[0]

    best["delta_auc_vs_full"] = best["repeated_cv_roc_auc_mean"] - baseline["repeated_cv_roc_auc_mean"]
    best["auc_drop_vs_full"] = baseline["repeated_cv_roc_auc_mean"] - best["repeated_cv_roc_auc_mean"]
    best["delta_brier_vs_full"] = best["repeated_cv_brier_mean"] - baseline["repeated_cv_brier_mean"]
    best["delta_ece_vs_full"] = best["repeated_cv_ece_mean"] - baseline["repeated_cv_ece_mean"]
    best = best.sort_values("auc_drop_vs_full", ascending=False).reset_index(drop=True)

    keep_cols = [
        "ablation_setting", "model", "n_features", "removed_features",
        "repeated_cv_roc_auc_mean", "repeated_cv_roc_auc_std", "delta_auc_vs_full", "auc_drop_vs_full",
        "repeated_cv_brier_mean", "repeated_cv_brier_std", "delta_brier_vs_full",
        "repeated_cv_ece_mean", "repeated_cv_ece_std", "delta_ece_vs_full",
        "holdout_roc_auc", "holdout_brier", "holdout_ece",
        "best_params",
    ]
    keep_cols = [c for c in keep_cols if c in best.columns]
    summary = best[keep_cols].copy()
    summary.to_csv(output_dir / "ablation_summary_best_by_setting.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_ablation_auc(summary: pd.DataFrame, output_dir: Path) -> None:
    df_plot = summary.copy().sort_values("repeated_cv_roc_auc_mean", ascending=True)
    y = np.arange(len(df_plot))

    plt.figure(figsize=(8, max(4.5, 0.45 * len(df_plot))))
    plt.barh(y, df_plot["repeated_cv_roc_auc_mean"])
    plt.yticks(y, df_plot["ablation_setting"])
    plt.xlabel("Repeated-CV ROC-AUC")
    plt.title("Feature-removal ablation: ROC-AUC")
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_auc_bar.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "ablation_auc_bar.pdf", bbox_inches="tight")
    plt.close()

    df_drop = summary.copy().sort_values("auc_drop_vs_full", ascending=True)
    plt.figure(figsize=(8, max(4.5, 0.45 * len(df_drop))))
    plt.barh(np.arange(len(df_drop)), df_drop["auc_drop_vs_full"])
    plt.yticks(np.arange(len(df_drop)), df_drop["ablation_setting"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("ROC-AUC drop compared with full model")
    plt.title("Feature-removal ablation: performance drop")
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_auc_drop_bar.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "ablation_auc_drop_bar.pdf", bbox_inches="tight")
    plt.close()


def make_report_text(df: pd.DataFrame, summary: pd.DataFrame) -> str:
    baseline = summary[summary["ablation_setting"] == "full_model"].iloc[0]
    largest_drop = summary[summary["ablation_setting"] != "full_model"].sort_values("auc_drop_vs_full", ascending=False).head(1)
    lines = []
    lines.append("Important-feature Removal Ablation Summary")
    lines.append("=" * 80)
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Total patients: {df[ID_COL].nunique()}")
    lines.append(f"ORN positive rows: {int((df[TARGET_COL] == 1).sum())}")
    lines.append(f"ORN negative rows: {int((df[TARGET_COL] == 0).sum())}")
    lines.append("")
    lines.append("Baseline full model:")
    lines.append(
        f"- {baseline['model']}: repeated-CV ROC-AUC = "
        f"{baseline['repeated_cv_roc_auc_mean']:.3f} ± {baseline['repeated_cv_roc_auc_std']:.3f}, "
        f"Brier = {baseline['repeated_cv_brier_mean']:.3f}, ECE = {baseline['repeated_cv_ece_mean']:.3f}"
    )
    lines.append("")
    lines.append("Ablation settings:")
    for _, row in summary.iterrows():
        if row["ablation_setting"] == "full_model":
            continue
        lines.append(
            f"- {row['ablation_setting']}: ROC-AUC = {row['repeated_cv_roc_auc_mean']:.3f} ± "
            f"{row['repeated_cv_roc_auc_std']:.3f}, ΔAUC = {row['delta_auc_vs_full']:.3f}, "
            f"AUC drop = {row['auc_drop_vs_full']:.3f}"
        )
    lines.append("")
    if len(largest_drop) > 0:
        r = largest_drop.iloc[0]
        lines.append(
            f"Largest observed AUC drop: {r['ablation_setting']} "
            f"(drop = {r['auc_drop_vs_full']:.3f})."
        )
    lines.append("")
    lines.append("Suggested thesis wording:")
    lines.append(
        "To examine the dependence of the full model on clinically important variables, "
        "an additional feature-removal ablation experiment was conducted. The training and validation "
        "procedure was kept identical to the main pipeline, while selected feature groups were removed "
        "from the full feature set. Performance changes were quantified using patient-level repeated "
        "cross-validation ROC-AUC, Brier score, and ECE. A decrease in ROC-AUC after feature removal "
        "was interpreted as evidence that the removed feature group provided complementary predictive information. "
        "If performance did not decrease substantially, the result was interpreted cautiously because correlated "
        "clinical and treatment variables may provide substitute information."
    )
    return "\n".join(lines)


def parse_models(raw: Sequence[str]) -> List[str]:
    out = []
    for item in raw:
        for x in str(item).split(","):
            x = x.strip()
            if x:
                out.append(x)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Additional important-feature removal ablation for ORN structured-data model.")
    parser.add_argument("--data_path", type=str, default="data_v3.1.xlsx")
    parser.add_argument("--sheet_name", type=str, default="model_full_pre_orn")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--models", nargs="+", default=["LogisticRegression"], help="Model names. Example: --models LogisticRegression RandomForest")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=20)
    parser.add_argument("--n_iter", type=int, default=20, help="RandomizedSearchCV iterations per model/setting.")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--no_tune", action="store_true", help="Disable hyperparameter tuning and use default model settings.")
    parser.add_argument("--auto_topk", type=int, default=5, help="Automatically add without_topK_permutation_features setting. Use 0 to disable.")
    parser.add_argument("--topk_model", type=str, default="LogisticRegression", help="Model used to derive automatic top-k features.")
    parser.add_argument("--permutation_repeats", type=int, default=30)
    args = parser.parse_args()

    models = parse_models(args.models)
    tune = not args.no_tune

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path.resolve()}")

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else data_path.resolve().parent / f"orn_feature_removal_ablation_outputs_{run_tag}"
    ensure_dir(output_dir)

    df = normalize_patient_id(load_data(data_path, args.sheet_name))

    duplicated = df[df.duplicated(ID_COL, keep=False)].sort_values(ID_COL)
    if len(duplicated) > 0:
        duplicated.to_csv(output_dir / "duplicated_ipatient_rows.csv", index=False, encoding="utf-8-sig")
        print(f"[Info] duplicated ipatient rows detected: {len(duplicated)} rows / {duplicated[ID_COL].nunique()} patients. Patient-level split is used.")

    ablation_settings: Dict[str, Sequence[str]] = dict(ABLATION_REMOVE_GROUPS)
    if args.auto_topk and args.auto_topk > 0:
        print(f"[Info] deriving top-{args.auto_topk} features by baseline permutation importance using {args.topk_model} ...")
        topk = compute_topk_features_by_permutation(
            df=df,
            model_name=args.topk_model,
            output_dir=output_dir,
            test_size=args.test_size,
            n_splits=args.n_splits,
            n_iter=args.n_iter,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            tune=tune,
            k=args.auto_topk,
            n_repeats=args.permutation_repeats,
        )
        ablation_settings[f"without_top{args.auto_topk}_permutation_features"] = topk
        print(f"[Info] top-{args.auto_topk} features: {topk}")

    make_ablation_manifest(df, ablation_settings, output_dir)

    all_rows = []
    for setting_name, remove_features in ablation_settings.items():
        result_df, _ = evaluate_ablation_setting(
            df=df,
            setting_name=setting_name,
            remove_features=remove_features,
            selected_models=models,
            output_dir=output_dir,
            test_size=args.test_size,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            n_iter=args.n_iter,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            tune=tune,
        )
        all_rows.append(result_df)

    all_results = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    all_results.to_csv(output_dir / "ablation_results_all_models.csv", index=False, encoding="utf-8-sig")

    summary = summarize_best_results(all_results, output_dir)
    plot_ablation_auc(summary, output_dir)
    report_text = make_report_text(df, summary)
    save_text(output_dir / "ablation_report_text.txt", report_text)

    config = vars(args).copy()
    config["models"] = models
    config["tune"] = tune
    config["ablation_settings"] = {k: list(v) for k, v in ablation_settings.items()}
    with open(output_dir / "ablation_run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nAblation experiment finished.")
    print(f"Outputs saved to: {output_dir}")
    print("Main table: ablation_summary_best_by_setting.csv")


if __name__ == "__main__":
    main()
