from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.base import clone
from sklearn.calibration import calibration_curve
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

# ============================================================
# 0. 專案設定
# ============================================================
PROJECT_DIR = Path(".")
DATA_PATH = PROJECT_DIR / "data_v3.1.xlsx"
PIPELINE_JSON = PROJECT_DIR / "orn_experiment_pipeline.json"
SHEET_NAME = "model_full_pre_orn"

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = PROJECT_DIR / f"orn_thesis_final_outputs_{RUN_TAG}"
EDA_DIR = OUTPUT_DIR / "EDA"
MODEL_COMP_DIR = OUTPUT_DIR / "model_comparison"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "report_ready"
for p in [OUTPUT_DIR, EDA_DIR, MODEL_COMP_DIR, FIG_DIR, TABLE_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BOOTSTRAP = 1000
N_REPEATS_CV = 20
N_SPLITS_CV = 5
TARGET_COL = "ORN_label"
ID_COL = "ipatient"

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

CLINICAL_FEATURES = [
    "年齡", "性別", "ECOG", "Tumor location", "T_analyze", "N_analyze", "M_analyze",
    "吸菸史", "喝酒史", "嚼檳榔史", "DM", "HTN", "ESRD", "Osteoporosis",
]

TREATMENT_ADDED_FEATURES = [
    "是否進行手術", "tooth_extraction_any", "mandible_resection_any", "maxilla_resection_any",
    "reconstruction_max", "has_L", "has_M", "CTV_H_dose_max", "CTV_M_dose_max", "CTV_L_dose_max",
]


# ============================================================
# Figure style: black-and-white friendly output
# ============================================================
# Use different line styles and markers rather than relying on color only.
# This keeps ROC and calibration plots distinguishable after grayscale printing.
BW_MODEL_STYLES = {
    "LogisticRegression": {"linestyle": "-",  "marker": "o", "color": "black"},
    "RandomForest":       {"linestyle": "--", "marker": "s", "color": "0.25"},
    "SVM":                {"linestyle": "-.", "marker": "^", "color": "0.45"},
    "XGBoost":            {"linestyle": ":",  "marker": "D", "color": "0.10"},
}
BW_FALLBACK_STYLES = [
    {"linestyle": "-",  "marker": "o", "color": "black"},
    {"linestyle": "--", "marker": "s", "color": "0.25"},
    {"linestyle": "-.", "marker": "^", "color": "0.45"},
    {"linestyle": ":",  "marker": "D", "color": "0.10"},
    {"linestyle": (0, (3, 1, 1, 1)), "marker": "v", "color": "0.60"},
]


def get_bw_style(model_name: str, idx: int = 0) -> Dict:
    style = BW_MODEL_STYLES.get(model_name, BW_FALLBACK_STYLES[idx % len(BW_FALLBACK_STYLES)]).copy()
    return style


def format_probability_axes(ax, equal_aspect: bool = False) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticks(np.linspace(0, 1, 11), minor=True)
    ax.set_yticks(np.linspace(0, 1, 11), minor=True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.35)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


def save_bw_figure(path: Path, dpi: int = 300) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    # Save a vector copy for thesis / conference layout.
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

# ============================================================
# 1. 基本工具
# ============================================================
def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL]]
    categorical_cols = [c for c in feature_cols if c in EXPECTED_CATEGORICAL or c in FORCE_CATEGORICAL]
    numeric_cols = [c for c in feature_cols if c in EXPECTED_NUMERIC]
    remaining = [c for c in feature_cols if c not in categorical_cols + numeric_cols]
    for c in remaining:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    temp = X.copy()
    temp[TARGET_COL] = 0
    numeric_cols, categorical_cols = infer_feature_types(temp)
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        *(([("scaler", StandardScaler())]) if scale_numeric else []),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])


def split_xy(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.dropna(subset=[TARGET_COL]).copy()
    y = pd.to_numeric(data[TARGET_COL], errors="coerce").astype(int)
    if feature_cols is None:
        X = data.drop(columns=[ID_COL, TARGET_COL], errors="ignore")
    else:
        feature_cols = [c for c in feature_cols if c in data.columns]
        X = data[feature_cols].copy()
    return X, y


def get_feature_names_from_pipeline(pipe: Pipeline) -> List[str]:
    names = pipe.named_steps["preprocess"].get_feature_names_out().tolist()
    return [n.replace("num__", "").replace("cat__", "") for n in names]


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        m = ids == i
        if np.sum(m) == 0:
            continue
        ece += abs(np.mean(y_prob[m]) - np.mean(y_true[m])) * (np.sum(m) / len(y_prob))
    return float(ece)


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstrap: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall_Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Precision_PPV": precision_score(y_true, y_pred, zero_division=0),
        "NPV": npv,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Brier": brier_score_loss(y_true, y_prob),
        "ECE": compute_ece(y_true, y_prob),
    }

# ============================================================
# 2. EDA / 報告表格
# ============================================================
def make_dataset_summary(df: pd.DataFrame) -> str:
    lines = [
        "ORN Dataset Summary",
        "=" * 80,
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"ORN positive (1): {int((df[TARGET_COL] == 1).sum())}",
        f"ORN negative (0): {int((df[TARGET_COL] == 0).sum())}",
        f"Prevalence: {df[TARGET_COL].mean():.3f}",
        "",
    ]
    for c in df.columns:
        lines.append(f"- {c} | dtype={df[c].dtype} | missing={int(df[c].isna().sum())}")
    return "\n".join(lines)


def save_missingness(df: pd.DataFrame) -> None:
    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": [df[c].isna().sum() for c in df.columns],
        "missing_ratio": [df[c].isna().mean() for c in df.columns],
    }).sort_values(["missing_ratio", "missing_count"], ascending=False)
    missing.to_csv(EDA_DIR / "missingness_summary.csv", index=False, encoding="utf-8-sig")


def create_table1(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols, categorical_cols = infer_feature_types(df)
    g0 = df[df[TARGET_COL] == 0]
    g1 = df[df[TARGET_COL] == 1]
    rows = []

    for col in numeric_cols:
        x0 = pd.to_numeric(g0[col], errors="coerce").dropna()
        x1 = pd.to_numeric(g1[col], errors="coerce").dropna()
        p = np.nan
        if len(x0) > 0 and len(x1) > 0:
            try:
                _, p = mannwhitneyu(x0, x1, alternative="two-sided")
            except Exception:
                pass
        rows.append({
            "variable": col,
            "summary_non_ORN": f"{x0.mean():.3f} ± {x0.std():.3f}" if len(x0) else "NA",
            "summary_ORN": f"{x1.mean():.3f} ± {x1.std():.3f}" if len(x1) else "NA",
            "p_value": p,
            "type": "numeric",
        })

    for col in categorical_cols:
        ct = pd.crosstab(df[col], df[TARGET_COL])
        p = np.nan
        try:
            if ct.shape == (2, 2) and (ct.values < 5).any():
                _, p = fisher_exact(ct.values)
            else:
                _, p, _, _ = chi2_contingency(ct)
        except Exception:
            pass
        for level in ct.index:
            n0 = int(ct.loc[level, 0]) if 0 in ct.columns else 0
            n1 = int(ct.loc[level, 1]) if 1 in ct.columns else 0
            d0 = len(g0) if len(g0) else 1
            d1 = len(g1) if len(g1) else 1
            rows.append({
                "variable": f"{col} = {level}",
                "summary_non_ORN": f"{n0} ({n0 / d0:.1%})",
                "summary_ORN": f"{n1} ({n1 / d1:.1%})",
                "p_value": p,
                "type": "categorical",
            })
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "Table1_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    return out


def univariate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    table1 = create_table1(df)
    out = table1[["variable", "type", "p_value"]].copy().sort_values("p_value", na_position="last")
    out.to_csv(TABLE_DIR / "univariate_analysis.csv", index=False, encoding="utf-8-sig")
    return out


def plot_correlation(df: pd.DataFrame) -> None:
    numeric_cols, _ = infer_feature_types(df)
    corr = df[numeric_cols].apply(pd.to_numeric, errors="coerce").corr()
    corr.to_csv(EDA_DIR / "correlation_matrix.csv", encoding="utf-8-sig")
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_matrix.png", dpi=200)
    plt.close()

# ============================================================
# 3. 模型工具
# ============================================================
def build_models(y_train: pd.Series) -> Dict[str, Dict]:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    models = {
        "LogisticRegression": {
            "model": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "scale_numeric": True,
            "param_dist": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l2"],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "scale_numeric": False,
            "param_dist": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [3, 4, 5],
                "model__min_samples_split": [4, 8, 12],
                "model__min_samples_leaf": [2, 4, 6],
            },
        },
        "SVM": {
            "model": SVC(
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "scale_numeric": True,
            "param_dist": {
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", 0.1, 0.01],
                "model__kernel": ["rbf"],
            },
        },
    }

    if HAS_XGBOOST:
        models["XGBoost"] = {
            "model": XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                scale_pos_weight=scale_pos_weight,
            ),
            "scale_numeric": False,
            "param_dist": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [3, 4],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 0.9],
                "model__colsample_bytree": [0.8, 0.9],
            },
        }

    return models


def repeated_cv_scores(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS_CV, n_repeats=N_REPEATS_CV, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "neg_brier_score": "neg_brier_score",
    }
    cv_df = pd.DataFrame(cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=None))
    cv_df["train_brier"] = -cv_df["train_neg_brier_score"]
    cv_df["test_brier"] = -cv_df["test_neg_brier_score"]

    ece_rows = []
    for i, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        m = clone(pipe)
        m.fit(X_tr, y_tr)
        p_tr = m.predict_proba(X_tr)[:, 1]
        p_te = m.predict_proba(X_te)[:, 1]
        ece_rows.append({
            "fold": i,
            "train_ece": compute_ece(y_tr.to_numpy(), p_tr),
            "test_ece": compute_ece(y_te.to_numpy(), p_te),
        })
    return cv_df, pd.DataFrame(ece_rows)


def plot_roc_pr_calibration(model_name: str, group_name: str, y_test: np.ndarray, y_prob: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
    style = get_bw_style(model_name)

    plt.figure(figsize=(5.6, 4.8))
    ax = plt.gca()
    ax.plot(
        fpr,
        tpr,
        drawstyle="steps-post",
        linewidth=2.0,
        markersize=5.0,
        markerfacecolor="white",
        markevery=max(1, len(fpr) // 8),
        label=f"{model_name} | AUC={roc_auc_score(y_test, y_prob):.3f}",
        **style,
    )
    ax.plot([0, 1], [0, 1], linestyle=(0, (2, 2)), color="0.55", linewidth=1.4, label="Chance")
    ax.set_title(f"ROC Curve - {group_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    format_probability_axes(ax, equal_aspect=True)
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    save_bw_figure(FIG_DIR / f"roc_{group_name}_{model_name}.png")

    plt.figure(figsize=(5.6, 4.8))
    ax = plt.gca()
    ax.plot(
        recall,
        precision,
        linewidth=2.0,
        markersize=5.0,
        markerfacecolor="white",
        markevery=max(1, len(recall) // 8),
        label=f"{model_name} | AP={average_precision_score(y_test, y_prob):.3f}",
        **style,
    )
    ax.set_title(f"Precision-Recall Curve - {group_name}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    format_probability_axes(ax, equal_aspect=False)
    ax.legend(frameon=True, fontsize=8, loc="lower left")
    save_bw_figure(FIG_DIR / f"pr_{group_name}_{model_name}.png")

    plt.figure(figsize=(5.6, 4.8))
    ax = plt.gca()
    ax.plot(
        mean_pred,
        frac_pos,
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        label=f"{model_name}",
        **style,
    )
    ax.plot([0, 1], [0, 1], linestyle=(0, (2, 2)), color="0.55", linewidth=1.4, label="Perfect calibration")
    ax.set_title(f"Calibration Curve - {group_name}")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Event Rate")
    format_probability_axes(ax, equal_aspect=True)
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    save_bw_figure(FIG_DIR / f"calibration_{group_name}_{model_name}.png")


def plot_combined_roc(model_probs: Dict[str, np.ndarray], y_true: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7.0, 5.6))
    ax = plt.gca()
    for idx, (name, prob) in enumerate(model_probs.items()):
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        style = get_bw_style(name, idx)
        ax.plot(
            fpr,
            tpr,
            drawstyle="steps-post",
            linewidth=2.0,
            markersize=5.0,
            markerfacecolor="white",
            markevery=max(1, len(fpr) // 8),
            label=f"{name} (AUC={auc:.3f})",
            **style,
        )
    ax.plot([0, 1], [0, 1], linestyle=(0, (2, 2)), color="0.55", linewidth=1.4, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    format_probability_axes(ax, equal_aspect=True)
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    save_bw_figure(out_path)


def plot_combined_calibration(model_probs: Dict[str, np.ndarray], y_true: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7.0, 5.6))
    ax = plt.gca()
    for idx, (name, prob) in enumerate(model_probs.items()):
        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
        style = get_bw_style(name, idx)
        ax.plot(
            mean_pred,
            frac_pos,
            linewidth=2.0,
            markersize=6.0,
            markerfacecolor="white",
            label=name,
            **style,
        )
    ax.plot([0, 1], [0, 1], linestyle=(0, (2, 2)), color="0.55", linewidth=1.4, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Event Rate")
    ax.set_title(title)
    format_probability_axes(ax, equal_aspect=True)
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    save_bw_figure(out_path)


def simple_decision_curve(model_name: str, group_name: str, y_test: np.ndarray, y_prob: np.ndarray) -> None:
    thresholds = np.linspace(0.05, 0.95, 91)
    prevalence = np.mean(y_test)
    model_nb, all_nb = [], []
    for pt in thresholds:
        pred = (y_prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        n = len(y_test)
        model_nb.append((tp / n) - (fp / n) * (pt / (1 - pt)))
        all_nb.append(prevalence - (1 - prevalence) * (pt / (1 - pt)))
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, model_nb, label=model_name)
    plt.plot(thresholds, all_nb, linestyle="--", label="Treat all")
    plt.plot(thresholds, np.zeros_like(thresholds), linestyle=":", label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"Decision Curve - {group_name} - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"decision_curve_{group_name}_{model_name}.png", dpi=200)
    plt.close()


def permutation_importance_table(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, out_csv: Path, out_png: Path) -> None:
    r = permutation_importance(pipe, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="roc_auc")
    out = pd.DataFrame({
        "feature": list(X_test.columns),
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    top = out.head(15)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["importance_mean"][::-1])
    plt.title(out_png.stem)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def shap_analysis(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    out_csv: Path,
    out_png: Path,
    out_beeswarm_png: Optional[Path] = None,
) -> None:
    if not HAS_SHAP:
        save_text(out_csv.with_suffix(".txt"), "SHAP 未安裝，已略過。")
        return

    model = pipe.named_steps["model"]
    pre = pipe.named_steps["preprocess"]
    Xt_train = pre.transform(X_train)
    Xt_test = pre.transform(X_test)
    feature_names = get_feature_names_from_pipeline(pipe)

    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_out = explainer.shap_values(Xt_test)
            values = shap_out[1] if isinstance(shap_out, list) else shap_out
        else:
            try:
                explainer = shap.LinearExplainer(model, Xt_train, feature_names=feature_names)
                shap_out = explainer.shap_values(Xt_test)
                values = shap_out[1] if isinstance(shap_out, list) else shap_out
            except Exception:
                explainer = shap.Explainer(model, Xt_train, feature_names=feature_names)
                shap_out = explainer(Xt_test)
                values = shap_out.values

        mean_abs = np.abs(values).mean(axis=0)
        out = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False)
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")

        top = out.head(20)
        plt.figure(figsize=(8, 7))
        plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
        plt.title(out_png.stem)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        if out_beeswarm_png is not None:
            plt.figure(figsize=(8, 7))
            shap.summary_plot(
                values,
                features=Xt_test,
                feature_names=feature_names,
                max_display=20,
                show=False,
            )
            plt.title(out_beeswarm_png.stem)
            plt.tight_layout()
            plt.savefig(out_beeswarm_png, dpi=200, bbox_inches="tight")
            plt.close()

    except Exception as e:
        save_text(out_csv.with_suffix(".txt"), f"SHAP 失敗：{e}")

# ============================================================
# 4. 分層模型流程
# ============================================================
@dataclass
class ModelRunResult:
    model_name: str
    y_prob: np.ndarray
    test_metrics: Dict[str, float]
    bootstrap_auc_mean: float
    bootstrap_auc_ci_low: float
    bootstrap_auc_ci_high: float
    cv_mean: Dict[str, float]
    cv_std: Dict[str, float]
    cv_ece_mean: float
    cv_ece_std: float


def run_model_group(df: pd.DataFrame, group_name: str, feature_cols: List[str]) -> pd.DataFrame:
    X, y = split_xy(df, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    models = build_models(y_train)
    group_dir = MODEL_COMP_DIR / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    results = []
    model_probs = {}
    for model_name, meta in models.items():
        print(f"Running {group_name} - {model_name} ...")
        pipe = Pipeline([
            ("preprocess", build_preprocessor(X_train, scale_numeric=meta["scale_numeric"])),
            ("model", meta["model"]),
        ])
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=meta["param_dist"],
            n_iter=20,
            scoring="roc_auc",
            cv=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)
        pipe = search.best_estimator_

        pd.DataFrame([{
            "feature_group": group_name,
            "model": model_name,
            "best_score_cv_auc": search.best_score_,
            "best_params": json.dumps(search.best_params_, ensure_ascii=False),
        }]).to_csv(
            group_dir / f"best_params_{model_name}.csv",
            index=False,
            encoding="utf-8-sig"
        )
        X_all = pd.concat([X_train, X_test], axis=0)
        y_all = pd.concat([y_train, y_test], axis=0)
        cv_df, ece_df = repeated_cv_scores(pipe, X_all, y_all)
        cv_df.to_csv(group_dir / f"repeated_cv_scores_{model_name}.csv", index=False, encoding="utf-8-sig")
        ece_df.to_csv(group_dir / f"repeated_cv_ece_{model_name}.csv", index=False, encoding="utf-8-sig")

        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        model_probs[model_name] = y_prob
        test_metrics = evaluate_threshold_metrics(y_test.to_numpy(), y_prob)
        b_mean, b_low, b_high = bootstrap_auc_ci(y_test.to_numpy(), y_prob, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE)

        pd.DataFrame([test_metrics]).to_csv(group_dir / f"test_metrics_{model_name}.csv", index=False, encoding="utf-8-sig")
        plot_roc_pr_calibration(model_name, group_name, y_test.to_numpy(), y_prob)
        simple_decision_curve(model_name, group_name, y_test.to_numpy(), y_prob)
        permutation_importance_table(
            pipe, X_test, y_test,
            group_dir / f"permutation_importance_{model_name}.csv",
            group_dir / f"permutation_importance_{model_name}.png",
        )
        shap_analysis(
            pipe, X_train, X_test,
            group_dir / f"shap_importance_{model_name}.csv",
            group_dir / f"shap_importance_{model_name}.png",
            group_dir / f"shap_beeswarm_{model_name}.png",
        )

        results.append({
            "feature_group": group_name,
            "n_features": X.shape[1],
            "model": model_name,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "bootstrap_auc_mean": b_mean,
            "bootstrap_auc_ci_low": b_low,
            "bootstrap_auc_ci_high": b_high,
            "repeated_cv_test_roc_auc_mean": cv_df["test_roc_auc"].mean(),
            "repeated_cv_test_roc_auc_std": cv_df["test_roc_auc"].std(),
            "repeated_cv_test_pr_auc_mean": cv_df["test_average_precision"].mean(),
            "repeated_cv_test_pr_auc_std": cv_df["test_average_precision"].std(),
            "repeated_cv_test_brier_mean": cv_df["test_brier"].mean(),
            "repeated_cv_test_brier_std": cv_df["test_brier"].std(),
            "repeated_cv_test_ece_mean": ece_df["test_ece"].mean(),
            "repeated_cv_test_ece_std": ece_df["test_ece"].std(),
            "optimism_gap_train_minus_test_roc_auc": cv_df["train_roc_auc"].mean() - cv_df["test_roc_auc"].mean(),
        })

    plot_combined_roc(model_probs, y_test.to_numpy(), group_dir / f"combined_roc_{group_name}.png", f"ROC Comparison - {group_name}")
    plot_combined_calibration(model_probs, y_test.to_numpy(), group_dir / f"combined_calibration_{group_name}.png", f"Calibration Comparison - {group_name}")

    out = pd.DataFrame(results).sort_values("repeated_cv_test_roc_auc_mean", ascending=False)
    out.to_csv(group_dir / f"model_comparison_{group_name}.csv", index=False, encoding="utf-8-sig")
    return out

# ============================================================
# 5. 論文 / 醫師報告摘要
# ============================================================
def make_progress_summary(df: pd.DataFrame, uni_df: pd.DataFrame, all_results: pd.DataFrame) -> str:
    best_full = all_results[all_results["feature_group"] == "full_model"].sort_values("repeated_cv_test_roc_auc_mean", ascending=False).iloc[0]
    best_overall = all_results.sort_values("repeated_cv_test_roc_auc_mean", ascending=False).iloc[0]
    top_uni = uni_df.dropna(subset=["p_value"]).sort_values("p_value").head(10)
    lines = []
    lines.append("ORN Progress Summary for Physician Meeting")
    lines.append("=" * 80)
    lines.append(f"Dataset: {len(df)} patients, ORN={int((df[TARGET_COL]==1).sum())}, non-ORN={int((df[TARGET_COL]==0).sum())}, prevalence={df[TARGET_COL].mean():.3f}")
    lines.append("")
    lines.append("1. Most informative variables in univariate screening")
    for _, r in top_uni.iterrows():
        lines.append(f"- {r['variable']} (p={r['p_value']:.4f})")
    lines.append("")
    lines.append("2. Model comparison by feature group")
    for group in ["clinical_model", "treatment_model", "full_model"]:
        sub = all_results[all_results["feature_group"] == group].sort_values("repeated_cv_test_roc_auc_mean", ascending=False)
        if len(sub) == 0:
            continue
        b = sub.iloc[0]
        lines.append(
            f"- {group}: best={b['model']}, CV ROC-AUC={b['repeated_cv_test_roc_auc_mean']:.3f} ± {b['repeated_cv_test_roc_auc_std']:.3f}, "
            f"Brier={b['repeated_cv_test_brier_mean']:.3f}, ECE={b['repeated_cv_test_ece_mean']:.3f}"
        )
    lines.append("")
    lines.append("3. Key interpretation")
    lines.append(f"- Best overall model: {best_overall['feature_group']} / {best_overall['model']}")
    lines.append(f"- Best full treatment-aware model: {best_full['model']}")
    lines.append("- Clinical-only model reflects pre-treatment baseline stratification capability.")
    lines.append("- Treatment model quantifies added value from surgery / extraction / RT dose information.")
    lines.append("- Full model reflects the strongest retrospective risk stratification scenario.")
    lines.append("")
    lines.append("4. Cautions")
    lines.append("- If any treatment variable was recorded after ORN onset, it must be removed before final manuscript analysis.")
    lines.append("- Repeated-CV should be emphasized over single hold-out AUC.")
    lines.append("- External validation is still needed before clinical deployment.")
    return "\n".join(lines)


def make_thesis_summary(df: pd.DataFrame, all_results: pd.DataFrame) -> str:
    lines = []
    lines.append("ORN Final Thesis Pipeline Summary")
    lines.append("=" * 80)
    lines.append(f"Total patients: {len(df)}")
    lines.append(f"ORN positive: {int((df[TARGET_COL] == 1).sum())}")
    lines.append(f"ORN negative: {int((df[TARGET_COL] == 0).sum())}")
    lines.append("")
    for group in ["clinical_model", "treatment_model", "full_model"]:
        sub = all_results[all_results["feature_group"] == group].sort_values("repeated_cv_test_roc_auc_mean", ascending=False)
        if len(sub) == 0:
            continue
        best = sub.iloc[0]
        lines.append(f"[{group}] best model: {best['model']}")
        lines.append(f"- repeated-CV ROC-AUC = {best['repeated_cv_test_roc_auc_mean']:.3f} ± {best['repeated_cv_test_roc_auc_std']:.3f}")
        lines.append(f"- repeated-CV Brier = {best['repeated_cv_test_brier_mean']:.3f} ± {best['repeated_cv_test_brier_std']:.3f}")
        lines.append(f"- repeated-CV ECE = {best['repeated_cv_test_ece_mean']:.3f} ± {best['repeated_cv_test_ece_std']:.3f}")
        lines.append(f"- hold-out ROC-AUC = {best['test_ROC_AUC']:.3f}")
        lines.append("")
    lines.append("Interpretation:")
    lines.append("- clinical_model supports baseline pre-treatment risk stratification.")
    lines.append("- treatment_model measures incremental predictive value of surgery / extraction / RT dose.")
    lines.append("- full_model represents the strongest retrospective stratification setting for thesis reporting.")
    return "\n".join(lines)

# ============================================================
# 6. 主程式
# ============================================================
def main() -> None:
    df = load_data()
    save_text(EDA_DIR / "dataset_summary.txt", make_dataset_summary(df))
    save_missingness(df)
    plot_correlation(df)
    uni_df = univariate_analysis(df)

    feature_groups = {
        "clinical_model": CLINICAL_FEATURES,
        "treatment_model": TREATMENT_ADDED_FEATURES,
        "full_model": CLINICAL_FEATURES + TREATMENT_ADDED_FEATURES,
    }

    all_results = []
    for group_name, feature_cols in feature_groups.items():
        res = run_model_group(df, group_name, feature_cols)
        all_results.append(res)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(REPORT_DIR / "Table2_model_comparison_all_groups.csv", index=False, encoding="utf-8-sig")

    physician_summary = make_progress_summary(df, uni_df, all_results_df)
    thesis_summary = make_thesis_summary(df, all_results_df)
    save_text(REPORT_DIR / "physician_progress_summary.txt", physician_summary)
    save_text(OUTPUT_DIR / "thesis_level_summary.txt", thesis_summary)

    readme = """
ORN Final Thesis Pipeline
=========================
This version creates three feature-group comparisons:
1. clinical_model
2. treatment_model
3. full_model

Report-ready outputs:
- Table1_baseline_characteristics.csv
- Table2_model_comparison_all_groups.csv
- physician_progress_summary.txt
- thesis_level_summary.txt

Recommended presentation flow for physician meeting:
1. Dataset overview
2. Significant variables from univariate screening
3. Clinical vs treatment vs full model comparison
4. Best model ROC / calibration / decision curve
5. Clinical interpretation and cautions
""".strip()
    save_text(OUTPUT_DIR / "README.txt", readme)

    print("Pipeline finished.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
