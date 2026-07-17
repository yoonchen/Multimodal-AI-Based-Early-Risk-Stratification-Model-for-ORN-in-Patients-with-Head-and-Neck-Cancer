from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")

TASK_TO_COLUMN = {"task1": "label_task1", "task2": "label_task2"}
TASK_TO_NAME = {
    "task1": "task1_visible_orn_vs_non_orn",
    "task2": "task2_orn_normal_vs_non_orn",
}
DROP_TABULAR_COLS = ["orn_diagnosis_date", "censor_date", "reference_date_for_model", "ORN_label"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patient-level multimodal ORN baseline pipeline")
    p.add_argument("--image_excel", type=str, default="image_data.xlsx")
    p.add_argument("--tabular_excel", type=str, default="data_v3.1.xlsx")
    p.add_argument("--tabular_sheet", type=str, default="model_full_pre_orn")
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--task", type=str, choices=["task1", "task2"], required=True)
    p.add_argument("--output_dir", type=str, default="orn_multimodal_outputs")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--use_pretrained", action="store_true")
    p.add_argument("--image_pooling", type=str, choices=["mean", "max", "meanmax"], default="meanmax")
    p.add_argument("--sensitivity_target", type=float, default=0.80)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_bootstrap: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        s_b = y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, s_b))
    if not aucs:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, target_sensitivity: float = 0.80) -> Tuple[float, pd.DataFrame]:
    thresholds = np.unique(np.round(y_score, 6))
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(specificity_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        })
    df = pd.DataFrame(rows).sort_values("threshold")
    feasible = df[df["sensitivity"] >= target_sensitivity].copy()
    if len(feasible) > 0:
        feasible = feasible.sort_values(["specificity", "f1", "threshold"], ascending=[False, False, True])
        best = feasible.iloc[0]
    else:
        best = df.sort_values(["f1", "specificity", "threshold"], ascending=[False, False, True]).iloc[0]
    return float(best["threshold"]), df


class ImageFeatureExtractor:
    def __init__(self, device: torch.device, img_size: int = 224, batch_size: int = 16, use_pretrained: bool = False):
        self.device = device
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = self._build_model(use_pretrained).to(device)
        self.model.eval()

    def _build_model(self, use_pretrained: bool) -> nn.Module:
        weights = None
        if use_pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(weights=None)
        model = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        for p in model.parameters():
            p.requires_grad = False
        return model

    def extract_from_paths(self, paths: Sequence[Path]) -> np.ndarray:
        out = []
        batch = []
        for path in paths:
            img = Image.open(path).convert("L")
            batch.append(self.transform(img))
            if len(batch) == self.batch_size:
                out.append(self._forward(torch.stack(batch, dim=0)))
                batch = []
        if batch:
            out.append(self._forward(torch.stack(batch, dim=0)))
        return np.vstack(out) if out else np.empty((0, 512), dtype=np.float32)

    def _forward(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.model(x.to(self.device)).cpu().numpy()


def resolve_image_path(root: Path, row: pd.Series) -> Optional[Path]:
    base_names = [str(row.get("image_name_std", "")).strip(), str(row.get("image_name", "")).strip(), str(row.get("image_id", "")).strip()]
    base_names = [b for b in base_names if b and b.lower() != "nan"]
    exts = ["", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
    subdirs = ["", "image", "images", "non_orn", "orn", "visible_orn", "orn_normal"]
    seen = set()
    candidates = []
    for base in base_names:
        for sd in subdirs:
            for ext in exts:
                p = root / sd / f"{base}{ext}" if sd else root / f"{base}{ext}"
                s = str(p)
                if s not in seen:
                    seen.add(s)
                    candidates.append(p)
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def load_and_merge(image_excel: Path, tabular_excel: Path, tabular_sheet: str, image_root: Path, task: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    image_df = pd.read_excel(image_excel)
    task_col = TASK_TO_COLUMN[task]
    image_df = image_df[image_df[task_col] != "exclude"].copy()
    image_df["label"] = image_df[task_col].map({"negative": 0, "positive": 1})
    image_df = image_df.dropna(subset=["label"]).copy()
    image_df["label"] = image_df["label"].astype(int)
    image_df["ipatient"] = image_df["ipatient"].astype(str)
    image_df["image_path"] = image_df.apply(lambda r: resolve_image_path(image_root, r), axis=1)
    image_df = image_df.dropna(subset=["image_path"]).copy()

    patient_labels = image_df.groupby("ipatient")["label"].max().reset_index()

    tabular_df = pd.read_excel(tabular_excel, sheet_name=tabular_sheet)
    tabular_df["ipatient"] = tabular_df["ipatient"].astype(str)
    merged_tab = patient_labels.merge(tabular_df, on="ipatient", how="inner")
    usable_patients = set(merged_tab["ipatient"].tolist())
    image_df = image_df[image_df["ipatient"].isin(usable_patients)].copy()
    return image_df, merged_tab


def infer_tabular_columns(tabular_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in tabular_df.columns if c not in ["ipatient", "label"] + DROP_TABULAR_COLS]
    numeric_cols, categorical_cols = [], []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(tabular_df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def build_tabular_preprocessor(tabular_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_tabular_columns(tabular_df)
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])


def aggregate_image_features(image_feats_df: pd.DataFrame, pooling: str = "meanmax") -> pd.DataFrame:
    feat_cols = [c for c in image_feats_df.columns if c.startswith("img_feat_")]
    rows = []
    for pid, sub in image_feats_df.groupby("ipatient"):
        arr = sub[feat_cols].to_numpy(dtype=np.float32)
        row = {"ipatient": pid}
        if pooling in ("mean", "meanmax"):
            avg = arr.mean(axis=0)
            for i, v in enumerate(avg):
                row[f"img_mean_{i}"] = float(v)
        if pooling in ("max", "meanmax"):
            mx = arr.max(axis=0)
            for i, v in enumerate(mx):
                row[f"img_max_{i}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def build_models(random_state: int) -> Dict[str, object]:
    return {
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_split=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def fit_predict_stage(stage_name: str, base_model, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, target_sensitivity: float) -> Tuple[np.ndarray, float, pd.DataFrame]:
    feature_cols = [c for c in train_df.columns if c not in ["ipatient", target_col] + DROP_TABULAR_COLS]

    if stage_name in ["clinical", "multimodal"]:
        pipe = Pipeline([
            ("pre", build_tabular_preprocessor(train_df)),
            ("clf", base_model),
        ])
        pipe.fit(train_df[feature_cols], train_df[target_col].to_numpy().ravel())
        val_score = pipe.predict_proba(val_df[feature_cols])[:, 1]
        threshold, sweep = choose_threshold(val_df[target_col].to_numpy().ravel(), val_score, target_sensitivity)

        trainval = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        pipe2 = Pipeline([
            ("pre", build_tabular_preprocessor(trainval)),
            ("clf", base_model.__class__(**base_model.get_params())),
        ])
        pipe2.fit(trainval[feature_cols], trainval[target_col].to_numpy().ravel())
        test_score = pipe2.predict_proba(test_df[feature_cols])[:, 1]
        return test_score, threshold, sweep

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_col].to_numpy().ravel()
    y_val = val_df[target_col].to_numpy().ravel()
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = base_model.__class__(**base_model.get_params())
    model.fit(X_train_s, y_train)
    val_score = model.predict_proba(X_val_s)[:, 1]
    threshold, sweep = choose_threshold(y_val, val_score, target_sensitivity)

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    scaler2 = StandardScaler()
    X_trainval_s = scaler2.fit_transform(X_trainval)
    X_test_s = scaler2.transform(X_test)

    model2 = base_model.__class__(**base_model.get_params())
    model2.fit(X_trainval_s, y_trainval)
    test_score = model2.predict_proba(X_test_s)[:, 1]
    return test_score, threshold, sweep


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity_score(y_true, y_pred)),
        "threshold": float(threshold),
    }


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, save_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f} (95% CI {ci_low:.3f}-{ci_high:.3f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def summarize_stage(stage_df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for k in ["auroc", "f1", "sensitivity", "specificity"]:
        out[f"{k}_mean"] = float(stage_df[k].mean())
        out[f"{k}_std"] = float(stage_df[k].std(ddof=1))
    return out


def run(cfg: argparse.Namespace) -> None:
    set_seed(cfg.random_state)
    out_dir = Path(cfg.output_dir)
    figs_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    ensure_dir(out_dir)
    ensure_dir(figs_dir)
    ensure_dir(tables_dir)

    device = torch.device("cpu" if cfg.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[Info] device={device}")

    image_df, patient_tab = load_and_merge(
        Path(cfg.image_excel),
        Path(cfg.tabular_excel),
        cfg.tabular_sheet,
        Path(cfg.image_root),
        cfg.task,
    )
    print(f"[Info] usable images={len(image_df)} usable patients={patient_tab['ipatient'].nunique()}")

    extractor = ImageFeatureExtractor(device=device, img_size=cfg.img_size, batch_size=cfg.batch_size, use_pretrained=cfg.use_pretrained)
    feats = extractor.extract_from_paths(image_df["image_path"].tolist())
    image_feats_df = image_df[["ipatient", "label"]].copy()
    for i in range(feats.shape[1]):
        image_feats_df[f"img_feat_{i}"] = feats[:, i]
    patient_img = aggregate_image_features(image_feats_df, pooling=cfg.image_pooling)

    patient_df = patient_tab.merge(patient_img, on="ipatient", how="inner").copy()
    patient_df["label"] = patient_df["label"].astype(int)

    image_cols = ["ipatient", "label"] + [
    c for c in patient_df.columns
    if c.startswith("img_mean_") or c.startswith("img_max_")
    ]

    clinical_cols = ["ipatient", "label"] + [
        c for c in patient_tab.columns
        if c not in ["ipatient", "label"]
    ]

    multimodal_cols = ["ipatient", "label"] + [
        c for c in patient_df.columns
        if c not in ["ipatient", "label"]
    ]

    y_pat = patient_df["label"].values
    binc = np.bincount(y_pat)
    if len(binc) < 2 or min(binc) < cfg.n_splits:
        raise ValueError(f"Positive/negative patient count is smaller than n_splits={cfg.n_splits}. Please reduce --n_splits.")
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    models_dict = build_models(cfg.random_state)

    fold_rows = []
    sweep_rows = []
    oof = {stage: {"y_true": np.full(len(patient_df), np.nan), "y_score": np.full(len(patient_df), np.nan)} for stage in ["clinical", "image", "multimodal"]}

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_pat)), y_pat), start=1):
        trainval_df = patient_df.iloc[trainval_idx].reset_index(drop=True)
        test_df = patient_df.iloc[test_idx].reset_index(drop=True)

        inner_train_idx, inner_val_idx = train_test_split(
            np.arange(len(trainval_df)),
            test_size=0.20,
            random_state=cfg.random_state + fold,
            stratify=trainval_df["label"].values,
        )
        train_df = trainval_df.iloc[inner_train_idx].reset_index(drop=True)
        val_df = trainval_df.iloc[inner_val_idx].reset_index(drop=True)

        stage_map = {
            "clinical": (train_df[clinical_cols], val_df[clinical_cols], test_df[clinical_cols]),
            "image": (train_df[image_cols], val_df[image_cols], test_df[image_cols]),
            "multimodal": (train_df[multimodal_cols], val_df[multimodal_cols], test_df[multimodal_cols]),
        }

        for stage in ["clinical", "image", "multimodal"]:
            tr_s, va_s, te_s = stage_map[stage]
            best_auc = -np.inf
            best_model_name = None
            best_score = None
            best_threshold = None
            best_sweep = None

            for model_name, model in models_dict.items():
                test_score, threshold, sweep = fit_predict_stage(stage, model, tr_s.copy(), va_s.copy(), te_s.copy(), "label", cfg.sensitivity_target)
                auc = roc_auc_score(te_s["label"].values, test_score)
                if auc > best_auc:
                    best_auc = auc
                    best_model_name = model_name
                    best_score = test_score
                    best_threshold = threshold
                    best_sweep = sweep.copy()

            metrics = compute_metrics(te_s["label"].values, best_score, best_threshold)
            fold_rows.append({"fold": fold, "stage": stage, "model": best_model_name, **metrics})
            global_idx = patient_df.index[test_idx]
            oof[stage]["y_true"][global_idx] = te_s["label"].values
            oof[stage]["y_score"][global_idx] = best_score

            best_sweep["fold"] = fold
            best_sweep["stage"] = stage
            best_sweep["model"] = best_model_name
            sweep_rows.append(best_sweep)
            print(f"[Fold {fold}] {stage:<10} best={best_model_name:<6} auc={metrics['auroc']:.3f} f1={metrics['f1']:.3f}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(tables_dir / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(sweep_rows, axis=0).to_csv(tables_dir / "threshold_sweeps.csv", index=False, encoding="utf-8-sig")

    summary = {
        "task": TASK_TO_NAME[cfg.task],
        "n_images": int(len(image_df)),
        "n_patients": int(patient_df["ipatient"].nunique()),
        "patient_class_counts": {str(k): int(v) for k, v in patient_df["label"].value_counts().sort_index().to_dict().items()},
        "cv": {},
        "oof": {},
    }

    lines = []
    lines.append(f"Task: {TASK_TO_NAME[cfg.task]}")
    lines.append(f"CV: patient-level {cfg.n_splits}-fold")
    lines.append(f"Total usable images: {len(image_df)}")
    lines.append(f"Total patients: {patient_df['ipatient'].nunique()}")
    lines.append(f"Patient class counts: {patient_df['label'].value_counts().sort_index().to_dict()}")
    lines.append("")

    for stage in ["clinical", "image", "multimodal"]:
        stage_df = fold_df[fold_df["stage"] == stage].copy()
        stats = summarize_stage(stage_df)
        summary["cv"][stage] = stats
        lines.append(f"{stage.capitalize()} CV (fold mean ± std):")
        lines.append(f"AUROC = {stats['auroc_mean']:.3f} ± {stats['auroc_std']:.3f}")
        lines.append(f"F1 = {stats['f1_mean']:.3f} ± {stats['f1_std']:.3f}")
        lines.append(f"Sensitivity = {stats['sensitivity_mean']:.3f} ± {stats['sensitivity_std']:.3f}")
        lines.append(f"Specificity = {stats['specificity_mean']:.3f} ± {stats['specificity_std']:.3f}")
        lines.append("")

        y_true = oof[stage]["y_true"]
        y_score = oof[stage]["y_score"]
        mask = ~np.isnan(y_true) & ~np.isnan(y_score)
        y_true = y_true[mask].astype(int)
        y_score = y_score[mask]
        auc = roc_auc_score(y_true, y_score)
        ci_low, ci_high = bootstrap_auc_ci(y_true, y_score, seed=cfg.random_state)
        summary["oof"][stage] = {"auroc": float(auc), "auroc_ci_low": ci_low, "auroc_ci_high": ci_high}
        lines.append(f"{stage.capitalize()} OOF AUROC = {auc:.3f} (95% CI {ci_low:.3f}-{ci_high:.3f})")
        lines.append("")
        plot_roc(y_true, y_score, figs_dir / f"roc_{stage}_oof.png", f"{stage.capitalize()} OOF ROC")

    best_stage = max(summary["oof"].keys(), key=lambda s: summary["oof"][s]["auroc"])
    summary["best_stage"] = best_stage
    lines.append("Conclusion:")
    lines.append(f"Best stage by OOF AUROC: {best_stage} (AUC={summary['oof'][best_stage]['auroc']:.3f})")

    (out_dir / "cv_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "cv_metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Final summary ===")
    print("\n".join(lines))


if __name__ == "__main__":
    run(parse_args())
