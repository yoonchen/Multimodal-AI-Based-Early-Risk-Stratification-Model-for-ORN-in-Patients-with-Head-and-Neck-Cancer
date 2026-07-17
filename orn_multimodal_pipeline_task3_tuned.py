from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")

TASK3_CLASS_NAMES = ["non_orn", "orn_normal", "visible_orn"]
TASK_TO_COLUMN = {
    "task1": "label_task1",
    "task2": "label_task2",
    "task3": "label_task3",
}
TASK_TO_NAME = {
    "task1": "task1_visible_orn_vs_non_orn",
    "task2": "task2_orn_normal_vs_non_orn",
    "task3": "task3_multiclass_exploration",
}
DROP_TABULAR_COLS = ["orn_diagnosis_date", "censor_date", "reference_date_for_model", "ORN_label"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Task3 multimodal ORN pipeline with safer multimodal tuning. "
            "Default keeps feature-level fusion but reduces image features by train-fold PCA."
        )
    )
    p.add_argument("--image_excel", type=str, default="image_data.xlsx")
    p.add_argument("--image_sheet", type=str, default="image_master")
    p.add_argument("--tabular_excel", type=str, default="data_v3.1.xlsx")
    p.add_argument("--tabular_sheet", type=str, default="model_full_pre_orn")
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--task", type=str, choices=["task3"], default="task3")
    p.add_argument("--output_dir", type=str, default="orn_task3_multimodal_tuned_outputs")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--use_pretrained", action="store_true")
    p.add_argument("--image_pooling", type=str, choices=["mean", "max", "meanmax"], default="meanmax")
    p.add_argument(
        "--task3_patient_label_strategy",
        type=str,
        choices=["max", "mode"],
        default="max",
        help=(
            "How to collapse multiple image-level task3 labels to one patient-level label. "
            "max uses severity order non_orn < orn_normal < visible_orn and is recommended for the thesis table; "
            "mode reproduces the previous majority-label behavior."
        ),
    )
    p.add_argument(
        "--image_pca_components",
        type=int,
        default=12,
        help="PCA components for image feature block in image-only and feature-level multimodal models. Use 0 to disable PCA.",
    )
    p.add_argument(
        "--multimodal_mode",
        type=str,
        choices=["pca_concat", "weighted_prob"],
        default="pca_concat",
        help=(
            "pca_concat: feature-level concatenation with PCA-reduced image block. "
            "weighted_prob: validation-tuned decision-level fusion of structured and image probabilities."
        ),
    )
    p.add_argument(
        "--fusion_alphas",
        type=str,
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0",
        help="For weighted_prob mode: alpha for structured probability in alpha*structured+(1-alpha)*image.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "rf"],
        choices=["logreg", "rf"],
        help="Candidate classifiers. Model selection is done on the inner validation split, not the test fold.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_auc_multiclass(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    vals = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        vals.append(roc_auc_score(y_bin, y_prob[:, c]))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_auc_ci_multiclass(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        auc = safe_auc_multiclass(y_true[idx], y_prob[idx], n_classes)
        if not math.isnan(auc):
            aucs.append(auc)
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute_multiclass_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> Dict[str, float]:
    y_pred = np.argmax(y_prob, axis=1)
    return {
        "macro_ovr_auroc": safe_auc_multiclass(y_true, y_prob, n_classes=n_classes),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def resolve_multiclass_label(col: pd.Series) -> pd.Series:
    x = col.astype(str).str.strip().str.lower()
    return x.map({name: i for i, name in enumerate(TASK3_CLASS_NAMES)})


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
        out, batch = [], []
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
    seen, candidates = set(), []
    for base in base_names:
        for sd in subdirs:
            for ext in exts:
                p = root / sd / f"{base}{ext}" if sd else root / f"{base}{ext}"
                if str(p) not in seen:
                    seen.add(str(p))
                    candidates.append(p)
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # fallback: recursive search by exact stem
    for base in base_names:
        stem = Path(base).stem
        for ext in exts[1:]:
            matches = list(root.rglob(f"{stem}{ext}"))
            if matches:
                return matches[0]
    return None


def load_and_merge(
    image_excel: Path,
    image_sheet: str,
    tabular_excel: Path,
    tabular_sheet: str,
    image_root: Path,
    task3_patient_label_strategy: str = "max",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    image_df = pd.read_excel(image_excel, sheet_name=image_sheet)
    image_df.columns = [str(c).strip() for c in image_df.columns]
    image_df["ipatient"] = image_df["ipatient"].astype(str)
    image_df["label_task3"] = image_df["label_task3"].astype(str).str.strip().str.lower()
    image_df = image_df[image_df["label_task3"].isin(TASK3_CLASS_NAMES)].copy()
    image_df["label"] = resolve_multiclass_label(image_df["label_task3"]).astype(int)

    image_df["image_path"] = image_df.apply(lambda r: resolve_image_path(image_root, r), axis=1)
    missing = image_df[image_df["image_path"].isna()].copy()
    image_df = image_df.dropna(subset=["image_path"]).copy()

    # Collapse image-level task3 labels to a patient-level label.
    # Class order is severity-oriented: non_orn=0, orn_normal=1, visible_orn=2.
    # For thesis-level patient classification, max is usually safer than mode because
    # a patient with any visible ORN image should remain in the visible_orn group.
    if task3_patient_label_strategy == "max":
        patient_labels = image_df.groupby("ipatient")["label"].max().reset_index()
    elif task3_patient_label_strategy == "mode":
        patient_labels = image_df.groupby("ipatient")["label"].agg(lambda x: int(x.mode().iloc[0])).reset_index()
    else:
        raise ValueError(f"Unknown task3_patient_label_strategy: {task3_patient_label_strategy}")
    tabular_df = pd.read_excel(tabular_excel, sheet_name=tabular_sheet)
    tabular_df.columns = [str(c).strip() for c in tabular_df.columns]
    tabular_df["ipatient"] = tabular_df["ipatient"].astype(str)

    merged_tab = patient_labels.merge(tabular_df, on="ipatient", how="inner")
    usable_patients = set(merged_tab["ipatient"].tolist())
    image_df = image_df[image_df["ipatient"].isin(usable_patients)].copy()
    return image_df, merged_tab


def infer_tabular_columns(tabular_df: pd.DataFrame, candidate_cols: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    if candidate_cols is None:
        feature_cols = [c for c in tabular_df.columns if c not in ["ipatient", "label"] + DROP_TABULAR_COLS]
    else:
        feature_cols = [c for c in candidate_cols if c in tabular_df.columns and c not in ["ipatient", "label"] + DROP_TABULAR_COLS]
    numeric_cols, categorical_cols = [], []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(tabular_df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def build_tabular_preprocessor(train_df: pd.DataFrame, tabular_cols: List[str]) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_tabular_columns(train_df, tabular_cols)
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols))
    return ColumnTransformer(transformers)


def make_image_pipeline(base_model, n_train: int, n_features: int, image_pca_components: int) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if image_pca_components and image_pca_components > 0:
        n_comp = min(int(image_pca_components), max(1, n_train - 1), n_features)
        steps.append(("pca", PCA(n_components=n_comp, random_state=42)))
    steps.append(("clf", clone(base_model)))
    return Pipeline(steps)


def make_multimodal_pca_concat_pipeline(
    train_df: pd.DataFrame,
    tabular_cols: List[str],
    image_cols: List[str],
    base_model,
    image_pca_components: int,
) -> Pipeline:
    numeric_cols, categorical_cols = infer_tabular_columns(train_df, tabular_cols)
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols))
    if image_cols:
        img_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        if image_pca_components and image_pca_components > 0:
            n_comp = min(int(image_pca_components), max(1, len(train_df) - 1), len(image_cols))
            img_steps.append(("pca", PCA(n_components=n_comp, random_state=42)))
        transformers.append(("img", Pipeline(img_steps), image_cols))
    return Pipeline([
        ("pre", ColumnTransformer(transformers)),
        ("clf", clone(base_model)),
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


def build_models(random_state: int, selected: List[str]) -> Dict[str, object]:
    models_dict: Dict[str, object] = {}
    if "logreg" in selected:
        models_dict["logreg"] = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        )
    if "rf" in selected:
        models_dict["rf"] = RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_split=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        )
    return models_dict


def fit_select_predict_stage(
    stage: str,
    models_dict: Dict[str, object],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tabular_cols: List[str],
    image_cols: List[str],
    image_pca_components: int,
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    y_train = train_df["label"].to_numpy().ravel()
    y_val = val_df["label"].to_numpy().ravel()
    y_trainval = pd.concat([train_df, val_df], axis=0)["label"].to_numpy().ravel()

    best = {
        "model_name": None,
        "val_auc": -np.inf,
        "val_prob": None,
        "test_prob": None,
    }

    for model_name, base_model in models_dict.items():
        if stage == "structured":
            pipe = Pipeline([
                ("pre", build_tabular_preprocessor(train_df, tabular_cols)),
                ("clf", clone(base_model)),
            ])
            pipe.fit(train_df[tabular_cols], y_train)
            val_prob = pipe.predict_proba(val_df[tabular_cols])

            trainval = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
            pipe2 = Pipeline([
                ("pre", build_tabular_preprocessor(trainval, tabular_cols)),
                ("clf", clone(base_model)),
            ])
            pipe2.fit(trainval[tabular_cols], y_trainval)
            test_prob = pipe2.predict_proba(test_df[tabular_cols])

        elif stage == "image":
            pipe = make_image_pipeline(base_model, len(train_df), len(image_cols), image_pca_components)
            pipe.fit(train_df[image_cols], y_train)
            val_prob = pipe.predict_proba(val_df[image_cols])

            trainval = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
            pipe2 = make_image_pipeline(base_model, len(trainval), len(image_cols), image_pca_components)
            pipe2.fit(trainval[image_cols], y_trainval)
            test_prob = pipe2.predict_proba(test_df[image_cols])

        elif stage == "multimodal":
            pipe = make_multimodal_pca_concat_pipeline(train_df, tabular_cols, image_cols, base_model, image_pca_components)
            pipe.fit(train_df[tabular_cols + image_cols], y_train)
            val_prob = pipe.predict_proba(val_df[tabular_cols + image_cols])

            trainval = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
            pipe2 = make_multimodal_pca_concat_pipeline(trainval, tabular_cols, image_cols, base_model, image_pca_components)
            pipe2.fit(trainval[tabular_cols + image_cols], y_trainval)
            test_prob = pipe2.predict_proba(test_df[tabular_cols + image_cols])

        else:
            raise ValueError(stage)

        val_auc = safe_auc_multiclass(y_val, val_prob, n_classes=len(TASK3_CLASS_NAMES))
        if val_auc > best["val_auc"]:
            best = {
                "model_name": model_name,
                "val_auc": val_auc,
                "val_prob": val_prob,
                "test_prob": test_prob,
            }

    return best["val_prob"], best["test_prob"], str(best["model_name"]), float(best["val_auc"])


def choose_weighted_fusion_alpha(
    y_val: np.ndarray,
    p_struct_val: np.ndarray,
    p_img_val: np.ndarray,
    alphas: List[float],
) -> Tuple[float, float]:
    rows = []
    for a in alphas:
        p = a * p_struct_val + (1.0 - a) * p_img_val
        auc = safe_auc_multiclass(y_val, p, n_classes=len(TASK3_CLASS_NAMES))
        f1 = f1_score(y_val, np.argmax(p, axis=1), average="macro", zero_division=0)
        rows.append((float(a), float(auc), float(f1)))
    # Primary: validation macro OVR AUC. Tie-break: macro F1, then smaller image over-weighting risk by favoring structured.
    best = max(rows, key=lambda x: (x[1], x[2], x[0]))
    return best[0], best[1]


def plot_confusion_matrix_any(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(5.6, 4.8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=20)
    plt.yticks(ticks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def summarize_stage(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for k in ["macro_ovr_auroc", "macro_f1", "accuracy"]:
        vals = pd.to_numeric(df[k], errors="coerce").dropna()
        out[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
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
    print(f"[Info] multimodal_mode={cfg.multimodal_mode}, image_pca_components={cfg.image_pca_components}")
    print(f"[Info] task3_patient_label_strategy={cfg.task3_patient_label_strategy}")

    image_df, patient_tab = load_and_merge(
        Path(cfg.image_excel),
        cfg.image_sheet,
        Path(cfg.tabular_excel),
        cfg.tabular_sheet,
        Path(cfg.image_root),
        task3_patient_label_strategy=cfg.task3_patient_label_strategy,
    )
    if len(image_df) == 0:
        raise ValueError("No usable images found. Check --image_root and image table columns.")
    print(f"[Info] usable images={len(image_df)} usable patients={patient_tab['ipatient'].nunique()}")

    extractor = ImageFeatureExtractor(
        device=device,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        use_pretrained=cfg.use_pretrained,
    )
    feats = extractor.extract_from_paths([Path(p) for p in image_df["image_path"].tolist()])
    image_feats_df = image_df[["ipatient", "label"]].copy()
    for i in range(feats.shape[1]):
        image_feats_df[f"img_feat_{i}"] = feats[:, i]

    patient_img = aggregate_image_features(image_feats_df, pooling=cfg.image_pooling)
    patient_df = patient_tab.merge(patient_img, on="ipatient", how="inner").copy()
    patient_df["label"] = patient_df["label"].astype(int)

    image_cols = [c for c in patient_df.columns if c.startswith("img_mean_") or c.startswith("img_max_")]
    tabular_cols = [c for c in patient_tab.columns if c not in ["ipatient", "label"] + DROP_TABULAR_COLS]
    all_cols = ["ipatient", "label"] + tabular_cols + image_cols
    patient_df = patient_df[all_cols].copy()

    y_pat = patient_df["label"].to_numpy()
    binc = np.bincount(y_pat, minlength=len(TASK3_CLASS_NAMES))
    if min(binc) < cfg.n_splits:
        raise ValueError(f"At least one class count is smaller than n_splits={cfg.n_splits}. Counts={binc.tolist()}")

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    models_dict = build_models(cfg.random_state, cfg.models)
    fusion_alphas = [float(x.strip()) for x in cfg.fusion_alphas.split(",") if x.strip() != ""]

    fold_rows = []
    pred_rows = []
    fusion_rows = []
    oof_prob = {stage: np.full((len(patient_df), len(TASK3_CLASS_NAMES)), np.nan) for stage in ["structured", "image", "multimodal"]}
    oof_true = np.full(len(patient_df), np.nan)

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
        y_val = val_df["label"].to_numpy().ravel()
        y_test = test_df["label"].to_numpy().ravel()

        val_struct, test_struct, model_struct, val_auc_struct = fit_select_predict_stage(
            "structured", models_dict, train_df, val_df, test_df, tabular_cols, image_cols, cfg.image_pca_components
        )
        val_img, test_img, model_img, val_auc_img = fit_select_predict_stage(
            "image", models_dict, train_df, val_df, test_df, tabular_cols, image_cols, cfg.image_pca_components
        )

        if cfg.multimodal_mode == "pca_concat":
            val_mm, test_mm, model_mm, val_auc_mm = fit_select_predict_stage(
                "multimodal", models_dict, train_df, val_df, test_df, tabular_cols, image_cols, cfg.image_pca_components
            )
            mm_model_label = f"{model_mm}_pca_concat"
            selected_alpha = np.nan
        else:
            selected_alpha, val_auc_mm = choose_weighted_fusion_alpha(y_val, val_struct, val_img, fusion_alphas)
            test_mm = selected_alpha * test_struct + (1.0 - selected_alpha) * test_img
            val_mm = selected_alpha * val_struct + (1.0 - selected_alpha) * val_img
            mm_model_label = f"weighted_prob_alpha={selected_alpha:.2f}"
            fusion_rows.append({
                "fold": fold,
                "selected_alpha_structured": selected_alpha,
                "selected_image_weight": 1.0 - selected_alpha,
                "val_macro_ovr_auc_fusion": val_auc_mm,
                "structured_model": model_struct,
                "image_model": model_img,
            })

        stage_outputs = {
            "structured": (test_struct, model_struct, val_auc_struct),
            "image": (test_img, model_img, val_auc_img),
            "multimodal": (test_mm, mm_model_label, val_auc_mm),
        }

        global_idx = patient_df.index[test_idx]
        oof_true[global_idx] = y_test

        for stage, (prob, model_name, val_auc) in stage_outputs.items():
            metrics = compute_multiclass_metrics(y_test, prob, n_classes=len(TASK3_CLASS_NAMES))
            fold_rows.append({
                "fold": fold,
                "stage": stage,
                "model": model_name,
                "inner_val_macro_ovr_auc": val_auc,
                "fusion_alpha_structured": selected_alpha if stage == "multimodal" and cfg.multimodal_mode == "weighted_prob" else np.nan,
                **metrics,
            })
            oof_prob[stage][global_idx, :] = prob
            y_pred = np.argmax(prob, axis=1)
            for i, pid in enumerate(test_df["ipatient"].values):
                row = {
                    "fold": fold,
                    "stage": stage,
                    "ipatient": pid,
                    "y_true": int(y_test[i]),
                    "y_pred": int(y_pred[i]),
                    "model": model_name,
                }
                for c, name in enumerate(TASK3_CLASS_NAMES):
                    row[f"prob_{name}"] = float(prob[i, c])
                pred_rows.append(row)

            plot_confusion_matrix_any(
                y_test,
                y_pred,
                TASK3_CLASS_NAMES,
                figs_dir / f"fold_{fold}_{stage}_confusion_matrix.png",
                f"fold {fold} {stage} confusion matrix",
            )
            print(f"[Fold {fold}] {stage:<11} model={model_name:<24} val_auc={val_auc:.3f} test_auc={metrics['macro_ovr_auroc']:.3f} macro_f1={metrics['macro_f1']:.3f}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(tables_dir / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(tables_dir / "oof_predictions_long.csv", index=False, encoding="utf-8-sig")
    if fusion_rows:
        pd.DataFrame(fusion_rows).to_csv(tables_dir / "fusion_alpha_by_fold.csv", index=False, encoding="utf-8-sig")

    summary = {
        "task": TASK_TO_NAME[cfg.task],
        "multimodal_mode": cfg.multimodal_mode,
        "image_pca_components": cfg.image_pca_components,
        "n_images": int(len(image_df)),
        "n_patients": int(patient_df["ipatient"].nunique()),
        "patient_class_counts": {TASK3_CLASS_NAMES[i]: int(v) for i, v in enumerate(np.bincount(y_pat, minlength=len(TASK3_CLASS_NAMES)))},
        "cv": {},
        "oof": {},
    }

    lines = [
        f"Task: {TASK_TO_NAME[cfg.task]}",
        f"CV: patient-level {cfg.n_splits}-fold",
        f"Multimodal mode: {cfg.multimodal_mode}",
        f"Image PCA components: {cfg.image_pca_components}",
        f"Total usable images: {len(image_df)}",
        f"Total patients: {patient_df['ipatient'].nunique()}",
        f"Patient class counts: {summary['patient_class_counts']}",
        "",
    ]

    for stage in ["structured", "image", "multimodal"]:
        stage_df = fold_df[fold_df["stage"] == stage].copy()
        stats = summarize_stage(stage_df)
        summary["cv"][stage] = stats
        lines.append(f"{stage.capitalize()} CV (fold mean ± std):")
        lines.append(f"Macro OVR AUROC = {stats['macro_ovr_auroc_mean']:.3f} ± {stats['macro_ovr_auroc_std']:.3f}")
        lines.append(f"Macro F1 = {stats['macro_f1_mean']:.3f} ± {stats['macro_f1_std']:.3f}")
        lines.append(f"Accuracy = {stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}")
        lines.append("")

        mask = ~np.isnan(oof_true)
        y_true = oof_true[mask].astype(int)
        y_prob = oof_prob[stage][mask]
        auc = safe_auc_multiclass(y_true, y_prob, n_classes=len(TASK3_CLASS_NAMES))
        ci_low, ci_high = bootstrap_auc_ci_multiclass(y_true, y_prob, n_classes=len(TASK3_CLASS_NAMES), seed=cfg.random_state)
        y_pred = np.argmax(y_prob, axis=1)
        oof_metrics = {
            "macro_ovr_auroc": float(auc),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
        summary["oof"][stage] = oof_metrics
        lines.append(f"{stage.capitalize()} OOF Macro OVR AUROC = {auc:.3f} (95% CI {ci_low:.3f}-{ci_high:.3f})")
        lines.append(f"{stage.capitalize()} OOF Macro F1 = {oof_metrics['macro_f1']:.3f}")
        lines.append(f"{stage.capitalize()} OOF Accuracy = {oof_metrics['accuracy']:.3f}")
        lines.append("")

        plot_confusion_matrix_any(
            y_true,
            y_pred,
            TASK3_CLASS_NAMES,
            figs_dir / f"{stage}_oof_confusion_matrix.png",
            f"{stage.capitalize()} OOF confusion matrix",
        )

    best_stage = max(summary["oof"].keys(), key=lambda s: summary["oof"][s]["macro_ovr_auroc"])
    summary["best_stage_by_oof_macro_ovr_auc"] = best_stage
    lines.append("Conclusion:")
    lines.append(
        f"Best stage by OOF Macro OVR AUROC: {best_stage} "
        f"(AUC={summary['oof'][best_stage]['macro_ovr_auroc']:.3f})"
    )

    if cfg.multimodal_mode == "weighted_prob" and fusion_rows:
        alphas = pd.DataFrame(fusion_rows)["selected_alpha_structured"].astype(float)
        lines.append("")
        lines.append("Weighted-fusion note:")
        lines.append(
            f"Mean selected structured weight alpha = {alphas.mean():.3f} ± {alphas.std(ddof=1):.3f}; "
            f"image weight = {1 - alphas.mean():.3f}."
        )
        lines.append(
            "If alpha is close to 1.0, the validation data indicate that image features add limited stable signal for Task3."
        )

    (out_dir / "cv_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "cv_metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Final summary ===")
    print("\n".join(lines))


if __name__ == "__main__":
    run(parse_args())
