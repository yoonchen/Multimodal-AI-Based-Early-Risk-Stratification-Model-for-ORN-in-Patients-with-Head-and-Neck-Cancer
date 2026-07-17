from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.manifold import TSNE

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
SUPPORTED_MODELS = ["resnet18", "densenet121", "efficientnet_b0", "mobilenet_v3_small"]
SUPPORTED_AUGS = ["none", "light", "strong"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_auc_binary(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def specificity_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


def bootstrap_auc_ci(y_true: Sequence[int], y_prob: Sequence[float], n_bootstrap: int = 2000, seed: int = 42) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_bootstrap_valid": 0}

    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        scores.append(roc_auc_score(y_b, p_b))

    if len(scores) == 0:
        return {"auroc": safe_auc_binary(y_true, y_prob), "ci_low": float("nan"), "ci_high": float("nan"), "n_bootstrap_valid": 0}

    scores = np.asarray(scores, dtype=float)
    return {
        "auroc": safe_auc_binary(y_true, y_prob),
        "ci_low": float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5)),
        "n_bootstrap_valid": int(len(scores)),
    }


def binary_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5, ci_seed: int = 42) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    auc_ci = bootstrap_auc_ci(y_true, y_prob, seed=ci_seed)
    return {
        "threshold": float(threshold),
        "auroc": auc_ci["auroc"],
        "auroc_ci_low": auc_ci["ci_low"],
        "auroc_ci_high": auc_ci["ci_high"],
        "auroc_ci_95": [auc_ci["ci_low"], auc_ci["ci_high"]],
        "auroc_bootstrap_valid": auc_ci["n_bootstrap_valid"],
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity_score(y_true, y_pred),
    }


def find_binary_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    min_sensitivity: float = 0.80,
    return_table: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    unique_thresholds = np.unique(np.round(y_prob, 6))
    candidates = np.unique(np.concatenate(([0.0], unique_thresholds, [1.0])))
    candidates = np.sort(candidates)

    rows = []
    for thr in candidates:
        m = binary_metrics(y_true, y_prob, threshold=float(thr))
        rows.append({k: v for k, v in m.items() if k != "auroc_ci_95"})

    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    valid = [r for r in rows if not math.isnan(r["sensitivity"]) and r["sensitivity"] >= min_sensitivity]
    if valid:
        best = max(valid, key=lambda r: (r["specificity"], r["f1"], -r["threshold"]))
        best["selection_rule"] = f"max_specificity_given_sensitivity>={min_sensitivity:.2f}"
        if return_table:
            return best, sweep_df
        return best

    best = max(rows, key=lambda r: (r["f1"], r["specificity"], -r["threshold"]))
    best["selection_rule"] = "fallback_max_f1"
    if return_table:
        return best, sweep_df
    return best


def multiclass_macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@dataclass
class TaskConfig:
    task_name: str
    label_col: str
    is_binary: bool
    class_names: List[str]
    positive_name: Optional[str] = None


def get_task_config(task: str) -> TaskConfig:
    task = task.lower().strip()
    if task == "task1":
        return TaskConfig(
            task_name="task1_visible_orn_vs_non_orn",
            label_col="label_task1",
            is_binary=True,
            class_names=["negative", "positive"],
            positive_name="positive",
        )
    if task == "task2":
        return TaskConfig(
            task_name="task2_orn_normal_vs_non_orn",
            label_col="label_task2",
            is_binary=True,
            class_names=["negative", "positive"],
            positive_name="positive",
        )
    if task == "task3":
        return TaskConfig(
            task_name="task3_multiclass_exploration",
            label_col="label_task3",
            is_binary=False,
            class_names=["non_orn", "orn_normal", "visible_orn"],
        )
    raise ValueError(f"Unknown task: {task}")


def load_metadata(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def filter_by_task(df: pd.DataFrame, cfg: TaskConfig) -> pd.DataFrame:
    out = df.copy()
    out[cfg.label_col] = out[cfg.label_col].astype(str).str.strip().str.lower()

    if cfg.is_binary:
        out = out[out[cfg.label_col] != "exclude"].copy()
        valid = {"positive", "negative"}
        out = out[out[cfg.label_col].isin(valid)].copy()
        out["target"] = out[cfg.label_col].map({"negative": 0, "positive": 1})
    else:
        valid = set(cfg.class_names)
        out = out[out[cfg.label_col].isin(valid)].copy()
        cls2idx = {c: i for i, c in enumerate(cfg.class_names)}
        out["target"] = out[cfg.label_col].map(cls2idx)

    out["ipatient"] = out["ipatient"].astype(str)
    return out.reset_index(drop=True)


def candidate_stems_from_row(row: pd.Series) -> List[str]:
    candidates = []
    for col in ["image_name_std", "image_name", "image_id"]:
        if col in row and pd.notna(row[col]):
            candidates.append(str(row[col]).strip())
    seen = set()
    unique = []
    for x in candidates:
        if x and x not in seen:
            seen.add(x)
            unique.append(x)
    return unique


def resolve_image_path(row: pd.Series, image_root: Path) -> Optional[Path]:
    stems = candidate_stems_from_row(row)
    for stem in stems:
        for ext in IMG_EXTS:
            p = image_root / f"{stem}{ext}"
            if p.exists():
                return p

    subdirs = []
    if "label_raw" in row and pd.notna(row["label_raw"]):
        subdirs.append(str(row["label_raw"]).strip())
    subdirs += ["", "non_orn", "orn", "images", "visible_orn", "orn_normal"]

    tried = set()
    for sub in subdirs:
        for stem in stems:
            for ext in IMG_EXTS:
                p = (image_root / sub / f"{stem}{ext}").resolve()
                key = str(p)
                if key in tried:
                    continue
                tried.add(key)
                if p.exists():
                    return p

    for stem in stems:
        for ext in IMG_EXTS:
            matches = list(image_root.rglob(f"{stem}{ext}"))
            if matches:
                return matches[0]
    return None


def attach_image_paths(df: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    out = df.copy()
    out["image_path"] = out.apply(lambda r: resolve_image_path(r, image_root), axis=1)
    out["image_path"] = out["image_path"].apply(lambda x: str(x) if x is not None else None)
    return out


class PanoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("L")
        img = Image.merge("RGB", (img, img, img))
        img = self.transform(img)
        return {
            "image": img,
            "target": int(row["target"]),
            "ipatient": row["ipatient"],
            "image_name": str(row["image_name"]) if "image_name" in row else str(idx),
        }


def build_transforms(img_size: int = 224, aug_mode: str = "light") -> Tuple[transforms.Compose, transforms.Compose]:
    aug_mode = aug_mode.lower().strip()
    if aug_mode not in SUPPORTED_AUGS:
        raise ValueError(f"Unsupported aug_mode: {aug_mode}")

    train_ops: List[transforms.Compose | transforms.RandomApply | transforms.RandomRotation] = [
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
    ]

    if aug_mode == "light":
        train_ops += [
            transforms.RandomApply([transforms.ColorJitter(brightness=0.10, contrast=0.10)], p=0.6),
            transforms.RandomRotation(degrees=5, interpolation=InterpolationMode.BILINEAR),
        ]
    elif aug_mode == "strong":
        train_ops += [
            transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.8),
            transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.97, 1.03), interpolation=InterpolationMode.BILINEAR),
        ]

    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
    ]
    train_tf = transforms.Compose(train_ops)
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
    ])
    return train_tf, eval_tf


class ImageBackboneClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name.lower().strip()
        self.num_outputs = num_classes
        self.feature_dim: int

        if self.model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            self.feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        elif self.model_name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            backbone = models.densenet121(weights=weights)
            self.feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        elif self.model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            self.feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        elif self.model_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            self.feature_dim = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        return logits, feat


def make_dataloaders_from_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    img_size: int,
    batch_size: int,
    num_workers: int,
    aug_mode: str,
) -> Dict[str, DataLoader]:
    train_tf, eval_tf = build_transforms(img_size=img_size, aug_mode=aug_mode)
    data_map = {
        "train": (df_train, train_tf, True),
        "val": (df_val, eval_tf, False),
        "test": (df_test, eval_tf, False),
    }
    loaders = {}
    for split, (split_df, tf, shuffle) in data_map.items():
        ds = PanoDataset(split_df, transform=tf)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(torch.cuda.is_available()),
        )
    return loaders


def get_loss_fn(df_train: pd.DataFrame, is_binary: bool, num_classes: int, device: torch.device):
    y = df_train["target"].to_numpy().astype(int)
    counts = np.bincount(y, minlength=(2 if is_binary else num_classes)).astype(float)
    counts[counts == 0] = 1.0
    if is_binary:
        neg = counts[0]
        pos = counts[1]
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights_t)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device, is_binary: bool):
    model.eval()
    all_targets, all_probs, all_preds, all_feats = [], [], [], []
    all_patients, all_image_names = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].numpy().astype(int)
        logits, feats = model(x)
        if is_binary:
            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            prob_out = probs
        else:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            prob_out = probs
        all_targets.append(y)
        all_probs.append(prob_out)
        all_preds.append(preds)
        all_feats.append(feats.cpu().numpy())
        all_patients.extend(batch["ipatient"])
        all_image_names.extend(batch["image_name"])

    return {
        "y_true": np.concatenate(all_targets, axis=0),
        "y_prob": np.concatenate(all_probs, axis=0),
        "y_pred": np.concatenate(all_preds, axis=0),
        "features": np.concatenate(all_feats, axis=0),
        "ipatient": np.asarray(all_patients),
        "image_name": np.asarray(all_image_names),
    }


def evaluate_binary_outputs(outputs: Dict) -> Dict[str, float]:
    return binary_metrics(outputs["y_true"], outputs["y_prob"], threshold=0.5)


def evaluate_multiclass_outputs(outputs: Dict, class_names: List[str]) -> Dict[str, float]:
    y_true = outputs["y_true"]
    y_pred = outputs["y_pred"]
    y_prob = outputs["y_prob"]
    metrics = {"macro_f1": multiclass_macro_f1(y_true, y_pred), "accuracy": float((y_true == y_pred).mean())}
    aurocs = []
    for c in range(len(class_names)):
        y_true_bin = (y_true == c).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            continue
        aurocs.append(roc_auc_score(y_true_bin, y_prob[:, c]))
    metrics["macro_ovr_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    return metrics


def aggregate_patient_level_binary(outputs: Dict, threshold: float = 0.5, strategy: str = "max") -> pd.DataFrame:
    df = pd.DataFrame({
        "ipatient": outputs["ipatient"],
        "y_true": outputs["y_true"],
        "prob": outputs["y_prob"],
        "pred_image": outputs["y_pred"],
        "image_name": outputs["image_name"],
    })

    if strategy == "max":
        agg = df.groupby("ipatient", as_index=False).agg(
            y_true=("y_true", "max"),
            prob=("prob", "max"),
            prob_mean=("prob", "mean"),
            n_images=("image_name", "count"),
        )
    elif strategy == "mean":
        agg = df.groupby("ipatient", as_index=False).agg(
            y_true=("y_true", "max"),
            prob=("prob", "mean"),
            prob_max=("prob", "max"),
            n_images=("image_name", "count"),
        )
    else:
        raise ValueError(f"Unknown patient aggregation strategy: {strategy}")

    agg["pred_patient"] = (agg["prob"] >= threshold).astype(int)
    agg["aggregation_strategy"] = strategy
    agg["threshold"] = float(threshold)
    return agg


def aggregate_patient_level_multiclass(outputs: Dict, class_names: List[str]) -> pd.DataFrame:
    probs = outputs["y_prob"]
    prob_cols = {f"prob_{name}": probs[:, i] for i, name in enumerate(class_names)}
    df = pd.DataFrame({"ipatient": outputs["ipatient"], "y_true": outputs["y_true"], "image_name": outputs["image_name"], **prob_cols})
    agg_dict = {"y_true": "max", "image_name": "count"}
    for name in class_names:
        agg_dict[f"prob_{name}"] = "mean"
    agg = df.groupby("ipatient", as_index=False).agg(agg_dict).rename(columns={"image_name": "n_images"})
    agg["pred_patient"] = np.argmax(agg[[f"prob_{name}" for name in class_names]].to_numpy(), axis=1)
    return agg


def train_one_epoch(model, loader, optimizer, loss_fn, device, is_binary: bool):
    model.train()
    running_loss = 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad()
        logits, _ = model(x)
        if is_binary:
            loss = loss_fn(logits, y.float().unsqueeze(1))
        else:
            loss = loss_fn(logits, y.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / max(len(loader.dataset), 1)


def fit_model(model, loaders, train_df, cfg, device, epochs, lr, weight_decay, out_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = get_loss_fn(train_df, is_binary=cfg.is_binary, num_classes=len(cfg.class_names), device=device)
    best_metric = -float("inf")
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, loss_fn, device, is_binary=cfg.is_binary)
        val_outputs = predict_loader(model, loaders["val"], device, is_binary=cfg.is_binary)
        if cfg.is_binary:
            val_metrics = evaluate_binary_outputs(val_outputs)
            monitor = val_metrics["auroc"] if not math.isnan(val_metrics["auroc"]) else val_metrics["f1"]
        else:
            val_metrics = evaluate_multiclass_outputs(val_outputs, cfg.class_names)
            monitor = val_metrics["macro_f1"]
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | val={val_metrics}")
        if monitor > best_metric:
            best_metric = monitor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
    if best_state is None:
        raise RuntimeError("No best model state found.")
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best_model.pt")
    return model


def plot_threshold_sweep(sweep_df: pd.DataFrame, selected_threshold: float, save_path: Path, title: str):
    if sweep_df.empty:
        return
    plt.figure(figsize=(6, 4.5))
    plt.plot(sweep_df["threshold"], sweep_df["sensitivity"], label="Sensitivity")
    plt.plot(sweep_df["threshold"], sweep_df["specificity"], label="Specificity")
    plt.plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
    plt.axvline(selected_threshold, linestyle="--", label=f"Selected={selected_threshold:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.ylim(0.0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_binary_roc(y_true, y_prob, save_path: Path, title: str, ci_info: Optional[Dict[str, float]] = None):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    score = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    label = f"AUROC={score:.3f}"
    if ci_info is not None and not math.isnan(ci_info.get("ci_low", float("nan"))):
        label += f" (95% CI {ci_info['ci_low']:.3f}-{ci_info['ci_high']:.3f})"
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix_any(y_true, y_pred, class_names: List[str], save_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(5.5, 4.8))
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


def multiclass_ovr_auc_ci(y_true: Sequence[int], y_prob: np.ndarray, class_names: List[str], n_bootstrap: int = 2000, seed: int = 42) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    if n == 0:
        return {"macro_ovr_auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_bootstrap_valid": 0}

    def macro_auc_once(yt: np.ndarray, yp: np.ndarray) -> float:
        vals = []
        for c in range(len(class_names)):
            y_bin = (yt == c).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            vals.append(roc_auc_score(y_bin, yp[:, c]))
        return float(np.mean(vals)) if vals else float("nan")

    point = macro_auc_once(y_true, y_prob)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        s = macro_auc_once(yt, yp)
        if not math.isnan(s):
            scores.append(s)

    if len(scores) == 0:
        return {"macro_ovr_auroc": point, "ci_low": float("nan"), "ci_high": float("nan"), "n_bootstrap_valid": 0}

    scores = np.asarray(scores, dtype=float)
    return {
        "macro_ovr_auroc": point,
        "ci_low": float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5)),
        "n_bootstrap_valid": int(len(scores)),
    }


def plot_embedding(features, labels, class_names, save_path: Path, method: str = "umap"):
    n_samples = len(features)
    if n_samples < 3:
        return
    try:
        if method == "umap" and HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42)
            emb = reducer.fit_transform(features)
            title = "UMAP feature visualization"
        else:
            perplexity = min(30, max(2, n_samples - 1))
            reducer = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity)
            emb = reducer.fit_transform(features)
            title = f"t-SNE feature visualization (perplexity={perplexity})"
    except Exception:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        emb = reducer.fit_transform(features)
        title = "PCA feature visualization"

    plt.figure(figsize=(6, 5))
    labels = np.asarray(labels)
    for i, cls in enumerate(class_names):
        mask = labels == i
        if np.sum(mask) == 0:
            continue
        plt.scatter(emb[mask, 0], emb[mask, 1], s=20, alpha=0.8, label=cls)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def build_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("ipatient", as_index=False).agg(target=("target", "max"), n_images=("target", "size"))


def make_patient_cv_splits(patient_df: pd.DataFrame, n_splits: int, seed: int, val_ratio: float) -> List[Dict[str, np.ndarray]]:
    y = patient_df["target"].to_numpy().astype(int)
    patient_ids = patient_df["ipatient"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(patient_ids, y), start=1):
        train_val_patients = patient_ids[train_val_idx]
        train_val_y = y[train_val_idx]
        test_patients = patient_ids[test_idx]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + fold_idx)
        tr_idx, val_idx = next(sss.split(train_val_patients, train_val_y))
        train_patients = train_val_patients[tr_idx]
        val_patients = train_val_patients[val_idx]
        splits.append({
            "fold": fold_idx,
            "train_patients": train_patients,
            "val_patients": val_patients,
            "test_patients": test_patients,
        })
    return splits


def summarize_cv_metrics(df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in metric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        out[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else float("nan")
    return out


def build_augmented_sample_count_table(
    split_dfs: Dict[str, pd.DataFrame],
    aug_mode: str,
    epochs: int,
) -> pd.DataFrame:
    """Summarize image counts before and after augmentation for each split.

    The current pipeline uses online augmentation in the training transform.
    Therefore, augmentation changes the image view sampled during training, but it
    does not physically duplicate rows or increase DataLoader.dataset length.
    Validation and test splits always use deterministic evaluation transforms.
    """
    rows = []
    aug_mode = aug_mode.lower().strip()
    for split_name in ["train", "val", "test"]:
        split_df = split_dfs[split_name]
        augmentation_applied = bool(split_name == "train" and aug_mode != "none")

        target_counts = split_df["target"].value_counts().sort_index().to_dict()
        for target, n in target_counts.items():
            n = int(n)
            rows.append({
                "split": split_name,
                "target": int(target),
                "n_original_images": n,
                "augmentation_applied": augmentation_applied,
                "n_after_augmentation_per_epoch": n,
                "n_image_presentations_across_epochs": int(n * epochs) if split_name == "train" else n,
                "note": (
                    "online augmentation: dataset length is unchanged; each epoch may sample a different transformed view"
                    if augmentation_applied
                    else "no augmentation applied to this split"
                ),
            })

        rows.append({
            "split": split_name,
            "target": "all",
            "n_original_images": int(len(split_df)),
            "augmentation_applied": augmentation_applied,
            "n_after_augmentation_per_epoch": int(len(split_df)),
            "n_image_presentations_across_epochs": int(len(split_df) * epochs) if split_name == "train" else int(len(split_df)),
            "note": (
                "online augmentation: dataset length is unchanged; each epoch may sample a different transformed view"
                if augmentation_applied
                else "no augmentation applied to this split"
            ),
        })

    return pd.DataFrame(rows)


def split_count_totals(count_df: pd.DataFrame) -> Dict[str, int]:
    totals = count_df[count_df["target"].astype(str) == "all"].set_index("split")
    return {
        "n_train_images_after_augmentation_per_epoch": int(totals.loc["train", "n_after_augmentation_per_epoch"]),
        "n_val_images_after_augmentation_per_epoch": int(totals.loc["val", "n_after_augmentation_per_epoch"]),
        "n_test_images_after_augmentation_per_epoch": int(totals.loc["test", "n_after_augmentation_per_epoch"]),
        "n_train_image_presentations_across_epochs": int(totals.loc["train", "n_image_presentations_across_epochs"]),
        "n_val_image_presentations": int(totals.loc["val", "n_image_presentations_across_epochs"]),
        "n_test_image_presentations": int(totals.loc["test", "n_image_presentations_across_epochs"]),
    }


def run_fold(
    fold_idx: int,
    df: pd.DataFrame,
    train_patients: np.ndarray,
    val_patients: np.ndarray,
    test_patients: np.ndarray,
    cfg: TaskConfig,
    args,
    output_root: Path,
    device: torch.device,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    fold_dir = output_root / f"fold_{fold_idx}"
    ensure_dir(fold_dir)

    df_train = df[df["ipatient"].isin(train_patients)].copy().reset_index(drop=True)
    df_val = df[df["ipatient"].isin(val_patients)].copy().reset_index(drop=True)
    df_test = df[df["ipatient"].isin(test_patients)].copy().reset_index(drop=True)

    split_dfs = {"train": df_train, "val": df_val, "test": df_test}
    augmented_count_df = build_augmented_sample_count_table(
        split_dfs=split_dfs,
        aug_mode=args.aug_mode,
        epochs=args.epochs,
    )
    augmented_count_df.to_csv(fold_dir / "augmented_sample_counts_by_split.csv", index=False, encoding="utf-8-sig")

    fold_split_counts = augmented_count_df.rename(columns={
        "n_original_images": "n",
    })[[
        "split",
        "target",
        "n",
        "augmentation_applied",
        "n_after_augmentation_per_epoch",
        "n_image_presentations_across_epochs",
        "note",
    ]]
    fold_split_counts.to_csv(fold_dir / "split_counts.csv", index=False, encoding="utf-8-sig")

    loaders = make_dataloaders_from_splits(
        df_train,
        df_val,
        df_test,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug_mode=args.aug_mode,
    )
    model = ImageBackboneClassifier(
        model_name=args.model_name,
        num_classes=(1 if cfg.is_binary else len(cfg.class_names)),
        pretrained=not args.no_pretrained,
    ).to(device)
    model = fit_model(model, loaders, df_train, cfg, device, args.epochs, args.lr, args.weight_decay, fold_dir)

    val_outputs = predict_loader(model, loaders["val"], device=device, is_binary=cfg.is_binary)
    test_outputs = predict_loader(model, loaders["test"], device=device, is_binary=cfg.is_binary)

    result = {
        "fold": fold_idx,
        "model_name": args.model_name,
        "aug_mode": args.aug_mode,
        "n_train_images": int(len(df_train)),
        "n_val_images": int(len(df_val)),
        "n_test_images": int(len(df_test)),
        "n_train_patients": int(len(np.unique(train_patients))),
        "n_val_patients": int(len(np.unique(val_patients))),
        "n_test_patients": int(len(np.unique(test_patients))),
        **split_count_totals(augmented_count_df),
        "augmentation_count_note": (
            "Current augmentation is online and applied only to the training DataLoader; "
            "it does not increase the physical number of image rows per epoch."
        ),
    }

    if cfg.is_binary:
        image_threshold_info, image_sweep_df = find_binary_threshold(val_outputs["y_true"], val_outputs["y_prob"], min_sensitivity=args.target_min_sensitivity, return_table=True)
        image_threshold = float(image_threshold_info["threshold"])
        test_outputs["y_pred"] = (np.asarray(test_outputs["y_prob"]) >= image_threshold).astype(int)

        patient_val_df = aggregate_patient_level_binary(val_outputs, threshold=0.5, strategy=args.patient_agg)
        patient_threshold_info, patient_sweep_df = find_binary_threshold(patient_val_df["y_true"], patient_val_df["prob"], min_sensitivity=args.target_min_sensitivity, return_table=True)
        patient_threshold = float(patient_threshold_info["threshold"])
        patient_test_df = aggregate_patient_level_binary(test_outputs, threshold=patient_threshold, strategy=args.patient_agg)

        image_metrics = binary_metrics(test_outputs["y_true"], test_outputs["y_prob"], threshold=image_threshold, ci_seed=args.seed + fold_idx)
        patient_metrics = binary_metrics(patient_test_df["y_true"], patient_test_df["prob"], threshold=patient_threshold, ci_seed=args.seed + fold_idx)

        result["threshold_selection"] = {
            "target_min_sensitivity": args.target_min_sensitivity,
            "image_level_from_val": image_threshold_info,
            "patient_level_aggregation": args.patient_agg,
            "patient_level_from_val": patient_threshold_info,
        }
        result["image_level"] = image_metrics
        result["patient_level"] = patient_metrics

        image_sweep_df.to_csv(fold_dir / "threshold_sweep_image_level_val.csv", index=False, encoding="utf-8-sig")
        patient_sweep_df.to_csv(fold_dir / "threshold_sweep_patient_level_val.csv", index=False, encoding="utf-8-sig")
        plot_threshold_sweep(image_sweep_df, image_threshold, fold_dir / "threshold_curve_image_level_val.png", f"fold {fold_idx} image-level threshold sweep")
        plot_threshold_sweep(patient_sweep_df, patient_threshold, fold_dir / "threshold_curve_patient_level_val.png", f"fold {fold_idx} patient-level threshold sweep")

        image_ci = {"ci_low": image_metrics["auroc_ci_low"], "ci_high": image_metrics["auroc_ci_high"]}
        patient_ci = {"ci_low": patient_metrics["auroc_ci_low"], "ci_high": patient_metrics["auroc_ci_high"]}
        plot_binary_roc(test_outputs["y_true"], test_outputs["y_prob"], fold_dir / "roc_image_level.png", f"fold {fold_idx} image-level ROC", ci_info=image_ci)
        plot_binary_roc(patient_test_df["y_true"].to_numpy(), patient_test_df["prob"].to_numpy(), fold_dir / "roc_patient_level.png", f"fold {fold_idx} patient-level ROC", ci_info=patient_ci)

        image_pred_df = pd.DataFrame({
            "fold": fold_idx,
            "model_name": args.model_name,
            "aug_mode": args.aug_mode,
            "ipatient": test_outputs["ipatient"],
            "image_name": test_outputs["image_name"],
            "y_true": test_outputs["y_true"],
            "y_pred": test_outputs["y_pred"],
            "prob_positive": test_outputs["y_prob"],
            "threshold": image_threshold,
        })
        patient_test_df.insert(0, "aug_mode", args.aug_mode)
        patient_test_df.insert(0, "model_name", args.model_name)
        patient_test_df.insert(0, "fold", fold_idx)
        image_pred_df.to_csv(fold_dir / "image_level_predictions.csv", index=False, encoding="utf-8-sig")
        patient_test_df.to_csv(fold_dir / "patient_level_predictions.csv", index=False, encoding="utf-8-sig")
    else:
        image_metrics = evaluate_multiclass_outputs(test_outputs, cfg.class_names)
        patient_test_df = aggregate_patient_level_multiclass(test_outputs, cfg.class_names)
        patient_prob = patient_test_df[[f"prob_{name}" for name in cfg.class_names]].to_numpy()
        patient_metrics = {
            "macro_f1": multiclass_macro_f1(patient_test_df["y_true"], patient_test_df["pred_patient"]),
            "accuracy": float((patient_test_df["y_true"] == patient_test_df["pred_patient"]).mean()),
            "macro_ovr_auroc": evaluate_multiclass_outputs(
                {
                    "y_true": patient_test_df["y_true"].to_numpy(),
                    "y_pred": patient_test_df["pred_patient"].to_numpy(),
                    "y_prob": patient_prob,
                },
                cfg.class_names,
            )["macro_ovr_auroc"],
        }
        result["image_level"] = image_metrics
        result["patient_level"] = patient_metrics

        image_pred_df = pd.DataFrame({
            "fold": fold_idx,
            "model_name": args.model_name,
            "aug_mode": args.aug_mode,
            "ipatient": test_outputs["ipatient"],
            "image_name": test_outputs["image_name"],
            "y_true": test_outputs["y_true"],
            "y_pred": test_outputs["y_pred"],
        })
        for i, name in enumerate(cfg.class_names):
            image_pred_df[f"prob_{name}"] = test_outputs["y_prob"][:, i]
        patient_test_df.insert(0, "aug_mode", args.aug_mode)
        patient_test_df.insert(0, "model_name", args.model_name)
        patient_test_df.insert(0, "fold", fold_idx)
        image_pred_df.to_csv(fold_dir / "image_level_predictions.csv", index=False, encoding="utf-8-sig")
        patient_test_df.to_csv(fold_dir / "patient_level_predictions.csv", index=False, encoding="utf-8-sig")

        plot_confusion_matrix_any(
            test_outputs["y_true"],
            test_outputs["y_pred"],
            cfg.class_names,
            fold_dir / "confusion_matrix_image_level.png",
            f"fold {fold_idx} image-level confusion matrix",
        )
        plot_confusion_matrix_any(
            patient_test_df["y_true"].to_numpy(),
            patient_test_df["pred_patient"].to_numpy(),
            cfg.class_names,
            fold_dir / "confusion_matrix_patient_level.png",
            f"fold {fold_idx} patient-level confusion matrix",
        )

    emb_method = "umap" if HAS_UMAP else "tsne"
    plot_embedding(test_outputs["features"], test_outputs["y_true"], (["negative", "positive"] if cfg.is_binary else cfg.class_names), fold_dir / f"{emb_method}_test_features.png", emb_method)
    save_json(result, fold_dir / "metrics_summary.json")
    return result, image_pred_df, patient_test_df


def run_single_experiment(args) -> Tuple[Dict, Path]:
    set_seed(args.seed)
    excel_path = Path(args.excel)
    image_root = Path(args.image_root)
    output_root = Path(args.output_dir)
    ensure_dir(output_root)
    cfg = get_task_config(args.task)
    experiment_name = f"{cfg.task_name}_{args.model_name}_{args.aug_mode}_cv"
    out_dir = output_root / experiment_name
    ensure_dir(out_dir)

    df = load_metadata(excel_path, args.sheet_name)
    df = filter_by_task(df, cfg)
    df = attach_image_paths(df, image_root)

    missing_df = df[df["image_path"].isna()].copy()
    if len(missing_df) > 0:
        missing_df.to_csv(out_dir / "missing_image_paths.csv", index=False, encoding="utf-8-sig")
    df = df[df["image_path"].notna()].copy().reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No valid images found after path resolution.")

    patient_df = build_patient_table(df)
    patient_df.to_csv(out_dir / "patient_table.csv", index=False, encoding="utf-8-sig")

    class_counts = patient_df["target"].value_counts().sort_index()
    if class_counts.min() < args.n_splits:
        raise ValueError(f"Too few patients in at least one class for {args.n_splits}-fold CV. Patient class counts: {class_counts.to_dict()}")

    folds = make_patient_cv_splits(patient_df, n_splits=args.n_splits, seed=args.seed, val_ratio=args.val_ratio)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Task: {cfg.task_name}")
    print(f"Model: {args.model_name}")
    print(f"Augmentation: {args.aug_mode}")
    print(f"Total usable images: {len(df)}")
    print(f"Total patients: {len(patient_df)}")
    print(f"Patient class counts: {class_counts.to_dict()}")

    all_fold_results = []
    all_image_preds = []
    all_patient_preds = []

    for split in folds:
        fold_idx = split["fold"]
        print(f"\n===== Fold {fold_idx}/{args.n_splits} =====")
        res, image_pred_df, patient_pred_df = run_fold(
            fold_idx=fold_idx,
            df=df,
            train_patients=split["train_patients"],
            val_patients=split["val_patients"],
            test_patients=split["test_patients"],
            cfg=cfg,
            args=args,
            output_root=out_dir,
            device=device,
        )
        flat = {
            "fold": fold_idx,
            "model_name": args.model_name,
            "aug_mode": args.aug_mode,
            "n_train_images": res["n_train_images"],
            "n_val_images": res["n_val_images"],
            "n_test_images": res["n_test_images"],
            "n_train_images_after_augmentation_per_epoch": res["n_train_images_after_augmentation_per_epoch"],
            "n_val_images_after_augmentation_per_epoch": res["n_val_images_after_augmentation_per_epoch"],
            "n_test_images_after_augmentation_per_epoch": res["n_test_images_after_augmentation_per_epoch"],
            "n_train_image_presentations_across_epochs": res["n_train_image_presentations_across_epochs"],
            "n_val_image_presentations": res["n_val_image_presentations"],
            "n_test_image_presentations": res["n_test_image_presentations"],
            "n_train_patients": res["n_train_patients"],
            "n_val_patients": res["n_val_patients"],
            "n_test_patients": res["n_test_patients"],
        }
        for level in ["image_level", "patient_level"]:
            for k, v in res[level].items():
                if isinstance(v, list):
                    continue
                flat[f"{level}_{k}"] = v
        all_fold_results.append(flat)
        all_image_preds.append(image_pred_df)
        all_patient_preds.append(patient_pred_df)

    fold_df = pd.DataFrame(all_fold_results).sort_values("fold")
    fold_df.to_csv(out_dir / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")

    count_cols = [
        "n_train_images",
        "n_val_images",
        "n_test_images",
        "n_train_images_after_augmentation_per_epoch",
        "n_val_images_after_augmentation_per_epoch",
        "n_test_images_after_augmentation_per_epoch",
        "n_train_image_presentations_across_epochs",
        "n_val_image_presentations",
        "n_test_image_presentations",
    ]
    fold_df[["fold", "model_name", "aug_mode"] + count_cols].to_csv(
        out_dir / "cv_augmented_sample_counts_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(all_image_preds, axis=0, ignore_index=True).to_csv(out_dir / "cv_image_level_predictions_all_folds.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_patient_preds, axis=0, ignore_index=True).to_csv(out_dir / "cv_patient_level_predictions_all_folds.csv", index=False, encoding="utf-8-sig")

    if cfg.is_binary:
        image_metric_cols = [
            "image_level_auroc", "image_level_f1", "image_level_sensitivity", "image_level_specificity",
        ]
        patient_metric_cols = [
            "patient_level_auroc", "patient_level_f1", "patient_level_sensitivity", "patient_level_specificity",
        ]
        summary = {
            "task": cfg.task_name,
            "model_name": args.model_name,
            "aug_mode": args.aug_mode,
            "cv_type": f"patient_level_{args.n_splits}_fold",
            "n_total_images": int(len(df)),
            "n_total_patients": int(len(patient_df)),
            "patient_class_counts": {str(k): int(v) for k, v in class_counts.to_dict().items()},
            "augmentation_count_note": (
                "Online augmentation is applied only to training images. "
                "The physical sample count per epoch is unchanged; n_train_image_presentations_across_epochs = n_train_images * epochs."
            ),
            "augmentation_sample_counts_cv": summarize_cv_metrics(fold_df, count_cols),
            "threshold_target_min_sensitivity": float(args.target_min_sensitivity),
            "patient_aggregation": args.patient_agg,
            "image_level_cv": summarize_cv_metrics(fold_df, image_metric_cols),
            "patient_level_cv": summarize_cv_metrics(fold_df, patient_metric_cols),
        }

        all_img = pd.concat(all_image_preds, axis=0, ignore_index=True)
        all_pat = pd.concat(all_patient_preds, axis=0, ignore_index=True)
        img_oof_ci = bootstrap_auc_ci(all_img["y_true"], all_img["prob_positive"], seed=args.seed)
        pat_oof_ci = bootstrap_auc_ci(all_pat["y_true"], all_pat["prob"], seed=args.seed)
        summary["image_level_oof"] = {
            "auroc": img_oof_ci["auroc"],
            "auroc_ci_low": img_oof_ci["ci_low"],
            "auroc_ci_high": img_oof_ci["ci_high"],
            "f1": float(f1_score(all_img["y_true"], all_img["y_pred"], zero_division=0)),
            "sensitivity": float(recall_score(all_img["y_true"], all_img["y_pred"], zero_division=0)),
            "specificity": specificity_score(all_img["y_true"], all_img["y_pred"]),
        }
        summary["patient_level_oof"] = {
            "auroc": pat_oof_ci["auroc"],
            "auroc_ci_low": pat_oof_ci["ci_low"],
            "auroc_ci_high": pat_oof_ci["ci_high"],
            "f1": float(f1_score(all_pat["y_true"], all_pat["pred_patient"], zero_division=0)),
            "sensitivity": float(recall_score(all_pat["y_true"], all_pat["pred_patient"], zero_division=0)),
            "specificity": specificity_score(all_pat["y_true"], all_pat["pred_patient"]),
        }

        plot_binary_roc(all_img["y_true"], all_img["prob_positive"], out_dir / "cv_roc_image_level_oof.png", f"{cfg.task_name} CV image-level OOF ROC", ci_info={"ci_low": img_oof_ci["ci_low"], "ci_high": img_oof_ci["ci_high"]})
        plot_binary_roc(all_pat["y_true"], all_pat["prob"], out_dir / "cv_roc_patient_level_oof.png", f"{cfg.task_name} CV patient-level OOF ROC", ci_info={"ci_low": pat_oof_ci["ci_low"], "ci_high": pat_oof_ci["ci_high"]})

        summary_lines = [
            f"Task: {cfg.task_name}",
            f"Model: {args.model_name}",
            f"Augmentation: {args.aug_mode}",
            f"CV: patient-level {args.n_splits}-fold",
            f"Total usable images: {len(df)}",
            f"Total patients: {len(patient_df)}",
            f"Patient class counts: {class_counts.to_dict()}",
            "",
            "Augmented sample counts by split (fold mean ± std):",
            f"Train images after augmentation per epoch = {summary['augmentation_sample_counts_cv']['n_train_images_after_augmentation_per_epoch_mean']:.1f} ± {summary['augmentation_sample_counts_cv']['n_train_images_after_augmentation_per_epoch_std']:.1f}",
            f"Validation images after augmentation per epoch = {summary['augmentation_sample_counts_cv']['n_val_images_after_augmentation_per_epoch_mean']:.1f} ± {summary['augmentation_sample_counts_cv']['n_val_images_after_augmentation_per_epoch_std']:.1f}",
            f"Test images after augmentation per epoch = {summary['augmentation_sample_counts_cv']['n_test_images_after_augmentation_per_epoch_mean']:.1f} ± {summary['augmentation_sample_counts_cv']['n_test_images_after_augmentation_per_epoch_std']:.1f}",
            f"Train image presentations across {args.epochs} epochs = {summary['augmentation_sample_counts_cv']['n_train_image_presentations_across_epochs_mean']:.1f} ± {summary['augmentation_sample_counts_cv']['n_train_image_presentations_across_epochs_std']:.1f}",
            "Note: augmentation is online and only applied to the training DataLoader; validation/test are not augmented.",
            "",
            "Image-level CV (fold mean ± std):",
            f"AUROC = {summary['image_level_cv']['image_level_auroc_mean']:.3f} ± {summary['image_level_cv']['image_level_auroc_std']:.3f}",
            f"F1 = {summary['image_level_cv']['image_level_f1_mean']:.3f} ± {summary['image_level_cv']['image_level_f1_std']:.3f}",
            f"Sensitivity = {summary['image_level_cv']['image_level_sensitivity_mean']:.3f} ± {summary['image_level_cv']['image_level_sensitivity_std']:.3f}",
            f"Specificity = {summary['image_level_cv']['image_level_specificity_mean']:.3f} ± {summary['image_level_cv']['image_level_specificity_std']:.3f}",
            "",
            "Patient-level CV (fold mean ± std):",
            f"AUROC = {summary['patient_level_cv']['patient_level_auroc_mean']:.3f} ± {summary['patient_level_cv']['patient_level_auroc_std']:.3f}",
            f"F1 = {summary['patient_level_cv']['patient_level_f1_mean']:.3f} ± {summary['patient_level_cv']['patient_level_f1_std']:.3f}",
            f"Sensitivity = {summary['patient_level_cv']['patient_level_sensitivity_mean']:.3f} ± {summary['patient_level_cv']['patient_level_sensitivity_std']:.3f}",
            f"Specificity = {summary['patient_level_cv']['patient_level_specificity_mean']:.3f} ± {summary['patient_level_cv']['patient_level_specificity_std']:.3f}",
            "",
            "OOF summary across all folds:",
            f"Image-level AUROC = {summary['image_level_oof']['auroc']:.3f} (95% CI {summary['image_level_oof']['auroc_ci_low']:.3f}-{summary['image_level_oof']['auroc_ci_high']:.3f})",
            f"Patient-level AUROC = {summary['patient_level_oof']['auroc']:.3f} (95% CI {summary['patient_level_oof']['auroc_ci_low']:.3f}-{summary['patient_level_oof']['auroc_ci_high']:.3f})",
        ]
    else:
        image_metric_cols = ["image_level_macro_f1", "image_level_accuracy", "image_level_macro_ovr_auroc"]
        patient_metric_cols = ["patient_level_macro_f1", "patient_level_accuracy", "patient_level_macro_ovr_auroc"]
        summary = {
            "task": cfg.task_name,
            "model_name": args.model_name,
            "aug_mode": args.aug_mode,
            "cv_type": f"patient_level_{args.n_splits}_fold",
            "n_total_images": int(len(df)),
            "n_total_patients": int(len(patient_df)),
            "patient_class_counts": {str(k): int(v) for k, v in class_counts.to_dict().items()},
            "augmentation_count_note": (
                "Online augmentation is applied only to training images. "
                "The physical sample count per epoch is unchanged; n_train_image_presentations_across_epochs = n_train_images * epochs."
            ),
            "augmentation_sample_counts_cv": summarize_cv_metrics(fold_df, count_cols),
            "image_level_cv": summarize_cv_metrics(fold_df, image_metric_cols),
            "patient_level_cv": summarize_cv_metrics(fold_df, patient_metric_cols),
        }

        all_img = pd.concat(all_image_preds, axis=0, ignore_index=True)
        all_pat = pd.concat(all_patient_preds, axis=0, ignore_index=True)
        img_prob = all_img[[f"prob_{name}" for name in cfg.class_names]].to_numpy()
        pat_prob = all_pat[[f"prob_{name}" for name in cfg.class_names]].to_numpy()

        img_oof = multiclass_ovr_auc_ci(all_img["y_true"], img_prob, cfg.class_names, seed=args.seed)
        pat_oof = multiclass_ovr_auc_ci(all_pat["y_true"], pat_prob, cfg.class_names, seed=args.seed)

        summary["image_level_oof"] = {
            "macro_ovr_auroc": img_oof["macro_ovr_auroc"],
            "ci_low": img_oof["ci_low"],
            "ci_high": img_oof["ci_high"],
            "macro_f1": multiclass_macro_f1(all_img["y_true"], all_img["y_pred"]),
            "accuracy": float((all_img["y_true"] == all_img["y_pred"]).mean()),
        }
        summary["patient_level_oof"] = {
            "macro_ovr_auroc": pat_oof["macro_ovr_auroc"],
            "ci_low": pat_oof["ci_low"],
            "ci_high": pat_oof["ci_high"],
            "macro_f1": multiclass_macro_f1(all_pat["y_true"], all_pat["pred_patient"]),
            "accuracy": float((all_pat["y_true"] == all_pat["pred_patient"]).mean()),
        }

        plot_confusion_matrix_any(
            all_img["y_true"].to_numpy(),
            all_img["y_pred"].to_numpy(),
            cfg.class_names,
            out_dir / "cv_confusion_matrix_image_level_oof.png",
            f"{cfg.task_name} CV image-level OOF confusion matrix",
        )
        plot_confusion_matrix_any(
            all_pat["y_true"].to_numpy(),
            all_pat["pred_patient"].to_numpy(),
            cfg.class_names,
            out_dir / "cv_confusion_matrix_patient_level_oof.png",
            f"{cfg.task_name} CV patient-level OOF confusion matrix",
        )

        summary_lines = [
            f"Task: {cfg.task_name}",
            f"Model: {args.model_name}",
            f"Augmentation: {args.aug_mode}",
            f"CV: patient-level {args.n_splits}-fold",
            f"Total usable images: {len(df)}",
            f"Total patients: {len(patient_df)}",
            f"Patient class counts: {class_counts.to_dict()}",
            "",
            "Augmented sample counts by split (fold mean ± std):",
            f"Train images after augmentation per epoch = {summary['augmentation_sample_counts_cv']['n_train_images_after_augmentation_per_epoch_mean']:.1f} ± {summary['augmentation_sample_counts_cv']['n_train_images_after_augmentation_per_epoch_std']:.1f}",
            f"Validation images after augmentation per epoch = {summary['augmentation_sample_counts_cv']['n_val_images_after_augmentation_per_epoch_mean']:.1f} ± {summary['augmentation_sample_counts_cv']['n_val_images_after_augmentation_per_epoch_std']:.1f}",
            f"Test images after augmentation per epoch = {summary['augmentation_sample_counts_cv']['n_test_images_after_augmentation_per_epoch_mean']:.1f} ± {summary['augmentation_sample_counts_cv']['n_test_images_after_augmentation_per_epoch_std']:.1f}",
            f"Train image presentations across {args.epochs} epochs = {summary['augmentation_sample_counts_cv']['n_train_image_presentations_across_epochs_mean']:.1f} ± {summary['augmentation_sample_counts_cv']['n_train_image_presentations_across_epochs_std']:.1f}",
            "Note: augmentation is online and only applied to the training DataLoader; validation/test are not augmented.",
            "",
            "Image-level CV (fold mean ± std):",
            f"Macro OVR AUROC = {summary['image_level_cv']['image_level_macro_ovr_auroc_mean']:.3f} ± {summary['image_level_cv']['image_level_macro_ovr_auroc_std']:.3f}",
            f"Macro F1 = {summary['image_level_cv']['image_level_macro_f1_mean']:.3f} ± {summary['image_level_cv']['image_level_macro_f1_std']:.3f}",
            f"Accuracy = {summary['image_level_cv']['image_level_accuracy_mean']:.3f} ± {summary['image_level_cv']['image_level_accuracy_std']:.3f}",
            "",
            "Patient-level CV (fold mean ± std):",
            f"Macro OVR AUROC = {summary['patient_level_cv']['patient_level_macro_ovr_auroc_mean']:.3f} ± {summary['patient_level_cv']['patient_level_macro_ovr_auroc_std']:.3f}",
            f"Macro F1 = {summary['patient_level_cv']['patient_level_macro_f1_mean']:.3f} ± {summary['patient_level_cv']['patient_level_macro_f1_std']:.3f}",
            f"Accuracy = {summary['patient_level_cv']['patient_level_accuracy_mean']:.3f} ± {summary['patient_level_cv']['patient_level_accuracy_std']:.3f}",
            "",
            "OOF summary across all folds:",
            f"Image-level Macro OVR AUROC = {summary['image_level_oof']['macro_ovr_auroc']:.3f} (95% CI {summary['image_level_oof']['ci_low']:.3f}-{summary['image_level_oof']['ci_high']:.3f})",
            f"Patient-level Macro OVR AUROC = {summary['patient_level_oof']['macro_ovr_auroc']:.3f} (95% CI {summary['patient_level_oof']['ci_low']:.3f}-{summary['patient_level_oof']['ci_high']:.3f})",
        ]

    save_json(summary, out_dir / "cv_metrics_summary.json")
    (out_dir / "cv_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n=== CV Final summary ===")
    print("\n".join(summary_lines))
    print(f"\nSaved to: {out_dir}")
    return summary, out_dir


def parse_multi_arg(raw: str, allowed: List[str]) -> List[str]:
    vals = [v.strip().lower() for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("Empty comparison list.")
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise ValueError(f"Unsupported values {bad}. Allowed: {allowed}")
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def select_primary_metric(task: str) -> str:
    return "patient_level_auroc" if task in ["task1", "task2"] else "patient_level_macro_ovr_auroc"


def select_secondary_metric(task: str) -> str:
    return "patient_level_f1" if task in ["task1", "task2"] else "patient_level_macro_f1"


def build_comparison_row(summary: Dict) -> Dict[str, object]:
    row: Dict[str, object] = {
        "task": summary["task"],
        "model_name": summary["model_name"],
        "aug_mode": summary["aug_mode"],
        "n_total_images": summary["n_total_images"],
        "n_total_patients": summary["n_total_patients"],
    }
    if "image_level_oof" in summary and "auroc" in summary["image_level_oof"]:
        row.update({
            "image_level_auroc": summary["image_level_oof"]["auroc"],
            "image_level_f1": summary["image_level_oof"]["f1"],
            "image_level_sensitivity": summary["image_level_oof"]["sensitivity"],
            "image_level_specificity": summary["image_level_oof"]["specificity"],
            "patient_level_auroc": summary["patient_level_oof"]["auroc"],
            "patient_level_f1": summary["patient_level_oof"]["f1"],
            "patient_level_sensitivity": summary["patient_level_oof"]["sensitivity"],
            "patient_level_specificity": summary["patient_level_oof"]["specificity"],
        })
    else:
        row.update({
            "image_level_macro_ovr_auroc": summary["image_level_oof"]["macro_ovr_auroc"],
            "image_level_macro_f1": summary["image_level_oof"]["macro_f1"],
            "image_level_accuracy": summary["image_level_oof"]["accuracy"],
            "patient_level_macro_ovr_auroc": summary["patient_level_oof"]["macro_ovr_auroc"],
            "patient_level_macro_f1": summary["patient_level_oof"]["macro_f1"],
            "patient_level_accuracy": summary["patient_level_oof"]["accuracy"],
        })
    return row


def run_experiment_grid(args) -> None:
    model_names = parse_multi_arg(args.compare_models, SUPPORTED_MODELS)
    aug_modes = parse_multi_arg(args.compare_augs, SUPPORTED_AUGS)
    rows = []
    output_root = Path(args.output_dir)
    ensure_dir(output_root)

    for model_name in model_names:
        for aug_mode in aug_modes:
            run_args = argparse.Namespace(**vars(args))
            run_args.model_name = model_name
            run_args.aug_mode = aug_mode
            print("\n" + "=" * 90)
            print(f"Running experiment | task={run_args.task} | model={model_name} | aug={aug_mode}")
            print("=" * 90)
            summary, out_dir = run_single_experiment(run_args)
            row = build_comparison_row(summary)
            row["result_dir"] = str(out_dir)
            rows.append(row)

    comp_df = pd.DataFrame(rows)
    primary_metric = select_primary_metric(args.task)
    secondary_metric = select_secondary_metric(args.task)
    sort_cols = [c for c in [primary_metric, secondary_metric] if c in comp_df.columns]
    comp_df = comp_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    if len(comp_df) > 0:
        comp_df["rank"] = np.arange(1, len(comp_df) + 1)
        comp_df["chosen_for_multimodal"] = ""
        comp_df.loc[0, "chosen_for_multimodal"] = "yes"

    comp_path = output_root / f"comparison_{args.task}.csv"
    comp_df.to_csv(comp_path, index=False, encoding="utf-8-sig")

    lines = [
        f"Task: {args.task}",
        f"Compared models: {', '.join(model_names)}",
        f"Compared augmentation modes: {', '.join(aug_modes)}",
        f"Primary ranking metric: {primary_metric}",
    ]
    if len(comp_df) > 0:
        best = comp_df.iloc[0]
        lines += [
            "",
            "Best configuration:",
            f"model_name = {best['model_name']}",
            f"aug_mode = {best['aug_mode']}",
            f"{primary_metric} = {best.get(primary_metric, np.nan):.3f}",
        ]
        if secondary_metric in comp_df.columns:
            lines.append(f"{secondary_metric} = {best.get(secondary_metric, np.nan):.3f}")
    (output_root / f"comparison_{args.task}.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== Comparison finished ===")
    print("\n".join(lines))
    print(f"Saved comparison table to: {comp_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="ORN panoramic image baseline pipeline with augmentation/model comparison")
    parser.add_argument("--excel", type=str, default="image_data.xlsx")
    parser.add_argument("--sheet_name", type=str, default="image_master")
    parser.add_argument("--image_root", type=str, required=True, help="影像資料夾根目錄")
    parser.add_argument("--output_dir", type=str, default="./orn_image_outputs")
    parser.add_argument("--task", type=str, choices=["task1", "task2", "task3"], required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--target_min_sensitivity", type=float, default=0.8)
    parser.add_argument("--patient_agg", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--model_name", type=str, default="resnet18", choices=SUPPORTED_MODELS)
    parser.add_argument("--aug_mode", type=str, default="light", choices=SUPPORTED_AUGS)
    parser.add_argument("--run_grid", action="store_true", help="是否依 compare_models x compare_augs 跑完整比較")
    parser.add_argument("--compare_models", type=str, default="resnet18,densenet121,efficientnet_b0,mobilenet_v3_small")
    parser.add_argument("--compare_augs", type=str, default="none,light")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.run_grid:
        run_experiment_grid(cli_args)
    else:
        run_single_experiment(cli_args)
