from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import ImageFile
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_auc_binary(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def specificity_score(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


def bootstrap_auc_ci_binary(y_true, y_prob, n_bootstrap: int = 1000, seed: int = 42) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(roc_auc_score(yt, yp))
    if not vals:
        return {"auroc": safe_auc_binary(y_true, y_prob), "ci_low": float("nan"), "ci_high": float("nan")}
    vals = np.asarray(vals)
    return {"auroc": safe_auc_binary(y_true, y_prob), "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5))}


def binary_metrics(y_true, y_prob, threshold: float = 0.5, seed: int = 42) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    ci = bootstrap_auc_ci_binary(y_true, y_prob, seed=seed)
    return {
        "threshold": float(threshold),
        "auroc": ci["auroc"],
        "auroc_ci_low": ci["ci_low"],
        "auroc_ci_high": ci["ci_high"],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity_score(y_true, y_pred),
    }


def multiclass_macro_ovr_auc(y_true, y_prob, n_classes: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    vals = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        vals.append(roc_auc_score(y_bin, y_prob[:, c]))
    return float(np.mean(vals)) if vals else float("nan")


def multiclass_metrics(y_true, y_prob, n_classes: int) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.argmax(y_prob, axis=1)
    return {"macro_ovr_auroc": multiclass_macro_ovr_auc(y_true, y_prob, n_classes), "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)), "accuracy": float(accuracy_score(y_true, y_pred))}


def summarize_cv(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, float]:
    out = {}
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        out[f"{c}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


class TaskInfo:
    def __init__(self, task: str):
        if task == "task1":
            self.name = "task1_visible_orn_vs_non_orn"
            self.label_col = "label_task1"
            self.class_names = ["negative", "positive"]
            self.is_binary = True
        elif task == "task2":
            self.name = "task2_orn_normal_vs_non_orn"
            self.label_col = "label_task2"
            self.class_names = ["negative", "positive"]
            self.is_binary = True
        elif task == "task3":
            self.name = "task3_multiclass_exploration"
            self.label_col = "label_task3"
            self.class_names = ["non_orn", "orn_normal", "visible_orn"]
            self.is_binary = False
        else:
            raise ValueError(f"Unknown task: {task}")


def load_metadata(excel: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(excel, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def filter_task(df: pd.DataFrame, task: TaskInfo) -> pd.DataFrame:
    out = df.copy()
    out[task.label_col] = out[task.label_col].astype(str).str.strip().str.lower()
    out["ipatient"] = out["ipatient"].astype(str)
    if task.is_binary:
        out = out[out[task.label_col] != "exclude"].copy()
        out = out[out[task.label_col].isin(["negative", "positive"])].copy()
        out["target"] = out[task.label_col].map({"negative": 0, "positive": 1}).astype(int)
    else:
        cls2idx = {c: i for i, c in enumerate(task.class_names)}
        out = out[out[task.label_col].isin(task.class_names)].copy()
        out["target"] = out[task.label_col].map(cls2idx).astype(int)
    return out.reset_index(drop=True)


def candidate_stems_from_row(row: pd.Series) -> List[str]:
    stems = []
    for col in ["image_name_std", "image_name", "image_id"]:
        if col in row and pd.notna(row[col]):
            x = str(row[col]).strip()
            if x and x.lower() != "nan":
                stems.append(x)
    return list(dict.fromkeys(stems))


def resolve_image_path(row: pd.Series, image_root: Path) -> Optional[Path]:
    stems = candidate_stems_from_row(row)
    subdirs = ["", "image", "images", "non_orn", "orn", "visible_orn", "orn_normal"]
    tried = set()
    for sd in subdirs:
        for stem in stems:
            for ext in [""] + IMG_EXTS:
                p = image_root / sd / f"{stem}{ext}" if sd else image_root / f"{stem}{ext}"
                key = str(p)
                if key in tried:
                    continue
                tried.add(key)
                if p.exists() and p.is_file():
                    return p
    for stem in stems:
        for ext in IMG_EXTS:
            hits = list(image_root.rglob(f"{stem}{ext}"))
            if hits:
                return hits[0]
    return None


def attach_image_paths(df: pd.DataFrame, image_root: Path, out_dir: Path) -> pd.DataFrame:
    ensure_dir(out_dir)
    out = df.copy()
    out["image_path"] = out.apply(lambda r: resolve_image_path(r, image_root), axis=1)
    missing = out[out["image_path"].isna()].copy()
    if len(missing) > 0:
        missing.to_csv(out_dir / "missing_image_paths.csv", index=False, encoding="utf-8-sig")
    out = out[out["image_path"].notna()].copy()
    out["image_path"] = out["image_path"].apply(lambda p: str(p))
    return out.reset_index(drop=True)


def build_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, sub in df.groupby("ipatient"):
        target = int(sub["target"].mode().iloc[0])
        rows.append({"ipatient": pid, "target": target, "n_images": int(len(sub))})
    return pd.DataFrame(rows)


def make_patient_cv_splits(patient_df: pd.DataFrame, n_splits: int, seed: int, val_ratio: float):
    y = patient_df["target"].to_numpy().astype(int)
    pids = patient_df["ipatient"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fold, (trainval_idx, test_idx) in enumerate(skf.split(pids, y), start=1):
        trainval_pids = pids[trainval_idx]
        trainval_y = y[trainval_idx]
        test_pids = pids[test_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + fold)
        tr_idx, va_idx = next(sss.split(trainval_pids, trainval_y))
        splits.append({"fold": fold, "train_patients": trainval_pids[tr_idx], "val_patients": trainval_pids[va_idx], "test_patients": test_pids})
    return splits


def safe_copy_image(src: Path, dst: Path, use_symlink: bool = False) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if use_symlink:
        try:
            dst.symlink_to(src.resolve())
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def make_yolo_cls_dataset(fold_dir: Path, train_df: pd.DataFrame, val_df: pd.DataFrame, task: TaskInfo, use_symlink: bool = False) -> Path:
    dataset_dir = fold_dir / "yolo_cls_dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        for i, row in split_df.reset_index(drop=True).iterrows():
            class_name = task.class_names[int(row["target"])]
            src = Path(row["image_path"])
            ext = src.suffix if src.suffix else ".jpg"
            dst = dataset_dir / split_name / class_name / f"{row['ipatient']}_{i}_{src.stem}{ext}"
            safe_copy_image(src, dst, use_symlink=use_symlink)
    return dataset_dir


def read_ultralytics_prediction(model, image_paths: Sequence[str], task: TaskInfo, imgsz: int, batch: int, device: str) -> pd.DataFrame:
    results = model.predict(source=[str(p) for p in image_paths], imgsz=imgsz, batch=batch, device=device, verbose=False)
    rows = []
    for p, r in zip(image_paths, results):
        probs = r.probs.data.detach().cpu().numpy().astype(float)
        pred = int(np.argmax(probs))
        row = {"image_path": str(p), "y_pred": pred}
        for i, name in enumerate(task.class_names):
            row[f"prob_{name}"] = float(probs[i]) if i < len(probs) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_patient_predictions(pred_df: pd.DataFrame, task: TaskInfo) -> pd.DataFrame:
    rows = []
    prob_cols = [f"prob_{c}" for c in task.class_names]
    for pid, sub in pred_df.groupby("ipatient"):
        row = {"ipatient": pid, "y_true": int(sub["y_true"].mode().iloc[0]), "n_images": int(len(sub))}
        for c in prob_cols:
            row[c] = float(sub[c].mean())
        probs = np.array([row[c] for c in prob_cols], dtype=float)
        row["y_pred"] = int(np.argmax(probs))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_predictions(pred_df: pd.DataFrame, task: TaskInfo, level: str, seed: int = 42) -> Dict[str, float]:
    y_true = pred_df["y_true"].to_numpy().astype(int)
    if task.is_binary:
        y_prob = pred_df["prob_positive"].to_numpy().astype(float)
        return {f"{level}_{k}": v for k, v in binary_metrics(y_true, y_prob, threshold=0.5, seed=seed).items()}
    y_prob = pred_df[[f"prob_{c}" for c in task.class_names]].to_numpy(dtype=float)
    return {f"{level}_{k}": v for k, v in multiclass_metrics(y_true, y_prob, n_classes=len(task.class_names)).items()}


def run_one_fold(model_name: str, fold_info: Dict, df: pd.DataFrame, task: TaskInfo, args, run_dir: Path):
    from ultralytics import YOLO
    fold = int(fold_info["fold"])
    fold_dir = run_dir / f"fold_{fold}"
    ensure_dir(fold_dir)
    train_df = df[df["ipatient"].isin(fold_info["train_patients"])].copy().reset_index(drop=True)
    val_df = df[df["ipatient"].isin(fold_info["val_patients"])].copy().reset_index(drop=True)
    test_df = df[df["ipatient"].isin(fold_info["test_patients"])].copy().reset_index(drop=True)
    dataset_dir = make_yolo_cls_dataset(fold_dir, train_df, val_df, task, use_symlink=args.symlink)
    model = YOLO(model_name)
    model.train(data=str(dataset_dir), epochs=args.epochs, imgsz=args.img_size, batch=args.batch_size, patience=args.patience, lr0=args.lr, workers=args.workers, device=args.device, project=str(fold_dir), name="train", exist_ok=True, verbose=False)
    best_pt = fold_dir / "train" / "weights" / "best.pt"
    if best_pt.exists():
        model = YOLO(str(best_pt))
    pred = read_ultralytics_prediction(model, test_df["image_path"].tolist(), task, args.img_size, args.batch_size, args.device)
    key = test_df[["ipatient", "image_path", "target"]].copy()
    key["image_path"] = key["image_path"].astype(str)
    pred["image_path"] = pred["image_path"].astype(str)
    pred = pred.merge(key, on="image_path", how="left").rename(columns={"target": "y_true"})
    pred.to_csv(fold_dir / "image_level_predictions.csv", index=False, encoding="utf-8-sig")
    patient_pred = aggregate_patient_predictions(pred, task)
    patient_pred.to_csv(fold_dir / "patient_level_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = {"fold": fold, "model": model_name, "n_train_images": int(len(train_df)), "n_val_images": int(len(val_df)), "n_test_images": int(len(test_df)), "n_train_patients": int(len(np.unique(fold_info["train_patients"]))), "n_val_patients": int(len(np.unique(fold_info["val_patients"]))), "n_test_patients": int(len(np.unique(fold_info["test_patients"])))}
    metrics.update(evaluate_predictions(pred, task, "image_level", args.seed + fold))
    metrics.update(evaluate_predictions(patient_pred, task, "patient_level", args.seed + fold))
    with open(fold_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if args.clean_fold_dataset:
        shutil.rmtree(dataset_dir, ignore_errors=True)
    return metrics, pred, patient_pred


def run_model(model_name: str, df: pd.DataFrame, patient_df: pd.DataFrame, task: TaskInfo, args, output_root: Path) -> Optional[Dict]:
    run_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_")
    run_dir = output_root / task.name / run_name
    ensure_dir(run_dir)
    class_counts = patient_df["target"].value_counts().sort_index()
    if class_counts.min() < args.n_splits:
        raise ValueError(f"Too few patients for {args.n_splits}-fold CV: {class_counts.to_dict()}")
    folds = make_patient_cv_splits(patient_df, args.n_splits, args.seed, args.val_ratio)
    fold_rows, all_img, all_pat = [], [], []
    try:
        for fold_info in folds:
            print(f"\n[{model_name}] Fold {fold_info['fold']}/{args.n_splits}")
            metrics, pred, patient_pred = run_one_fold(model_name, fold_info, df, task, args, run_dir)
            fold_rows.append(metrics)
            all_img.append(pred.assign(fold=int(fold_info["fold"]), model=model_name))
            all_pat.append(patient_pred.assign(fold=int(fold_info["fold"]), model=model_name))
    except Exception as e:
        err = {"model": model_name, "error": str(e), "note": "If this weight name is unsupported, replace --models with a valid Ultralytics classification .pt weight path/name."}
        with open(run_dir / "FAILED.json", "w", encoding="utf-8") as f:
            json.dump(err, f, ensure_ascii=False, indent=2)
        print(f"[FAILED] {model_name}: {e}")
        return None
    fold_df = pd.DataFrame(fold_rows).sort_values("fold")
    fold_df.to_csv(run_dir / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    all_img_df = pd.concat(all_img, ignore_index=True)
    all_pat_df = pd.concat(all_pat, ignore_index=True)
    all_img_df.to_csv(run_dir / "cv_image_level_predictions_all_folds.csv", index=False, encoding="utf-8-sig")
    all_pat_df.to_csv(run_dir / "cv_patient_level_predictions_all_folds.csv", index=False, encoding="utf-8-sig")
    if task.is_binary:
        image_cols = ["image_level_auroc", "image_level_f1", "image_level_sensitivity", "image_level_specificity"]
        patient_cols = ["patient_level_auroc", "patient_level_f1", "patient_level_sensitivity", "patient_level_specificity"]
        oof_img = binary_metrics(all_img_df["y_true"], all_img_df["prob_positive"], 0.5, args.seed)
        oof_pat = binary_metrics(all_pat_df["y_true"], all_pat_df["prob_positive"], 0.5, args.seed)
        summary = {"task": task.name, "model": model_name, "n_images": int(len(df)), "n_patients": int(len(patient_df)), "patient_class_counts": {str(k): int(v) for k, v in class_counts.to_dict().items()}, "image_level_cv": summarize_cv(fold_df, image_cols), "patient_level_cv": summarize_cv(fold_df, patient_cols), "image_level_oof": oof_img, "patient_level_oof": oof_pat}
        flat = {"task": task.name, "model": model_name, "n_images": int(len(df)), "n_patients": int(len(patient_df)), "image_oof_auroc": oof_img["auroc"], "image_oof_f1": oof_img["f1"], "image_oof_sensitivity": oof_img["sensitivity"], "image_oof_specificity": oof_img["specificity"], "patient_oof_auroc": oof_pat["auroc"], "patient_oof_f1": oof_pat["f1"], "patient_oof_sensitivity": oof_pat["sensitivity"], "patient_oof_specificity": oof_pat["specificity"], "patient_oof_auroc_ci_low": oof_pat["auroc_ci_low"], "patient_oof_auroc_ci_high": oof_pat["auroc_ci_high"]}
    else:
        image_cols = ["image_level_macro_ovr_auroc", "image_level_macro_f1", "image_level_accuracy"]
        patient_cols = ["patient_level_macro_ovr_auroc", "patient_level_macro_f1", "patient_level_accuracy"]
        img_prob = all_img_df[[f"prob_{c}" for c in task.class_names]].to_numpy(dtype=float)
        pat_prob = all_pat_df[[f"prob_{c}" for c in task.class_names]].to_numpy(dtype=float)
        oof_img = multiclass_metrics(all_img_df["y_true"], img_prob, len(task.class_names))
        oof_pat = multiclass_metrics(all_pat_df["y_true"], pat_prob, len(task.class_names))
        summary = {"task": task.name, "model": model_name, "n_images": int(len(df)), "n_patients": int(len(patient_df)), "patient_class_counts": {str(k): int(v) for k, v in class_counts.to_dict().items()}, "image_level_cv": summarize_cv(fold_df, image_cols), "patient_level_cv": summarize_cv(fold_df, patient_cols), "image_level_oof": oof_img, "patient_level_oof": oof_pat}
        flat = {"task": task.name, "model": model_name, "n_images": int(len(df)), "n_patients": int(len(patient_df)), "image_oof_macro_ovr_auroc": oof_img["macro_ovr_auroc"], "image_oof_macro_f1": oof_img["macro_f1"], "image_oof_accuracy": oof_img["accuracy"], "patient_oof_macro_ovr_auroc": oof_pat["macro_ovr_auroc"], "patient_oof_macro_f1": oof_pat["macro_f1"], "patient_oof_accuracy": oof_pat["accuracy"]}
    with open(run_dir / "cv_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return flat


def parse_models(models_arg: str) -> List[str]:
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def main():
    parser = argparse.ArgumentParser(description="Simple Ultralytics YOLO classification CV for ORN panoramic image tasks")
    parser.add_argument("--excel", type=str, default="image_data.xlsx")
    parser.add_argument("--sheet_name", type=str, default="image_master")
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["task1", "task2", "task3"], default="task1")
    parser.add_argument("--models", type=str, default="yolo11n-cls.pt,yolo12n-cls.pt,yolo26n-cls.pt", help="Comma-separated Ultralytics classification weights. Unsupported names will be logged as FAILED.")
    parser.add_argument("--output_dir", type=str, default="orn_yolo_cls_outputs")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="0", help="'0' for first GPU, or 'cpu'.")
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--clean_fold_dataset", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    output_root = Path(args.output_dir)
    ensure_dir(output_root)
    task = TaskInfo(args.task)
    df = load_metadata(Path(args.excel), args.sheet_name)
    df = filter_task(df, task)
    df = attach_image_paths(df, Path(args.image_root), output_root / task.name)
    if len(df) == 0:
        raise RuntimeError("No valid images found. Check --image_root and image_data.xlsx.")
    patient_df = build_patient_table(df)
    patient_df.to_csv(output_root / task.name / "patient_table.csv", index=False, encoding="utf-8-sig")
    print(f"Task: {task.name}")
    print(f"Usable images: {len(df)}")
    print(f"Usable patients: {len(patient_df)}")
    print(f"Patient class counts: {patient_df['target'].value_counts().sort_index().to_dict()}")
    print(f"Models: {parse_models(args.models)}")
    rows = []
    for model_name in parse_models(args.models):
        result = run_model(model_name, df, patient_df, task, args, output_root)
        if result is not None:
            rows.append(result)
    if rows:
        comp = pd.DataFrame(rows)
        sort_col = "patient_oof_auroc" if task.is_binary else "patient_oof_macro_ovr_auroc"
        if sort_col in comp.columns:
            comp = comp.sort_values(sort_col, ascending=False)
        comp.to_csv(output_root / task.name / "comparison_yolo_models.csv", index=False, encoding="utf-8-sig")
        print("\n===== YOLO model comparison =====")
        print(comp.to_string(index=False))
    else:
        print("No model finished successfully. Check FAILED.json under each model folder.")


if __name__ == "__main__":
    main()
