
import os
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, Sequence, List

from PIL import Image, ImageFile

import torch
from torch import nn
from torchvision import models, transforms

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def safe_strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "column": df.columns,
        "missing_count": [df[c].isna().sum() for c in df.columns],
        "missing_ratio": [df[c].isna().mean() for c in df.columns],
    }).sort_values(["missing_ratio", "missing_count"], ascending=False)
    return out


def normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def robust_map_mandible_worst(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = normalize_text_series(s)
    mapping = {
        "none": 0, "no": 0, "0": 0,
        "marginal": 1, "marginal resection": 1, "1": 1,
        "segmental": 2, "segmental resection": 2, "2": 2,
    }
    return x.map(mapping)


def robust_map_reconstruction_max(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = normalize_text_series(s)
    mapping = {
        "none": 0, "no reconstruction": 0, "0": 0,
        "local flap": 1, "local": 1, "1": 1,
        "regional flap": 2, "regional": 2, "2": 2,
        "free flap": 3, "free": 3, "3": 3,
    }
    return x.map(mapping)


def group_tumor_location(x: pd.Series) -> pd.Series:
    s = x.astype(str).str.strip()
    mapping = {
        "Gum": "Oral cavity",
        "Buccal / other oral cavity": "Oral cavity",
        "Floor of mouth": "Oral cavity",
        "Tongue": "Oral cavity",
        "Lip": "Oral cavity",
        "Palate": "Oral cavity",
        "Base of tongue": "Oropharynx",
        "Tonsil": "Oropharynx",
        "Oropharynx": "Oropharynx",
        "Hypopharynx": "Hypopharynx / Larynx",
        "Larynx": "Hypopharynx / Larynx",
        "Pyriform sinus": "Hypopharynx / Larynx",
        "Nasopharynx": "Nasopharynx",
        "Parotid gland": "Salivary gland",
        "Other major salivary glands": "Salivary gland",
    }
    out = s.map(mapping).fillna("Other")
    out = out.replace({"nan": "Other", "None": "Other", "": "Other"})
    return out


def winsorize_series(s: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return s
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def candidate_stems_from_row(row: pd.Series) -> List[str]:
    candidates = []
    for col in ["image_name", "image_name_std", "image_id"]:
        if col in row and pd.notna(row[col]):
            raw = str(row[col]).strip()
            if raw and raw.lower() != "nan":
                candidates.append(raw)
    out = []
    seen = set()
    for x in candidates:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def resolve_image_paths(row: pd.Series, image_root: Path) -> List[Path]:
    """
    一列對多張圖：
    若 image_name = 10241485，接受：
    - 10241485.jpg
    - 10241485-10.jpg
    - 10241485_10.jpg
    - 10241485-osteolytic 0.jpg
    """
    stems = candidate_stems_from_row(row)
    subdirs = []
    if "label_raw" in row and pd.notna(row["label_raw"]):
        subdirs.append(str(row["label_raw"]).strip())
    subdirs += ["", "non_orn", "orn", "visible_orn", "orn_normal", "images", "image"]

    matched = []

    # 直接掃整個資料夾最穩，不猜副檔名/子資料夾
    for p in image_root.rglob("*"):
        if not p.is_file():
            continue
        file_stem = p.stem.strip()
        file_suffix = p.suffix.lower()
        if file_suffix not in IMG_EXTS:
            continue

        for stem in stems:
            if file_stem == stem or file_stem.startswith(stem + "-") or file_stem.startswith(stem + "_"):
                matched.append(p)
                break

    # 去重 + 排序
    uniq = sorted({str(p.resolve()): p.resolve() for p in matched}.values(), key=lambda x: str(x))
    return uniq


def expand_rows_to_images(image_df: pd.DataFrame, image_root: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    unresolved = []

    for _, row in image_df.iterrows():
        paths = resolve_image_paths(row, image_root)
        if len(paths) == 0:
            unresolved.append(row.to_dict())
            continue

        for p in paths:
            new_row = row.copy()
            new_row["image_path"] = str(p)
            new_row["resolved_filename"] = p.name
            rows.append(new_row)

    expanded = pd.DataFrame(rows)
    if unresolved:
        pd.DataFrame(unresolved).to_csv(output_dir / "missing_image_paths.csv", index=False, encoding="utf-8-sig")
    return expanded


class ImageFeatureExtractor:
    def __init__(self, device: torch.device, img_size: int = 224, batch_size: int = 16, pretrained: bool = True):
        self.device = device
        self.batch_size = batch_size
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.model = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten()).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, paths: Sequence[Path]) -> np.ndarray:
        feats = []
        batch = []
        for p in paths:
            img = Image.open(p).convert("L")
            batch.append(self.tf(img))
            if len(batch) == self.batch_size:
                feats.append(self._forward(torch.stack(batch, dim=0)))
                batch = []
        if batch:
            feats.append(self._forward(torch.stack(batch, dim=0)))
        return np.vstack(feats) if feats else np.empty((0, 512), dtype=np.float32)

    def _forward(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.model(x.to(self.device)).cpu().numpy()


def aggregate_patient_image_features(df_feat: pd.DataFrame, strategy: str = "meanmax") -> pd.DataFrame:
    feat_cols = [c for c in df_feat.columns if c.startswith("img_feat_")]
    rows = []
    for pid, sub in df_feat.groupby("ipatient"):
        arr = sub[feat_cols].to_numpy(dtype=np.float32)
        row = {"ipatient": pid, "n_images": int(len(sub))}
        if strategy in ("mean", "meanmax"):
            meanv = arr.mean(axis=0)
            for i, v in enumerate(meanv):
                row[f"img_mean_{i}"] = float(v)
        if strategy in ("max", "meanmax"):
            maxv = arr.max(axis=0)
            for i, v in enumerate(maxv):
                row[f"img_max_{i}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def reduce_image_features(patient_img: pd.DataFrame, n_components: int = 3):
    img_cols = [c for c in patient_img.columns if c.startswith("img_mean_") or c.startswith("img_max_")]
    X = patient_img[img_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_components = min(n_components, Xs.shape[0] - 1, Xs.shape[1])
    if n_components < 1:
        raise ValueError("PCA components 無法設定，可能病人數太少。")

    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(Xs)

    out = patient_img[["ipatient", "n_images"]].copy()
    for i in range(n_components):
        out[f"img_pc{i+1}"] = Z[:, i]

    var_text = "\n".join(
        [f"img_pc{i+1}: explained_variance_ratio={pca.explained_variance_ratio_[i]:.4f}" for i in range(n_components)]
    )
    return out, scaler, pca, var_text


def plot_forest(cph_summary: pd.DataFrame, save_path: Path) -> None:
    df_plot = cph_summary.copy()
    df_plot = df_plot.sort_values("exp(coef)")
    y = np.arange(len(df_plot))

    plt.figure(figsize=(7, max(5, 0.45 * len(df_plot))))
    plt.errorbar(
        df_plot["exp(coef)"],
        y,
        xerr=[
            df_plot["exp(coef)"] - df_plot["exp(coef) lower 95%"],
            df_plot["exp(coef) upper 95%"] - df_plot["exp(coef)"],
        ],
        fmt="o"
    )
    plt.axvline(x=1, linestyle="--")
    plt.yticks(y, df_plot.index)
    plt.xlabel("Hazard Ratio (HR)")
    plt.title("Forest Plot of Multimodal Cox Model")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_km(df_out: pd.DataFrame, save_path: Path) -> dict:
    """Plot ORN-free Kaplan-Meier curves and run a log-rank test.

    The log-rank test evaluates whether the event-time distributions of
    the high-risk and low-risk groups differ. The returned dictionary is
    also saved by main() for thesis reporting.
    """
    kmf = KaplanMeierFitter()
    mask_low = df_out["risk_group"] == "Low"
    mask_high = df_out["risk_group"] == "High"

    low_time = df_out.loc[mask_low, "time"]
    high_time = df_out.loc[mask_high, "time"]
    low_event = df_out.loc[mask_low, "event"]
    high_event = df_out.loc[mask_high, "event"]

    lr = logrank_test(
        low_time,
        high_time,
        event_observed_A=low_event,
        event_observed_B=high_event,
    )

    plt.figure(figsize=(7, 5))

    kmf.fit(low_time, event_observed=low_event, label="Low risk")
    ax = kmf.plot_survival_function(ci_show=True, show_censors=True)

    kmf.fit(high_time, event_observed=high_event, label="High risk")
    kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=True)

    p_text = "p < 0.001" if float(lr.p_value) < 0.001 else f"p = {float(lr.p_value):.3f}"
    plt.title(f"ORN-free Kaplan-Meier Curve by Risk Group\nLog-rank test: {p_text}")
    plt.xlabel("Time (years)")
    plt.ylabel("ORN-free probability")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

    return {
        "logrank_p_value": float(lr.p_value),
        "logrank_test_statistic": float(lr.test_statistic),
        "n_low": int(mask_low.sum()),
        "n_high": int(mask_high.sum()),
        "events_low": int(low_event.sum()),
        "events_high": int(high_event.sum()),
    }


def build_summary_table(cph_summary: pd.DataFrame) -> pd.DataFrame:
    out = cph_summary.copy().reset_index()
    feature_col = out.columns[0]
    out = out.rename(columns={feature_col: "feature"})
    keep_cols = [
        "feature", "coef", "exp(coef)", "se(coef)", "z", "p",
        "coef lower 95%", "coef upper 95%",
        "exp(coef) lower 95%", "exp(coef) upper 95%"
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    return out[keep_cols].sort_values("p", ascending=True)


def main():
    parser = argparse.ArgumentParser(description="Clinical + image multimodal survival analysis (expand one-to-many images)")
    parser.add_argument("--data_path", type=str, default="data_v3.1.xlsx")
    parser.add_argument("--time_sheet", type=str, default="ALL")
    parser.add_argument("--feature_sheet", type=str, default="model_full_pre_orn")
    parser.add_argument("--image_excel", type=str, default="image_data.xlsx")
    parser.add_argument("--image_sheet", type=str, default="image_master")
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="survival_outputs_multimodal_v2")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_pooling", type=str, choices=["mean", "max", "meanmax"], default="meanmax")
    parser.add_argument("--image_pca_components", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # clinical survival
    df_time = safe_strip_columns(pd.read_excel(args.data_path, sheet_name=args.time_sheet))
    df_feat = safe_strip_columns(pd.read_excel(args.data_path, sheet_name=args.feature_sheet))

    id_col = find_first_existing_column(df_feat, ["ipatient", "patient_id", "ID"])
    time_id_col = find_first_existing_column(df_time, [id_col, "ipatient", "patient_id", "ID"])
    index_col = find_first_existing_column(df_time, ["Index Date", "index_date", "Index_Date"])
    end_col = find_first_existing_column(df_time, ["End point", "end_point", "End_Point", "censor_date", "end_date"])
    event_col = find_first_existing_column(df_feat, ["ORN_label", "event", "label"])

    if None in [id_col, time_id_col, index_col, end_col, event_col]:
        raise ValueError("時間表或 feature 表缺少必要欄位。")

    if event_col != "ORN_label":
        df_feat = df_feat.rename(columns={event_col: "ORN_label"})

    df_time_use = df_time[[time_id_col, index_col, end_col]].copy().rename(columns={
        time_id_col: id_col,
        index_col: "Index Date",
        end_col: "End point",
    })

    df = pd.merge(df_feat, df_time_use, on=id_col, how="inner")
    df["Index Date"] = pd.to_datetime(df["Index Date"], errors="coerce")
    df["End point"] = pd.to_datetime(df["End point"], errors="coerce")
    df["ORN_label"] = pd.to_numeric(df["ORN_label"], errors="coerce")
    df["time"] = (df["End point"] - df["Index Date"]).dt.days / 365.25
    df["event"] = df["ORN_label"]
    df = df.dropna(subset=["time", "event"]).copy()
    df = df[df["time"] >= 0].copy()
    df["event"] = df["event"].astype(int)
    df[id_col] = df[id_col].astype(str)

    # image table
    image_df = safe_strip_columns(pd.read_excel(args.image_excel, sheet_name=args.image_sheet))
    if "ipatient" not in image_df.columns:
        raise ValueError("image sheet 缺少 ipatient。")
    image_df["ipatient"] = image_df["ipatient"].astype(str)

    image_root = Path(args.image_root)
    if not image_root.exists():
        raise ValueError(f"image_root 不存在：{image_root}")

    print(f"資料夾存在: {image_root.exists()}")
    print(f"前10個檔案: {[p.name for p in image_root.rglob('*') if p.is_file()][:10]}")

    # 只保留 survival cohort 內的病人
    image_df = image_df[image_df["ipatient"].isin(df[id_col])].copy().reset_index(drop=True)

    expanded_image_df = expand_rows_to_images(image_df, image_root, output_dir)
    print(f"\nimage rows original: {len(image_df)}")
    print(f"image rows expanded: {len(expanded_image_df)}")
    print(f"patients with matched images: {expanded_image_df['ipatient'].nunique() if len(expanded_image_df) else 0}")

    if len(expanded_image_df) == 0:
        raise ValueError("找不到可用影像，請檢查 image_root 或 image_data.xlsx。")

    expanded_image_df.to_csv(output_dir / "expanded_image_table.csv", index=False, encoding="utf-8-sig")

    # feature extraction
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    extractor = ImageFeatureExtractor(
        device=device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        pretrained=(not args.no_pretrained),
    )

    feats = extractor.extract([Path(p) for p in expanded_image_df["image_path"].tolist()])
    image_feat_df = expanded_image_df[["ipatient", "image_path", "resolved_filename"]].copy()
    for i in range(feats.shape[1]):
        image_feat_df[f"img_feat_{i}"] = feats[:, i]

    image_feat_df.to_csv(output_dir / "image_level_features.csv", index=False, encoding="utf-8-sig")

    patient_img = aggregate_patient_image_features(image_feat_df, strategy=args.image_pooling)
    patient_img.to_csv(output_dir / "patient_image_features_raw.csv", index=False, encoding="utf-8-sig")

    patient_img_reduced, scaler, pca, var_text = reduce_image_features(patient_img, n_components=args.image_pca_components)
    patient_img_reduced.to_csv(output_dir / "patient_image_features_pca.csv", index=False, encoding="utf-8-sig")
    save_text(output_dir / "image_pca_summary.txt", var_text)

    # multimodal merge
    df_mm = pd.merge(df, patient_img_reduced, left_on=id_col, right_on="ipatient", how="inner")
    if "ipatient_y" in df_mm.columns:
        df_mm = df_mm.drop(columns=["ipatient_y"])
    if "ipatient_x" in df_mm.columns:
        df_mm = df_mm.rename(columns={"ipatient_x": id_col})

    print(f"\nmultimodal patients used: {len(df_mm)}")

    candidate_features = [
        "mandible_resection_worst",
        "reconstruction_max",
        "tooth_extraction_count",
        "neck_dissection_pre_orn_any",
        "Tumor location",
        "年齡",
        "ECOG",
    ]
    selected_clinical = [c for c in candidate_features if c in df_mm.columns]
    image_pc_cols = [c for c in patient_img_reduced.columns if c.startswith("img_pc")]

    df_model = df_mm[[id_col, "time", "event"] + selected_clinical + image_pc_cols + ["n_images"]].copy()

    if "mandible_resection_worst" in df_model.columns:
        df_model["mandible_resection_worst"] = robust_map_mandible_worst(df_model["mandible_resection_worst"])
    if "reconstruction_max" in df_model.columns:
        df_model["reconstruction_max"] = robust_map_reconstruction_max(df_model["reconstruction_max"])

    for col in ["tooth_extraction_count", "neck_dissection_pre_orn_any", "年齡", "ECOG", "n_images"] + image_pc_cols:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    if "tooth_extraction_count" in df_model.columns:
        df_model["tooth_extraction_count"] = winsorize_series(df_model["tooth_extraction_count"], 0.01, 0.99)
    if "年齡" in df_model.columns:
        df_model["年齡"] = winsorize_series(df_model["年齡"], 0.01, 0.99)
    if "ECOG" in df_model.columns:
        df_model["ECOG"] = winsorize_series(df_model["ECOG"], 0.01, 0.99)

    if "Tumor location" in df_model.columns:
        df_model["Tumor location_grouped"] = group_tumor_location(df_model["Tumor location"])
        df_model = df_model.drop(columns=["Tumor location"])

    summarize_missing(df_model).to_csv(output_dir / "missingness_before_imputation.csv", index=False, encoding="utf-8-sig")

    obj_cols = [c for c in df_model.select_dtypes(include=["object"]).columns if c != id_col]
    for col in obj_cols:
        df_model[col] = df_model[col].fillna("Unknown")

    num_cols = [c for c in df_model.select_dtypes(include=[np.number]).columns if c != "event"]
    for col in num_cols:
        if df_model[col].isna().all():
            df_model = df_model.drop(columns=[col])
        else:
            df_model[col] = df_model[col].fillna(df_model[col].median())

    if "Tumor location_grouped" in df_model.columns:
        dummies = pd.get_dummies(df_model["Tumor location_grouped"], prefix="Tumor location_grouped", drop_first=True)
        df_model = pd.concat([df_model.drop(columns=["Tumor location_grouped"]), dummies], axis=1)

    summarize_missing(df_model).to_csv(output_dir / "missingness_after_imputation.csv", index=False, encoding="utf-8-sig")

    feature_cols = [c for c in df_model.columns if c not in [id_col, "time", "event"]]
    constant_cols = [c for c in feature_cols if df_model[c].nunique(dropna=True) <= 1]
    if len(constant_cols) > 0:
        df_model = df_model.drop(columns=constant_cols)

    feature_cols = [c for c in df_model.columns if c not in [id_col, "time", "event"]]
    clinical_only_cols = [c for c in feature_cols if not c.startswith("img_pc")]
    multimodal_cols = feature_cols.copy()

    # clinical-only
    df_clin = df_model[["time", "event"] + clinical_only_cols].copy()
    cph_clin = CoxPHFitter(penalizer=0.20, l1_ratio=0.0)
    cph_clin.fit(df_clin, duration_col="time", event_col="event")
    clinical_cindex = float(cph_clin.concordance_index_)

    # multimodal
    df_multi = df_model[["time", "event"] + multimodal_cols].copy()
    cph_multi = CoxPHFitter(penalizer=0.20, l1_ratio=0.0)
    cph_multi.fit(df_multi, duration_col="time", event_col="event")
    multimodal_cindex = float(cph_multi.concordance_index_)

    build_summary_table(cph_clin.summary).to_csv(output_dir / "cox_clinical_only.csv", index=False, encoding="utf-8-sig")
    mm_summary = build_summary_table(cph_multi.summary)
    mm_summary.to_csv(output_dir / "cox_multimodal.csv", index=False, encoding="utf-8-sig")

    try:
        ph_test = proportional_hazard_test(cph_multi, df_multi, time_transform="rank")
        ph_df = ph_test.summary.reset_index().rename(columns={"index": "feature"})
        ph_df.to_csv(output_dir / "proportional_hazards_test_multimodal.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        save_text(output_dir / "proportional_hazards_test_multimodal_error.txt", str(e))

    df_out = df_model[[id_col, "time", "event"]].copy()
    df_out["risk_score"] = cph_multi.predict_partial_hazard(df_multi).values
    median_risk = df_out["risk_score"].median()
    df_out["risk_group"] = np.where(df_out["risk_score"] >= median_risk, "High", "Low")
    df_out.to_csv(output_dir / "patient_risk_scores_multimodal.csv", index=False, encoding="utf-8-sig")

    logrank_result = plot_km(df_out, output_dir / "km_curve_multimodal.png")
    pd.DataFrame([logrank_result]).to_csv(output_dir / "logrank_test_high_vs_low.csv", index=False, encoding="utf-8-sig")
    logrank_p = logrank_result["logrank_p_value"]
    plot_forest(cph_multi.summary, output_dir / "forest_plot_multimodal.png")

    summary_text = f"""ORN multimodal survival analysis summary
============================================================
Patients used: {len(df_out)}
Events: {int(df_out['event'].sum())}
Censored: {int((df_out['event'] == 0).sum())}
Median follow-up / time-to-event (years): {float(df_out['time'].median()):.3f}

Original image table rows: {len(image_df)}
Expanded matched image rows: {len(expanded_image_df)}
Patients with matched images: {expanded_image_df['ipatient'].nunique()}

Clinical-only C-index: {clinical_cindex:.4f}
Multimodal C-index: {multimodal_cindex:.4f}
C-index improvement: {multimodal_cindex - clinical_cindex:+.4f}

Log-rank p-value (High vs Low risk): {logrank_p:.6f}
Image pooling: {args.image_pooling}
Image PCA components: {len(image_pc_cols)}

Final multimodal features:
""" + "\n".join([f"- {c}" for c in multimodal_cols])

    print("\n" + summary_text)
    save_text(output_dir / "summary_multimodal.txt", summary_text)
    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
