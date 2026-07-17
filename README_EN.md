# Multimodal AI-Based Risk Stratification for Osteoradionecrosis of the Jaw in Patients with Head and Neck Cancer

This repository integrates structured clinical and treatment data, panoramic radiographs, deep-learning image models, multimodal fusion, survival analysis, and a patient-level risk explanation interface to investigate risk stratification for osteoradionecrosis of the jaw (ORN) in patients with head and neck cancer.

> This code is intended solely for academic research and retrospective analysis. It must not be used directly for clinical diagnosis, treatment planning, or follow-up decisions.

## Research Tasks

| Task | Classification setting | Description |
|---|---|---|
| Task 1 | `visible_orn` vs. `non_orn` | Identifies ORN cases with radiographically visible lesions. |
| Task 2 | `orn_normal` vs. `non_orn` | Identifies clinically diagnosed ORN cases without obvious radiographic lesions. |
| Task 3 | `non_orn` / `orn_normal` / `visible_orn` | Exploratory three-class classification. |

All major image and multimodal experiments use patient-level data splitting to prevent images from the same patient from appearing in both training and test sets.

---

## Repository Files

| File | Purpose |
|---|---|
| `pipeline_final_beeswarm_threshold_bw.py` | Main structured-data pipeline. Compares Clinical, Treatment, and Full models and performs preprocessing, univariable analysis, model comparison, hyperparameter search, repeated stratified 5-fold cross-validation, ROC/PR analysis, calibration analysis, decision-curve analysis, permutation importance, and SHAP beeswarm visualization. Figures are generated in a grayscale-friendly format. |
| `orn_image_baseline_pipeline_v8.py` | Panoramic-radiograph baseline pipeline. Supports Tasks 1–3, patient-level 5-fold cross-validation, image-level and patient-level evaluation, out-of-fold predictions, bootstrap 95% confidence intervals, multiple CNN backbones, and online augmentation comparisons. |
| `orn_multimodal_pipeline_v1.py` | Binary multimodal baseline pipeline for Tasks 1 and 2. Compares structured-only, image-only, and multimodal inputs. |
| `orn_multimodal_pipeline_task3_tuned.py` | Tuned multimodal pipeline for Task 3. Supports PCA fitted within training folds, `pca_concat` feature-level fusion, and `weighted_prob` decision-level fusion. |
| `orn_multimodal_survival_logrank.py` | Clinical-image multimodal survival-analysis pipeline. Outputs Cox proportional hazards models, Kaplan–Meier curves, log-rank tests, forest plots, and patient risk scores. |
| `orn_feature_ablation_old.py` | Feature-removal ablation experiment for the structured-data model. Evaluates AUC changes after removing the top five features, jaw-resection features, reconstruction features, tooth-extraction features, or radiation-dose features. Retained mainly for thesis-result reproduction. |
| `orn_yolo_cls_simple_cv.py` | YOLO classification experiment with cross-validation for panoramic-radiograph classification. |
| `yolo11n-cls.pt` | YOLO11n classification model weights. |
| `orn_experiment_pipeline.json` | Experiment configuration and workflow record. Some scripts still rely primarily on constants defined in the source code, so confirm whether this file is actually loaded before editing it. |
| `orn_risk_ui_true_pipelines_v11.zip` | Packaged patient-level risk stratification and explanation interface. |
| `ORN輸出介面.ipynb` | Notebook for preparing or displaying patient-level model outputs and interface content. |

---

## Data Requirements

### 1. Structured Data

Default file:

```text
data_v3.1.xlsx
```

Primary worksheet:

```text
model_full_pre_orn
```

Survival analysis additionally uses:

```text
ALL
```

Commonly required columns:

```text
ipatient
ORN_label
Index Date
End point
orn_diagnosis_date
censor_date
reference_date_for_model
```

Only information available at the index time or before ORN diagnosis should be used for model development. Post-event information must be excluded to prevent temporal data leakage.

### 2. Image Index File

Default file:

```text
image_data.xlsx
```

Primary worksheet:

```text
image_master
```

Commonly used columns:

```text
ipatient
image_name
image_name_std
image_id
label_task1
label_task2
label_task3
```

Images may be stored in one root directory or organized into subdirectories such as:

```text
Image/
├─ non_orn/
├─ orn/
├─ orn_normal/
└─ visible_orn/
```

The scripts resolve image paths using `image_name_std`, `image_name`, or `image_id` and common image extensions.

### 3. Data Privacy

Patient data, original radiographs, and identifiable information must not be committed to a public Git repository. At minimum, add the following entries to `.gitignore`:

```gitignore
# Patient data
data_v*.xlsx
image_data*.xlsx
Image/
images/

# Outputs
orn_*_outputs*/
survival_outputs*/
orn_thesis_final_outputs_*/

# Python
.venv/
__pycache__/
*.pyc

# Notebook
.ipynb_checkpoints/
```

---

## Environment Setup

Python 3.10 or later and an isolated virtual environment are recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install PyTorch according to the local CPU/CUDA environment, then install the remaining packages:

```bash
pip install pandas numpy scipy matplotlib scikit-learn openpyxl pillow
pip install torch torchvision
pip install shap xgboost umap-learn lifelines ultralytics
pip install jupyter ipykernel
```

Notes:

- `xgboost`, `shap`, and `umap-learn` are optional for some analyses.
- `lifelines` is required for survival analysis.
- `ultralytics` is required for YOLO classification experiments.
- The user-interface package may require additional dependencies. Check its source imports or included requirements file after extraction.

---

## Recommended Execution Order

```text
1. Structured-data modeling
2. Image baseline and augmentation/backbone comparison
3. Multimodal modeling for Tasks 1 and 2
4. Tuned multimodal modeling for Task 3
5. Feature-removal ablation analysis
6. Multimodal survival analysis
7. Patient-level risk output and interface
8. YOLO classification as a supplementary image experiment
```

---

## Usage

### 1. Structured-Data Modeling

This script currently specifies the input path through constants in the source code:

```python
DATA_PATH = Path("data_v3.1.xlsx")
SHEET_NAME = "model_full_pre_orn"
```

Place the data file in the project root and run:

```bash
python pipeline_final_beeswarm_threshold_bw.py
```

Main output structure:

```text
orn_thesis_final_outputs_<timestamp>/
├─ EDA/
├─ model_comparison/
├─ figures/
├─ tables/
├─ report_ready/
├─ thesis_level_summary.txt
└─ README.txt
```

The pipeline compares:

- Logistic Regression
- Random Forest
- Support Vector Machine
- XGBoost, when installed

It evaluates three feature configurations:

- `clinical_model`
- `treatment_model`
- `full_model`

### 2. Image Baseline Model

Example for Task 1 on Windows:

```bash
python orn_image_baseline_pipeline_v8.py ^
  --excel image_data.xlsx ^
  --sheet_name image_master ^
  --image_root "C:\path\to\Image" ^
  --output_dir "./orn_image_outputs" ^
  --task task1 ^
  --model_name resnet18 ^
  --aug_mode light ^
  --epochs 20 ^
  --n_splits 5 ^
  --patient_agg max
```

On Linux or macOS, replace `^` with `\`.

For Task 2 or Task 3, use:

```bash
--task task2
```

or

```bash
--task task3
```

Available backbones:

```text
resnet18
densenet121
efficientnet_b0
mobilenet_v3_small
```

Available augmentation modes:

```text
none
light
strong
```

To compare multiple backbones and augmentation modes:

```bash
python orn_image_baseline_pipeline_v8.py ^
  --excel image_data.xlsx ^
  --sheet_name image_master ^
  --image_root "C:\path\to\Image" ^
  --output_dir "./orn_image_grid" ^
  --task task1 ^
  --run_grid ^
  --compare_models "resnet18,densenet121,efficientnet_b0,mobilenet_v3_small" ^
  --compare_augs "none,light"
```

#### Online Augmentation

Random augmentation is applied only to the training set. Validation and test images are not randomly augmented. Augmented images are generated dynamically and are not saved as separate files, so the physical dataset size remains unchanged within each epoch.

The total number of image presentations during training is calculated as:

```text
training image presentations = training images per epoch × number of epochs
```

This quantity represents the total number of image inputs seen by the model during training. It does not represent the number of newly created independent image files.

### 3. Multimodal Modeling for Tasks 1 and 2

Task 1 example:

```bash
python orn_multimodal_pipeline_v1.py ^
  --image_excel image_data.xlsx ^
  --tabular_excel data_v3.1.xlsx ^
  --tabular_sheet model_full_pre_orn ^
  --image_root "C:\path\to\Image" ^
  --output_dir "./orn_multimodal_task1" ^
  --task task1 ^
  --n_splits 5 ^
  --image_pooling meanmax ^
  --use_pretrained
```

Task 2 example:

```bash
python orn_multimodal_pipeline_v1.py ^
  --image_excel image_data.xlsx ^
  --tabular_excel data_v3.1.xlsx ^
  --tabular_sheet model_full_pre_orn ^
  --image_root "C:\path\to\Image" ^
  --output_dir "./orn_multimodal_task2" ^
  --task task2 ^
  --n_splits 5 ^
  --image_pooling meanmax ^
  --use_pretrained
```

Typical outputs:

```text
figures/
tables/cv_fold_metrics.csv
tables/threshold_sweeps.csv
cv_summary.txt
cv_metrics_summary.json
```

> Keeping `--use_pretrained` is recommended. Without pretrained weights, the image feature extractor may use randomly initialized weights, making the extracted image representation unsuitable for meaningful comparison.

### 4. Tuned Task 3 Multimodal Model

Recommended PCA-based feature-level fusion:

```bash
python orn_multimodal_pipeline_task3_tuned.py ^
  --image_excel image_data.xlsx ^
  --image_sheet image_master ^
  --tabular_excel data_v3.1.xlsx ^
  --tabular_sheet model_full_pre_orn ^
  --image_root "C:\path\to\Image" ^
  --output_dir "./orn_task3_multimodal_tuned_outputs" ^
  --task3_patient_label_strategy max ^
  --image_pooling meanmax ^
  --image_pca_components 12 ^
  --multimodal_mode pca_concat ^
  --models logreg rf ^
  --n_splits 5 ^
  --use_pretrained
```

Decision-level weighted-probability fusion:

```bash
python orn_multimodal_pipeline_task3_tuned.py ^
  --image_excel image_data.xlsx ^
  --image_sheet image_master ^
  --tabular_excel data_v3.1.xlsx ^
  --tabular_sheet model_full_pre_orn ^
  --image_root "C:\path\to\Image" ^
  --output_dir "./orn_task3_weighted_prob" ^
  --task3_patient_label_strategy max ^
  --image_pooling meanmax ^
  --image_pca_components 12 ^
  --multimodal_mode weighted_prob ^
  --models logreg rf ^
  --use_pretrained
```

`task3_patient_label_strategy` options:

- `max`: uses the most severe label observed for a patient, following `non_orn < orn_normal < visible_orn`.
- `mode`: uses the most frequent image label for each patient and is retained for reproducing earlier behavior.

Task 3 is evaluated using Macro One-vs-Rest AUROC, Macro F1-score, and accuracy.

### 5. Multimodal Survival Analysis

```bash
python orn_multimodal_survival_logrank.py ^
  --data_path data_v3.1.xlsx ^
  --time_sheet ALL ^
  --feature_sheet model_full_pre_orn ^
  --image_excel image_data.xlsx ^
  --image_sheet image_master ^
  --image_root "C:\path\to\Image" ^
  --output_dir "./survival_outputs_multimodal" ^
  --image_pooling meanmax ^
  --image_pca_components 3
```

Main outputs:

```text
expanded_image_table.csv
image_level_features.csv
patient_image_features_raw.csv
patient_image_features_pca.csv
cox_clinical_only.csv
cox_multimodal.csv
proportional_hazards_test_multimodal.csv
patient_risk_scores_multimodal.csv
km_curve_multimodal.png
forest_plot_multimodal.png
summary_multimodal.txt
```

Differences between the high- and low-risk groups are evaluated using Kaplan–Meier curves and the log-rank test. The Cox model additionally reports hazard ratios and 95% confidence intervals.

### 6. Feature Ablation Analysis

```bash
python orn_feature_ablation_old.py
```

Before execution, verify the following settings in the script:

```text
input file name
worksheet name
output directory
feature-group column names
```

This script is retained for reproducing thesis tables. If the structured-data schema has changed, update the feature-removal groups accordingly to avoid inconsistent results caused by obsolete column names.

### 7. YOLO Classification

First inspect the available arguments:

```bash
python orn_yolo_cls_simple_cv.py --help
```

Confirm that `yolo11n-cls.pt` is located at a path accessible to the script, then specify the image data and output directory according to the available command-line arguments.

YOLO classification is treated as a supplementary image experiment and should not be combined directly with the results of the main patient-level ResNet/CNN pipeline as though they were the same model.

### 8. Patient-Level Risk Output and Interface

Extract:

```text
orn_risk_ui_true_pipelines_v11.zip
```

Launch the notebook:

```bash
jupyter notebook ORN輸出介面.ipynb
```

The interface is designed to display:

- patient-level risk score;
- relative risk group;
- model-performance summary;
- SHAP values or major predictive factors; and
- patient-level textual explanation.

The interface does not retrain the prediction model and must not alter the original model probabilities. All generated explanations should be grounded in previously generated model outputs.

---

## Evaluation Metrics

### Binary Tasks

- AUROC
- F1-score
- Sensitivity
- Specificity
- PR-AUC
- Brier score
- Expected Calibration Error (ECE)
- Bootstrap 95% confidence interval

### Multiclass Task

- Macro One-vs-Rest AUROC
- Macro F1-score
- Accuracy
- Confusion matrix

### Survival Analysis

- Concordance index
- Hazard ratio with 95% confidence interval
- Proportional hazards assumption test
- Kaplan–Meier curve
- Log-rank p-value

---

## Experimental Design Notes

1. **Patient-level splitting:** Images from the same patient must not be distributed across training, validation, and test sets.
2. **No test-set model selection:** Model choice, threshold selection, PCA dimensionality, and fusion weights must be determined using training and validation data only.
3. **PCA must be fitted within each training fold:** PCA must not be fitted on the full dataset before cross-validation.
4. **Online augmentation is applied only to the training set.**
5. **Prevent temporal leakage:** Include only features available at the index time or before ORN onset.
6. **Prioritize out-of-fold evaluation:** Primary thesis results should emphasize patient-level cross-validation and out-of-fold predictions rather than relying solely on a single hold-out split.
7. **SHAP is not causal evidence:** SHAP values describe feature contributions to model predictions, not causal effects on ORN development.
8. **Separate threshold purposes:** A binary classification threshold and the cutoffs used to define low-, intermediate-, and high-risk groups are different concepts.
9. **External validation remains necessary:** The reported thresholds and performance estimates must not be treated as deployment-ready clinical criteria.

---

## Troubleshooting

### Images Cannot Be Found

Check the following:

- whether `--image_root` is correct;
- whether `image_name`, `image_name_std`, or `image_id` in the Excel file matches the image filename;
- whether the file extension is `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, or `.webp`; and
- whether `missing_image_paths.csv` was generated in the output directory.

### The Number of Patients in One Class Is Smaller Than the Number of Folds

Reduce the number of folds:

```bash
--n_splits 3
```

However, if the primary thesis experiment is defined as 5-fold cross-validation, first verify the labels and inclusion criteria rather than immediately reducing the number of folds.

### Insufficient GPU Memory

Reduce the batch size:

```bash
--batch_size 8
```

or run on the CPU:

```bash
--cpu
```

### SHAP Fails to Run

```bash
pip install shap
```

If SHAP still fails, the scripts generally save an error text file while completing the remaining model evaluation steps.

### UMAP Fails to Run

```bash
pip install umap-learn
```

When UMAP is unavailable, some scripts fall back to t-SNE or PCA.

---

## Reproducibility

The main scripts commonly use the following defaults:

```text
random seed = 42
patient-level 5-fold cross-validation
image size = 224 × 224
batch size = 16
```

Results may still vary with:

- PyTorch, CUDA, and cuDNN versions;
- GPU model;
- pretrained-weight version;
- data-file content and schema version;
- patient inclusion and exclusion criteria;
- augmentation settings; and
- threshold settings.

For each experiment, retain at least:

```text
execution command
code commit hash
data version
output directory
cv_metrics_summary.json
cv_summary.txt
```

---

## Citation and Use Restrictions

This repository contains code developed for a master's thesis. Any use of the workflow, results, or interface should cite the corresponding thesis and specify the code version.

Because the study data contain patient information, the public repository must not include:

- original patient-level data;
- identifiable radiographs;
- medical record numbers or other identifiers; or
- intermediate outputs that have not been de-identified.
