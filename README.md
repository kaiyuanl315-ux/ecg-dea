# ECG-DEA Main Cohort Public Release

This folder is the final SCI submission-ready public package for the **main cohort minimal reproducible workflow**.

## Scope
- Main-cohort data processing
- Main model training
- AUROC-oriented evaluation
- External validation is intentionally excluded

## Included Files
- `model/` (pretrained model folder)
- `src/train_multimodal_main_cohort_model.py`
- `src/prepare_auroc_evaluation_inputs.py`
- `notebooks/note/build_main_cohort_dataset.ipynb`
- `notebooks/note/evaluate_main_cohort_auroc.ipynb`

## Required Inputs
- MIMIC-IV core tables (including `hosp/admissions.csv`)
- MIMIC-IV-ED tables
- MIMIC-IV-ECG waveform dataset
- Python environment with dependencies used by `tsai`, `torch`, `numpy`, `pandas`, `scikit-learn`, and plotting packages
- Install dependencies: `pip install -r requirements.txt`

## Reproducibility Steps
1. Run `notebooks/note/build_main_cohort_dataset.ipynb` to produce:
   - `ecg_multi.csv`
   - `multi_ecg_x.npz`
2. Train the model:
   - `python src/train_multimodal_main_cohort_model.py --csv /path/to/ecg_multi.csv --npz /path/to/multi_ecg_x.npz --out_dir data`
3. Prepare evaluation aliases and run AUROC notebook:
   - `python src/prepare_auroc_evaluation_inputs.py --pred_dir data --table_csv /path/to/ecg_multi.csv`
   - Open `notebooks/note/evaluate_main_cohort_auroc.ipynb`

## Naming Alignment
- `ecg_multi.csv` -> `ecg_muti.csv` (legacy compatibility)
- `proba_train.csv` -> `proba_train_72.csv`
- `proba_val.csv` -> `proba_val_72.csv`
- `proba_test.csv` -> `proba_test_72.csv`


## Minimal Run Example
```bash
# 1) Build the main cohort dataset in notebook:
# notebooks/note/build_main_cohort_dataset.ipynb

# 2) Train multimodal model
python src/train_multimodal_main_cohort_model.py \
  --csv data/ecg_multi.csv \
  --npz data/multi_ecg_x.npz \
  --out_dir data

# 3) Prepare AUROC notebook input aliases
python src/prepare_auroc_evaluation_inputs.py --pred_dir data --table_csv data/ecg_multi.csv

# 4) Run AUROC evaluation notebook:
# notebooks/note/evaluate_main_cohort_auroc.ipynb
```
