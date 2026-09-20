# ECG based early mortality risk analysis

This repository provides a research workflow for ECG-based mortality risk analysis. The original release contains main-cohort processing, multimodal model training, AUROC evaluation and ten pretrained ensemble checkpoints.

The September 2026 supplement adds source snapshots and aggregate results for calibration, threshold and alert analyses, mortality comparators, clinical phenotypes, subsequent care, focused exploratory analyses and integrated gradients. Start with [Reproducibility and source scope](REPRODUCIBILITY.md), [aggregate results](aggregate_results/README.md), and [data access](DATA_ACCESS.md).

The added scripts retain historical research dependencies and require restricted inputs and private provenance/QC records. They are inspectable scientific source snapshots, not a standalone or newly validated full-study pipeline. Some statistical scripts use prepared external-cohort predictions; external-cohort data processing and waveform-to-score reconstruction remain outside this public release.

`SOURCE_MANIFEST.json` records source hashes and transformations. `requirements-supplement.txt` lists extra direct Python dependencies. The original scripts and ten checkpoints remain unchanged. See [license and attribution status](THIRD_PARTY_NOTICES.md) before reuse.

The manuscript's additional analyses identify MIMIC-IV v3.0, MIMIC-IV-ED v2.2 and MIMIC-IV-ECG v1.0. The MIMIC-IV v2.2 reference retained below is part of the historical release; it is not the version specification for the current supplemental analyses.

## Original main-cohort workflow
- Main-cohort data processing
- Main model training
- AUROC-oriented evaluation
- External-cohort data preparation and waveform-to-score reconstruction are excluded from the original public workflow.

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
## Original release citations
If you use this workflow, please cite the MIMIC-IV-ECG dataset:

```bibtex
@article{PhysioNet-mimic-iv-ecg-1.0,
  author = {Gow, Brian and Pollard, Tom and Nathanson, Larry A and Johnson, Alistair and Moody, Benjamin and Fernandes, Chrystinne and Greenbaum, Nathaniel and Waks, Jonathan W and Eslami, Parastou and Carbonati, Tanner and Chaudhari, Ashish and Herbst, Elizabeth and Moukheiber, Dana and Berkowitz, Seth and Mark, Roger and Horng, Steven},
  title = {{MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset}},
  journal = {{PhysioNet}},
  year = {2023},
  month = sep,
  note = {Version 1.0},
  doi = {10.13026/4nqg-sb35},
  url = {https://doi.org/10.13026/4nqg-sb35}
}

@article{PhysioNet-mimiciv-2.2,
  author = {Johnson, Alistair and Bulgarelli, Lucas and Pollard, Tom and Horng, Steven and Celi, Leo Anthony and Mark, Roger},
  title = {{MIMIC-IV}},
  journal = {{PhysioNet}},
  year = {2023},
  month = jan,
  note = {Version 2.2},
  doi = {10.13026/6mm1-ek67},
  url = {https://doi.org/10.13026/6mm1-ek67}
}


@article{10.1093/eurheartj/ehaf254,
    author = {Büscher, Antonius and Plagwitz, Lucas and Yildirim, Kemal and Brix, Tobias J and Neuhaus, Philipp and Bickmann, Lucas and Menke, Amélie F and van Almsick, Vincent F and Pavenstädt, Hermann and Kümpers, Philipp and Heider, Dominik and Varghese, Julian and Eckardt, Lars},
    title = {Deep Learning Electrocardiogram Model for Risk Stratification of Coronary Revascularization Need in the Emergency Department},
    journal = {European Heart Journal},
    pages = {ehaf254},
    year = {2025},
    month = {03},
    issn = {0195-668X},
    doi = {10.1093/eurheartj/ehaf254},
    url = {https://doi.org/10.1093/eurheartj/ehaf254},
    eprint = {https://academic.oup.com/eurheartj/advance-article-pdf/doi/10.1093/eurheartj/ehaf254/62788722/ehaf254.pdf},
}
```

