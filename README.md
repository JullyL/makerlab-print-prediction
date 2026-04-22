# MakerLAB Print Prediction

Machine learning project for predicting 3D print failure risk before print execution in the Cornell Tech MakerLAB workflow.

## Project overview

The project combines:
- synthetic tabular print-setting data,
- real print file feature extraction from `.gcode.3mf`,
- binary classification models (Logistic Regression and Neural Network),
- and a Streamlit interface for pre-print risk feedback.

Core goal: detect likely failures early and provide actionable setup guidance before a print starts.

## Repository structure

- `data/raw/` - source datasets and EDA handoff JSON
- `notebooks/` - EDA, preprocessing, model training notebooks
- `src/` - reusable modeling and preprocessing modules
- `streamlit_app/` - deployment UI shell and pages
- `scripts/setup_data.sh` - local setup helper for large dataset copy

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run EDA notebook:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Running the Streamlit app

Make sure dependencies are installed (see Getting started above), then run from the repo root:

```bash
streamlit run streamlit_app/app.py
```

The app will open at `http://localhost:8501` by default. It expects the preprocessing artifacts (`scaler_params.json`, `feature_cols.json`, `ohe_cols.json`) to be present in `data/raw/processed/`. Model weights (`lr_weights.pkl`, `nn_weights.pkl`) in `models/` are optional — the app shows placeholder cards until they are available.

To extract G-code features from a folder of `.gcode.3mf` files and write `real_prints_features.csv`:

```bash
python -m src.parse_3mf --input_dir ./3mf_files --output real_prints_features.csv
```

## Large dataset setup (no Git LFS)

`data/raw/makerlab_dataset_5000_rows.csv` is intentionally not tracked in git because it exceeds GitHub's 100MB file limit.

**Download the file from Google Drive:**
[https://drive.google.com/file/d/1g7kyCBtcptaYtGbFlp8K6YvXFuS9-7Ph/view](https://drive.google.com/file/d/1g7kyCBtcptaYtGbFlp8K6YvXFuS9-7Ph/view)

**Setup steps:**

1. Open the link above and click **Download** (top-right menu or the download icon).
2. Move the downloaded file into the `data/raw/` directory of this repo and confirm it is named exactly `makerlab_dataset_5000_rows.csv`:

```bash
mv ~/Downloads/makerlab_dataset_5000_rows.csv data/raw/makerlab_dataset_5000_rows.csv
```

3. Verify the file is in place:

```bash
ls data/raw/makerlab_dataset_5000_rows.csv
```

The notebooks and preprocessing scripts expect the file at that exact path.

## Group task assignment (revised)

This project is split across 6 members, each owning a distinct area of the pipeline.

- **Jully Li** owns exploratory data analysis: comparing synthetic datasets at different scales, identifying data quality issues, reviewing class balance, and producing the EDA plots and feature candidates that inform preprocessing decisions.
- **Sophie Su** owns data preprocessing: cleaning, imputation, one-hot encoding, min-max scaling (fit on train only), stratified 70/15/15 splitting, and exporting all shared artifacts (`train.npz`, `val.npz`, `test.npz`, `scaler_params.json`, `feature_cols.json`).
- **Weicong (Wendy) Hong** owns the parser and Streamlit app: extracting real-print features from `.gcode.3mf` files, building the full-stack UI, and wiring preprocessing artifacts and model weights into the inference flow.
- **Bryant Jiang** owns real-print labeling: defining the labeling rubric, annotating timelapse outcomes, and producing `real_eval.csv` by joining extracted features with ground-truth labels.
- **Hannah Liang** owns the logistic regression model: implementing LR from scratch in NumPy, applying class weighting, tuning on validation, and reporting final test metrics.
- **Carina Hu** owns the neural network and final evaluation: implementing the NN from scratch in NumPy, tuning on validation, running the LR-vs-NN comparison on both synthetic and real data, and producing the final evaluation outputs.

### Key dependencies and rules

- Sophie must deliver preprocessing artifacts before Hannah and Carina run final training and reporting.
- Tune only on `val.npz`; do not touch `test.npz` until hyperparameters are finalized.
- `scaler_params.json` and `feature_cols.json` are shared interfaces and must stay consistent between training and inference.
- `real_eval.csv` is for transfer evaluation only (not training).
- Hannah and Carina should use the same class-weighting scheme for fair comparison.

### Shared interface files

- `train.npz`, `val.npz`, `test.npz` (Sophie -> Hannah / Carina)
- `scaler_params.json`, `feature_cols.json` (Sophie -> Wendy / Hannah / Carina)
- `ohe_cols.json` (Sophie -> Wendy)
- `real_prints_features.csv` (Wendy -> Bryant / Carina)
- `labels.csv` (Bryant -> Bryant join step)
- `real_eval.csv` (Bryant -> Carina)
- `lr_weights.pkl` (Hannah -> Wendy)
- `nn_weights.pkl` (Carina -> Wendy)
