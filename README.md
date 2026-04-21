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

To set it up locally:

```bash
bash scripts/setup_data.sh /path/to/ChallengeData
```

Example on macOS:

```bash
bash scripts/setup_data.sh "/Users/jullyli/Downloads/ChallengeData"
```

## Group task assignment (revised)

This project is split across 6 owners over 4 weeks, with two independent tracks converging at model training.

- Jully Li (EDA lead): compare 100/1000/5000-row synthetic datasets, identify zero-variance columns, review class balance, generate EDA plots, and hand off cleaned feature candidates to Owner B by end of week 1.
- Sophie Su (critical path, preprocessing lead): imputation, one-hot encoding, min-max scaling on train split only, stratified 70/15/15 split, and export of `train.npz`, `val.npz`, `test.npz`, `scaler_params.json`, and `feature_cols.json`.
- Weicong (Wendy) Hong (parser + app implementation): extract real-print features from `.gcode.3mf` files into `real_prints_features.csv`, build full-stack Streamlit app (implement UI features, wire preprocessing artifacts for inference, and integrate model weights into Streamlit).
- Bryant Jiang (labeling lead): define labeling rubric, label timelapse outcomes, produce `labels.csv`, and join with C's features to create `real_eval.csv`.
- Hannah Liang (logistic regression lead): implement LR from scratch (NumPy), use weighted loss, tune on validation, report metrics on test, and integrate LR weights into Streamlit.
- Carina Hu (neural network + final eval lead): implement NN from scratch (NumPy), tune on validation, run final synthetic + real-data evaluation, produce LR-vs-NN comparison outputs, and integrate NN weights into Streamlit.

### Key dependencies and rules

- B must deliver preprocessing artifacts before E/F run final training and reporting.
- Tune only on `val.npz`; do not touch `test.npz` until hyperparameters are finalized.
- `scaler_params.json` and `feature_cols.json` are shared interfaces and must stay consistent between training and inference.
- `real_eval.csv` is for transfer evaluation only (not training).
- E and F should use the same class-weighting scheme for fair comparison.

### Shared interface files

- `train.npz`, `val.npz`, `test.npz` (B -> E/F)
- `scaler_params.json`, `feature_cols.json` (B -> C/E/F)
- `ohe_cols.json` (B -> C)
- `real_prints_features.csv` (C -> D/F)
- `labels.csv` (D -> D join step)
- `real_eval.csv` (D -> F)
- `lr_weights.pkl` (E -> C)
- `nn_weights.pkl` (F -> C)
