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

Make sure dependencies are installed (see **Getting started** above), then run from the repo root:

```bash
streamlit run streamlit_app/app.py
```

The app opens at `http://localhost:8501`. It requires the preprocessing artifacts in `data/raw/processed/` (`scaler_params.json`, `feature_cols.json`, `ohe_cols.json`) and the trained model weights in `models/` (`lr_weights.pkl`, `nn_weights.pkl`).

### Quick-start with the example file

An example print file is included so you can try the full prediction flow immediately without needing your own G-code:

```
examples/knob.gcode.3mf   ← 1.3 MB Bambu Studio export of a parametric knob
```

**Step-by-step walkthrough:**

1. Start the app with `streamlit run streamlit_app/app.py`.
2. Click **Predict** in the left sidebar.
3. Drag **`examples/knob.gcode.3mf`** onto the upload zone (or click **Browse files** and select it). The app extracts G-code geometry features automatically.
4. Adjust the print settings in the sidebar (Material, speed, temperature, fan, layer height) to match your intended slicer configuration — or leave the defaults (PLA, 75 mm/s, 210 °C, 80 % fan, 0.20 mm layer).
5. Click the **Predict ↗** button (it activates once a file is loaded). Both the Logistic Regression and Neural Network models run instantly.
6. Review the **Results** tab for the ensemble success score and per-model confidence breakdown, then check the **Suggestions** and **File summary** tabs.
7. Navigate to **Model Explorer** to see a detailed breakdown of each model's performance metrics, feature importance, and confusion matrix for your prediction run.
8. To start a new analysis, click **✕ Clear** next to the file banner or **Reset defaults** in the sidebar.

### Batch G-code feature extraction

To extract features from a folder of `.gcode.3mf` files and write a CSV:

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
