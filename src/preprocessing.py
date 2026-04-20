"""
Responsibilities:
  - Load feature candidates from EDA handoff (feature_candidates.json)
  - Median imputation for numeric features (fit on train only)
  - Mode imputation for categorical features (fit on train only)
  - One-hot encoding of filament_type (fit on train only)
  - Min-max scaling of numeric features (fit on train only)
  - Stratified 70 / 15 / 15 split
  - Export:
      data/processed/train.npz, val.npz, test.npz
      data/processed/scaler_params.json
      data/processed/feature_cols.json
      data/processed/ohe_cols.json

scaler_params.json format (matches 02_preprocessing.ipynb):
  {
    "numeric_cols":    [...],
    "min":             {"col": float, ...},
    "max":             {"col": float, ...},
    "median_impute":   {"col": float, ...},
    "mode_impute":     {"col": str,   ...}
  }

Usage:
  from src.preprocessing import run_preprocessing
  run_preprocessing("data/raw/makerlab_dataset_5000_rows.csv")
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ── helpers ───────────────────────────────────────────────────────────────────

def load_feature_candidates(json_path="data/raw/feature_candidates.json"):
    with open(json_path) as f:
        return json.load(f)


def load_dataset(csv_path, fc):
    df = pd.read_csv(csv_path)
    keep = fc["numeric"] + fc["categorical"] + [fc["target"]]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from dataset: {missing}")
    return df[keep].copy()


# ── imputation ────────────────────────────────────────────────────────────────

def fit_imputer(train_df, numeric_cols, categorical_cols):
    """Fit imputation values on train split only.
    Median for numeric, mode for categorical."""
    imputer = {}
    for col in numeric_cols:
        imputer[col] = float(train_df[col].median())
    for col in categorical_cols:
        imputer[col] = str(train_df[col].mode()[0])
    return imputer


def apply_imputer(df, imputer):
    df = df.copy()
    for col, fill_value in imputer.items():
        df[col] = df[col].fillna(fill_value)
    return df


# ── one-hot encoding ──────────────────────────────────────────────────────────

def fit_ohe(train_df, categorical_cols):
    """Returns sorted categories per column seen in training (all 5, no drop)."""
    ohe_map = {}
    for col in categorical_cols:
        cats = sorted(train_df[col].dropna().unique().tolist())
        ohe_map[col] = cats
    return ohe_map


def apply_ohe(df, ohe_map):
    """Applies one-hot encoding — 5 binary columns per the spec (no drop-first)."""
    df = df.copy()
    ohe_cols = []
    for col, cats in ohe_map.items():
        for cat in cats:
            new_col = f"{col}_{cat}"
            df[new_col] = (df[col] == cat).astype(float)
            ohe_cols.append(new_col)
        df.drop(columns=[col], inplace=True)
    return df, ohe_cols


# ── min-max scaling ───────────────────────────────────────────────────────────

def fit_scaler(train_df, numeric_cols, imputer):
    """
    Fits min-max scaler on train split only.
    Format matches 02_preprocessing.ipynb:
    {
        "numeric_cols":   [...],
        "min":            {col: float, ...},
        "max":            {col: float, ...},
        "median_impute":  {col: float, ...},
        "mode_impute":    {col: str,   ...}
    }
    """
    return {
        "numeric_cols":  numeric_cols,
        "min":           {col: float(train_df[col].min()) for col in numeric_cols},
        "max":           {col: float(train_df[col].max()) for col in numeric_cols},
        "median_impute": {col: v for col, v in imputer.items() if isinstance(v, float)},
        "mode_impute":   {col: v for col, v in imputer.items() if isinstance(v, str)},
    }


def apply_scaler(df, scaler):
    """Applies min-max scaling. Clips to [0,1] for out-of-range val/test values."""
    df = df.copy()
    for col in scaler["numeric_cols"]:
        col_min = scaler["min"][col]
        col_max = scaler["max"][col]
        rng = col_max - col_min
        if rng == 0:
            df[col] = 0.0
        else:
            df[col] = (df[col] - col_min) / rng
            df[col] = df[col].clip(0.0, 1.0)
    return df


# ── inference helper (for Owner C / Streamlit) ────────────────────────────────

def preprocess_single(raw_dict, feature_cols, scaler, ohe_map):
    """
    Transform a single raw print-settings dict into a feature vector
    ready for model inference.

    Args:
        raw_dict     : dict with raw feature values
        feature_cols : list from feature_cols.json
        scaler       : dict from scaler_params.json
        ohe_map      : dict from ohe_cols.json

    Returns:
        np.ndarray of shape (1, n_features), dtype float32
    """
    df = pd.DataFrame([raw_dict])

    # Impute using stored values inside scaler
    for col, val in scaler["median_impute"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col, val in scaler["mode_impute"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # OHE
    df, _ = apply_ohe(df, ohe_map)

    # Scale
    df = apply_scaler(df, scaler)

    # Align to expected feature order
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_cols].values.astype(np.float32)


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing(
    csv_path,
    feature_json_path="data/raw/feature_candidates.json",
    output_dir="data/processed",
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    verbose=True,
):
    """
    Full preprocessing pipeline. Returns a dict of all artifacts for inspection.

    Steps:
      1. Load feature candidates and dataset
      2. Stratified 70/15/15 split  ← BEFORE fitting anything
      3. Fit imputer on train, apply to all splits
      4. Fit OHE on train, apply to all splits
      5. Fit min-max scaler on train (numeric cols only), apply to all splits
      6. Save .npz and JSON artifacts to output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load
    fc = load_feature_candidates(feature_json_path)
    df = load_dataset(csv_path, fc)
    numeric_cols     = fc["numeric"]
    categorical_cols = fc["categorical"]
    target_col       = fc["target"]

    if verbose:
        total = len(df)
        n_fail = df[target_col].sum()
        print(f"Dataset loaded: {df.shape[0]} rows")
        print(f"  Numeric features    : {len(numeric_cols)}")
        print(f"  Categorical features: {len(categorical_cols)}")
        print(f"  Class balance       : {n_fail}/{total} failures ({n_fail/total*100:.1f}%)")

    # 2. Stratified split — BEFORE fitting anything
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val, stratify=y_trainval, random_state=random_state
    )

    if verbose:
        total = len(y_train) + len(y_val) + len(y_test)
        print(f"\nSplit (stratified):")
        print(f"  Train : {len(X_train):4d}  ({len(X_train)/total*100:.1f}%)")
        print(f"  Val   : {len(X_val):4d}  ({len(X_val)/total*100:.1f}%)")
        print(f"  Test  : {len(X_test):4d}  ({len(X_test)/total*100:.1f}%)")

    # 3. Imputation (fit on train only)
    imputer = fit_imputer(X_train, numeric_cols, categorical_cols)
    X_train = apply_imputer(X_train, imputer)
    X_val   = apply_imputer(X_val,   imputer)
    X_test  = apply_imputer(X_test,  imputer)

    # 4. OHE (fit on train only)
    ohe_map = fit_ohe(X_train, categorical_cols)
    X_train, ohe_cols = apply_ohe(X_train, ohe_map)
    X_val,   _        = apply_ohe(X_val,   ohe_map)
    X_test,  _        = apply_ohe(X_test,  ohe_map)

    # Canonical feature order: numeric first, then OHE
    feature_cols = numeric_cols + ohe_cols

    # 5. Min-max scaling (fit on train only, numeric cols only)
    scaler = fit_scaler(X_train, numeric_cols, imputer)
    X_train = apply_scaler(X_train, scaler)
    X_val   = apply_scaler(X_val,   scaler)
    X_test  = apply_scaler(X_test,  scaler)

    # Convert to numpy
    X_train_np = X_train[feature_cols].values.astype(np.float32)
    X_val_np   = X_val[feature_cols].values.astype(np.float32)
    X_test_np  = X_test[feature_cols].values.astype(np.float32)
    y_train_np = y_train.astype(np.float32)
    y_val_np   = y_val.astype(np.float32)
    y_test_np  = y_test.astype(np.float32)

    if verbose:
        print(f"\nFeature matrix: {X_train_np.shape[1]} total features")
        print(f"  Numeric (scaled): {len(numeric_cols)}")
        print(f"  OHE columns     : {len(ohe_cols)}  {ohe_cols}")
        print(f"\nClass balance after split:")
        for name, y_arr in [("Train", y_train_np), ("Val", y_val_np), ("Test", y_test_np)]:
            print(f"  {name}: {y_arr.mean()*100:.1f}% failure")

    # 6. Save artifacts
    np.savez(os.path.join(output_dir, "train.npz"), X=X_train_np, y=y_train_np)
    np.savez(os.path.join(output_dir, "val.npz"),   X=X_val_np,   y=y_val_np)
    np.savez(os.path.join(output_dir, "test.npz"),  X=X_test_np,  y=y_test_np)

    with open(os.path.join(output_dir, "scaler_params.json"), "w") as f:
        json.dump(scaler, f, indent=2)
    with open(os.path.join(output_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(output_dir, "ohe_cols.json"), "w") as f:
        json.dump(ohe_map, f, indent=2)

    if verbose:
        print(f"\nArtifacts written to '{output_dir}/':")
        for fname in ["train.npz", "val.npz", "test.npz",
                      "scaler_params.json", "feature_cols.json", "ohe_cols.json"]:
            size = os.path.getsize(os.path.join(output_dir, fname))
            print(f"  {fname}  ({size:,} bytes)")

    return {
        "X_train": X_train_np, "y_train": y_train_np,
        "X_val":   X_val_np,   "y_val":   y_val_np,
        "X_test":  X_test_np,  "y_test":  y_test_np,
        "feature_cols": feature_cols,
        "scaler":  scaler,
        "ohe_map": ohe_map,
        "imputer": imputer,
    }
