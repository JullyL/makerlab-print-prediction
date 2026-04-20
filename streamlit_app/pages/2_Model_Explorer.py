import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import pickle
import numpy as np
import streamlit as st

from streamlit_app.utils import (
    section_label,
    confusion_matrix_html, coef_bar_html,
)

_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(_ROOT, "models")
PROC_DIR   = os.path.join(_ROOT, "data", "raw", "processed")


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _load_pkl(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


lr_metrics   = _load_json(os.path.join(MODELS_DIR, "lr_metrics.json"))
nn_metrics   = _load_json(os.path.join(MODELS_DIR, "nn_metrics.json"))
lr_weights   = _load_pkl(os.path.join(MODELS_DIR, "lr_weights.pkl"))
feature_cols = _load_json(os.path.join(PROC_DIR, "feature_cols.json")) or []

with st.sidebar:
    section_label("Model")
    model_view = st.radio(
        "Show results for",
        options=["Logistic Regression", "Neural Network"],
        label_visibility="collapsed",
    )

active         = lr_metrics if model_view == "Logistic Regression" else nn_metrics
active_weights = lr_weights  if model_view == "Logistic Regression" else None

accuracy = active.get("accuracy") if active else None
f1       = active.get("f1")       if active else None
roc_auc  = active.get("roc_auc")  if active else None

acc_str = f"{float(accuracy):.1%}" if accuracy is not None else "—"
f1_str  = f"{float(f1):.2f}"       if f1       is not None else "—"
auc_str = f"{float(roc_auc):.2f}"  if roc_auc  is not None else "—"

st.markdown(
    f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-value">{acc_str}</div>
        <div class="metric-label">Test accuracy</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{f1_str}</div>
        <div class="metric-label">F1 score</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{auc_str}</div>
        <div class="metric-label">AUC-ROC</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if active is None:
    st.markdown(
        f"""
        <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;
                    padding:10px 16px;font-size:0.87rem;color:#92400e;margin-bottom:16px;">
          ⌛ &nbsp;Metrics not yet available for {model_view}.
          Drop <code>models/lr_metrics.json</code> / <code>nn_metrics.json</code> to populate.
        </div>
        """,
        unsafe_allow_html=True,
    )

coef_title = "Logistic regression coefficient" if model_view == "Logistic Regression" else "Neural Network — feature importance"
st.markdown(
    f'<div class="card"><div style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:14px;">{coef_title}</div>',
    unsafe_allow_html=True,
)

if model_view == "Logistic Regression":
    if active_weights is not None and feature_cols:
        w = np.array(active_weights.get("w", []))
        if len(w) == len(feature_cols):
            abs_w   = np.abs(w)
            max_abs = abs_w.max() if abs_w.max() > 0 else 1
            pcts    = (abs_w / max_abs * 100).astype(int)
            order   = np.argsort(pcts)[::-1][:8]
            bars    = "".join(coef_bar_html(feature_cols[i], int(pcts[i])) for i in order)
            st.markdown(bars, unsafe_allow_html=True)
        else:
            st.warning("Weight vector length doesn't match feature_cols. Check lr_weights.pkl.")
    elif lr_metrics and "top_features" in lr_metrics:
        bars = "".join(coef_bar_html(f["name"], f["pct"]) for f in lr_metrics["top_features"][:8])
        st.markdown(bars, unsafe_allow_html=True)
    else:
        demo = [("Print speed", 85), ("Layer height", 72), ("Material", 61), ("Infill %", 48), ("Supports", 34)]
        st.markdown("".join(coef_bar_html(n, p) for n, p in demo), unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.75rem;color:#d1d5db;margin-top:8px;'>"
            "⌛ Showing demo values — drop <code>models/lr_weights.pkl</code> for real coefficients.</p>",
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        "<p style='color:#9ca3af;font-size:0.88rem;'>"
        "Feature importance for neural networks is not directly interpretable via coefficients.<br>"
        "Use SHAP or gradient-based attribution.</p>",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="card"><div style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:14px;">Confusion matrix</div>',
    unsafe_allow_html=True,
)

if active and "confusion_matrix" in active:
    cm = active["confusion_matrix"]
    try:
        tn, fp = int(cm[0][0]), int(cm[0][1])
        fn, tp = int(cm[1][0]), int(cm[1][1])
        st.markdown(confusion_matrix_html(tp, fp, fn, tn), unsafe_allow_html=True)
    except Exception:
        st.warning("Could not parse confusion_matrix. Expected format: [[TN, FP], [FN, TP]]")
else:
    st.markdown(confusion_matrix_html(142, 11, 8, 89), unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.75rem;color:#d1d5db;margin-top:10px;'>"
        "⌛ Showing demo values — add <code>confusion_matrix</code> key to metrics JSON for real data.</p>",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
