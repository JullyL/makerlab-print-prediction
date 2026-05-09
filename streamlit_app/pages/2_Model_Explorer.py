import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import pickle
import numpy as np
import streamlit as st

from streamlit_app.utils import (
    section_label,
    confusion_matrix_html, coef_bar_html, conf_bar_html,
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
lr_weights   = _load_pkl(os.path.join(PROC_DIR, "lr_weights.pkl"))
nn_weights   = _load_pkl(os.path.join(MODELS_DIR, "nn_weights.pkl"))
feature_cols = _load_json(os.path.join(PROC_DIR, "feature_cols.json")) or []

lr_prob = st.session_state.get("lr_prob")
nn_prob = st.session_state.get("nn_prob")
has_prediction = lr_prob is not None or nn_prob is not None

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    section_label("Model")
    model_view = st.radio(
        "Show results for",
        options=["Logistic Regression", "Neural Network"],
        label_visibility="collapsed",
        disabled=not has_prediction,
    )

# ── Empty state ─────────────────────────────────────────────────────────────
if not has_prediction:
    st.markdown(
        """
        <div class="card" style="text-align:center;padding:56px 28px;">
          <div style="font-size:2.2rem;margin-bottom:16px;">📊</div>
          <div style="font-size:1.1rem;font-weight:600;color:#111827;margin-bottom:10px;">
            No prediction run yet
          </div>
          <div style="font-size:0.92rem;color:#6b7280;line-height:1.7;max-width:400px;margin:0 auto;">
            Navigate to the <strong>Predict</strong> page, configure your print settings,
            and click <strong>Predict ↗</strong>.<br>
            Return here to explore a detailed breakdown of both models' results for your run.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Your prediction ──────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:0.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;'
    'color:#6b7280;margin-bottom:10px;">Your prediction</div>',
    unsafe_allow_html=True,
)

pred_cols = st.columns(2)

def _pred_card(label, prob, col):
    if prob is None:
        html = (
            f'<div class="card" style="text-align:center;padding:24px 20px;">'
            f'<div style="font-size:0.8rem;color:#9ca3af;margin-bottom:6px;">{label}</div>'
            f'<div style="font-size:2rem;font-weight:700;color:#d1d5db;">—</div>'
            f'<div style="font-size:0.78rem;color:#9ca3af;margin-top:6px;">Model not loaded</div>'
            f'</div>'
        )
    else:
        success = 100 - prob
        if success >= 60:
            color, badge = "#22c55e", "Likely to succeed"
            badge_style = "background:#f0fdf4;color:#166534;"
        elif success >= 40:
            color, badge = "#f59e0b", "Uncertain"
            badge_style = "background:#fffbeb;color:#92400e;"
        else:
            color, badge = "#ef4444", "High failure risk"
            badge_style = "background:#fff1f2;color:#e11d48;"
        html = (
            f'<div class="card" style="text-align:center;padding:24px 20px;">'
            f'<div style="font-size:0.8rem;color:#9ca3af;margin-bottom:6px;">{label}</div>'
            f'<div style="font-size:2.4rem;font-weight:700;color:{color};line-height:1;">{success:.0f}%</div>'
            f'<div style="font-size:0.78rem;margin-top:8px;">'
            f'<span style="{badge_style}border-radius:20px;padding:3px 10px;">{badge}</span>'
            f'</div>'
            f'<div style="font-size:0.75rem;color:#9ca3af;margin-top:8px;">success score</div>'
            f'</div>'
        )
    col.markdown(html, unsafe_allow_html=True)

_pred_card("Logistic Regression", lr_prob, pred_cols[0])
_pred_card("Neural Network",      nn_prob, pred_cols[1])

# confidence bars (combined view)
bars = ""
if nn_prob is not None:
    bars += conf_bar_html("Neural Network", 100 - nn_prob, "#22c55e")
if lr_prob is not None:
    bars += conf_bar_html("Logistic Reg.",  100 - lr_prob, "#f59e0b")

st.markdown(
    f'<div class="card" style="padding:20px 28px;">'
    f'<div style="font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:14px;">Confidence breakdown</div>'
    f'{bars}'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:8px 0 18px 0;'>", unsafe_allow_html=True)

# ── Model performance (active model) ────────────────────────────────────────
active         = lr_metrics if model_view == "Logistic Regression" else nn_metrics
active_weights = lr_weights if model_view == "Logistic Regression" else nn_weights

st.markdown(
    f'<div style="font-size:0.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;'
    f'color:#6b7280;margin-bottom:10px;">{model_view} — model performance</div>',
    unsafe_allow_html=True,
)

accuracy = active.get("accuracy") if active else None
f1       = active.get("f1")       if active else None
roc_auc  = active.get("roc_auc")  if active else None
precision = active.get("precision") if active else None
recall    = active.get("recall")    if active else None

acc_str  = f"{float(accuracy):.1%}"  if accuracy  is not None else "—"
f1_str   = f"{float(f1):.2f}"        if f1        is not None else "—"
auc_str  = f"{float(roc_auc):.2f}"   if roc_auc   is not None else "—"
prec_str = f"{float(precision):.2f}" if precision is not None else "—"
rec_str  = f"{float(recall):.2f}"    if recall    is not None else "—"

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
      <div class="metric-card">
        <div class="metric-value">{prec_str}</div>
        <div class="metric-label">Precision</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{rec_str}</div>
        <div class="metric-label">Recall</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Feature importance / coefficients ───────────────────────────────────────
coef_title = (
    "Logistic regression coefficients"
    if model_view == "Logistic Regression"
    else "Neural Network — feature importance"
)
st.markdown(
    f'<div class="card"><div style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:14px;">{coef_title}</div>',
    unsafe_allow_html=True,
)

if model_view == "Logistic Regression":
    if active_weights is not None and feature_cols:
        w = np.array(active_weights.get("weights", []))
        if len(w) == len(feature_cols):
            abs_w   = np.abs(w)
            max_abs = abs_w.max() if abs_w.max() > 0 else 1
            pcts    = (abs_w / max_abs * 100).astype(int)
            order   = np.argsort(pcts)[::-1][:8]
            bars    = "".join(coef_bar_html(feature_cols[i], int(pcts[i])) for i in order)
            st.markdown(bars, unsafe_allow_html=True)
        else:
            st.warning("Weight vector length doesn't match feature_cols.")
    elif lr_metrics and "top_features" in lr_metrics:
        bars = "".join(coef_bar_html(f["name"], f["pct"]) for f in lr_metrics["top_features"][:8])
        st.markdown(bars, unsafe_allow_html=True)
    else:
        st.markdown(
            "<p style='color:#9ca3af;font-size:0.88rem;'>Coefficient data not available.</p>",
            unsafe_allow_html=True,
        )
else:
    if nn_metrics and "top_features" in nn_metrics:
        bars = "".join(coef_bar_html(f["name"], f["pct"]) for f in nn_metrics["top_features"][:8])
        st.markdown(bars, unsafe_allow_html=True)
    elif active_weights is not None and feature_cols and "W1" in active_weights:
        w1 = np.array(active_weights["W1"])
        if w1.shape[0] == len(feature_cols):
            importance = np.mean(np.abs(w1), axis=1)
            max_abs    = importance.max() if importance.max() > 0 else 1
            pcts       = (importance / max_abs * 100).astype(int)
            order      = np.argsort(pcts)[::-1][:8]
            bars       = "".join(coef_bar_html(feature_cols[i], int(pcts[i])) for i in order)
            st.markdown(bars, unsafe_allow_html=True)
        else:
            st.warning("First-layer weight shape doesn't match feature_cols.")
    else:
        st.markdown(
            "<p style='color:#9ca3af;font-size:0.88rem;'>Feature importance not available.</p>",
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)

# ── Confusion matrix ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="card"><div style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:14px;">Confusion matrix — test set</div>',
    unsafe_allow_html=True,
)

if active and "confusion_matrix" in active:
    cm = active["confusion_matrix"]
    try:
        tn, fp = int(cm[0][0]), int(cm[0][1])
        fn, tp = int(cm[1][0]), int(cm[1][1])
        st.markdown(confusion_matrix_html(tp, fp, fn, tn), unsafe_allow_html=True)
    except Exception:
        st.warning("Could not parse confusion_matrix.")
else:
    st.markdown(
        "<p style='color:#9ca3af;font-size:0.88rem;'>Confusion matrix not available.</p>",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
