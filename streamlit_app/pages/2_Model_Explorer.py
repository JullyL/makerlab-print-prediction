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


lr_metrics    = _load_json(os.path.join(MODELS_DIR, "lr_metrics.json"))
nn_metrics    = _load_json(os.path.join(MODELS_DIR, "nn_metrics.json"))
geo_cv        = _load_json(os.path.join(MODELS_DIR, "geometry_cv_results.json"))
lr_weights    = _load_pkl(os.path.join(PROC_DIR,    "lr_weights.pkl"))
nn_weights    = _load_pkl(os.path.join(MODELS_DIR,  "nn_weights.pkl"))
combined_lr_w = _load_pkl(os.path.join(MODELS_DIR,  "combined_lr_weights.pkl"))
combined_nn_w = _load_pkl(os.path.join(MODELS_DIR,  "combined_nn_weights.pkl"))
feature_cols  = _load_json(os.path.join(PROC_DIR,   "feature_cols.json")) or []

lr_prob          = st.session_state.get("lr_prob")
nn_prob          = st.session_state.get("nn_prob")
combined_lr_prob = st.session_state.get("combined_lr_prob")
combined_nn_prob = st.session_state.get("combined_nn_prob")
has_prediction   = any(p is not None for p in [lr_prob, nn_prob, combined_lr_prob, combined_nn_prob])

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    section_label("Model")
    model_view = st.radio(
        "Show results for",
        options=["Neural Network", "Logistic Regression",
                 "Combined NN * (+ Geometry)", "Combined LR * (+ Geometry)"],
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
          <div style="font-size:0.92rem;color:#6b7280;line-height:1.7;max-width:420px;margin:0 auto;">
            Navigate to <strong>Predict</strong>, upload a <code>.3mf</code> or <code>.gcode</code> file,
            configure your print settings, and click <strong>Predict ↗</strong>.<br>
            Return here to explore both slicer-only and geometry-aware model results.
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

def _pred_card(label, prob, color_active, col):
    if prob is None:
        html = (
            f'<div class="card" style="text-align:center;padding:20px 16px;">'
            f'<div style="font-size:0.75rem;color:#9ca3af;margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:1.8rem;font-weight:700;color:#d1d5db;">—</div>'
            f'<div style="font-size:0.72rem;color:#9ca3af;margin-top:4px;">not available</div>'
            f'</div>'
        )
    else:
        success = 100 - prob
        color   = "#22c55e" if success >= 60 else ("#f59e0b" if success >= 40 else "#ef4444")
        html = (
            f'<div class="card" style="text-align:center;padding:20px 16px;'
            f'border-top:3px solid {color_active};">'
            f'<div style="font-size:0.75rem;color:#9ca3af;margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:1.9rem;font-weight:700;color:{color};line-height:1;">{success:.0f}%</div>'
            f'<div style="font-size:0.72rem;color:#9ca3af;margin-top:4px;">success score</div>'
            f'</div>'
        )
    col.markdown(html, unsafe_allow_html=True)

pred_cols = st.columns(4)
_pred_card("Neural Network", nn_prob,          "#22c55e", pred_cols[0])
_pred_card("Logistic Reg.",  lr_prob,          "#f59e0b", pred_cols[1])
_pred_card("Combined NN *",  combined_nn_prob, "#6366f1", pred_cols[2])
_pred_card("Combined LR *",  combined_lr_prob, "#8b5cf6", pred_cols[3])

# Confidence bars — same order
bars = ""
if nn_prob is not None:
    bars += conf_bar_html("Neural Network", 100 - nn_prob,          "#22c55e")
if lr_prob is not None:
    bars += conf_bar_html("Logistic Reg.",  100 - lr_prob,          "#f59e0b")
if combined_nn_prob is not None:
    bars += conf_bar_html("Combined NN *",  100 - combined_nn_prob, "#6366f1")
if combined_lr_prob is not None:
    bars += conf_bar_html("Combined LR *",  100 - combined_lr_prob, "#8b5cf6")

st.markdown(
    f'<div class="card" style="padding:20px 28px;">'
    f'<div style="font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:14px;">Confidence breakdown</div>'
    f'{bars}'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:8px 0 18px 0;'>", unsafe_allow_html=True)

# ── Active model performance ─────────────────────────────────────────────────
is_combined = model_view.startswith("Combined")
is_lr       = "LR" in model_view or model_view == "Logistic Regression"

st.markdown(
    f'<div style="font-size:0.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;'
    f'color:#6b7280;margin-bottom:10px;">{model_view} — model performance</div>',
    unsafe_allow_html=True,
)

if not is_combined:
    active         = lr_metrics if model_view == "Logistic Regression" else nn_metrics
    active_weights = lr_weights  if model_view == "Logistic Regression" else nn_weights
    accuracy  = active.get("accuracy")  if active else None
    f1        = active.get("f1")        if active else None
    roc_auc   = active.get("roc_auc")   if active else None
    precision = active.get("precision") if active else None
    recall    = active.get("recall")    if active else None
else:
    # Pull CV C_combined metrics for combined models
    cv_key = "C_combined"
    cv_sub = "lr" if is_lr else "nn"
    cv_m   = (geo_cv or {}).get(cv_key, {}).get(cv_sub, {})
    active         = cv_m if cv_m else None
    active_weights = combined_lr_w if is_lr else combined_nn_w
    accuracy  = cv_m.get("accuracy")  if cv_m else None
    f1        = cv_m.get("f1")        if cv_m else None
    roc_auc   = cv_m.get("auc")       if cv_m else None
    precision = cv_m.get("precision") if cv_m else None
    recall    = cv_m.get("recall")    if cv_m else None

fmt_pct = lambda v: f"{float(v):.1%}" if v is not None else "—"
fmt_2   = lambda v: f"{float(v):.2f}" if v is not None else "—"

st.markdown(
    f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-value">{fmt_pct(accuracy)}</div>
        <div class="metric-label">Accuracy</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{fmt_2(f1)}</div>
        <div class="metric-label">F1 score</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{fmt_2(roc_auc)}</div>
        <div class="metric-label">AUC-ROC</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{fmt_2(precision)}</div>
        <div class="metric-label">Precision</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{fmt_2(recall)}</div>
        <div class="metric-label">Recall</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if is_combined:
    st.markdown(
        "<p style='font-size:0.78rem;color:#9ca3af;margin:-8px 0 14px 0;'>"
        "Metrics from leave-one-out cross-validation on 55 real-world prints (combined slicer + geometry features).</p>",
        unsafe_allow_html=True,
    )

# ── Feature importance ────────────────────────────────────────────────────────
coef_title = f"{model_view} — feature importance"
st.markdown(
    f'<div class="card"><div style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:14px;">{coef_title}</div>',
    unsafe_allow_html=True,
)

if model_view == "Logistic Regression":
    if lr_weights is not None and feature_cols:
        w = np.array(lr_weights.get("weights", []))
        if len(w) == len(feature_cols):
            abs_w = np.abs(w); max_abs = abs_w.max() or 1
            pcts  = (abs_w / max_abs * 100).astype(int)
            order = np.argsort(pcts)[::-1][:8]
            st.markdown("".join(coef_bar_html(feature_cols[i], int(pcts[i])) for i in order), unsafe_allow_html=True)
        else:
            st.warning("Weight length doesn't match feature_cols.")
    elif lr_metrics and "top_features" in lr_metrics:
        st.markdown("".join(coef_bar_html(f["name"], f["pct"]) for f in lr_metrics["top_features"][:8]), unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#9ca3af;font-size:0.88rem;'>Coefficient data not available.</p>", unsafe_allow_html=True)

elif model_view == "Neural Network":
    if nn_metrics and "top_features" in nn_metrics:
        st.markdown("".join(coef_bar_html(f["name"], f["pct"]) for f in nn_metrics["top_features"][:8]), unsafe_allow_html=True)
    elif nn_weights and feature_cols and "W1" in nn_weights:
        w1 = np.array(nn_weights["W1"])
        if w1.shape[0] == len(feature_cols):
            imp = np.mean(np.abs(w1), axis=1); max_abs = imp.max() or 1
            pcts = (imp / max_abs * 100).astype(int)
            order = np.argsort(pcts)[::-1][:8]
            st.markdown("".join(coef_bar_html(feature_cols[i], int(pcts[i])) for i in order), unsafe_allow_html=True)
        else:
            st.warning("W1 shape doesn't match feature_cols.")
    else:
        st.markdown("<p style='color:#9ca3af;font-size:0.88rem;'>Feature importance not available.</p>", unsafe_allow_html=True)

else:  # Combined models
    geo_scaler_data = _load_json(os.path.join(MODELS_DIR, "geometry_scaler_params.json"))
    all_cols = feature_cols + (geo_scaler_data["geo_cols"] if geo_scaler_data else [])
    w_dict   = combined_lr_w if is_lr else combined_nn_w

    if w_dict and all_cols:
        if is_lr:
            w = np.array(w_dict.get("weights", []))
            if len(w) == len(all_cols):
                abs_w = np.abs(w); max_abs = abs_w.max() or 1
                pcts  = (abs_w / max_abs * 100).astype(int)
                order = np.argsort(pcts)[::-1][:10]
                st.markdown("".join(coef_bar_html(all_cols[i], int(pcts[i])) for i in order), unsafe_allow_html=True)
            else:
                st.warning(f"Weight length {len(w)} doesn't match feature columns {len(all_cols)}.")
        else:
            if "W1" in w_dict:
                w1 = np.array(w_dict["W1"])
                if w1.shape[0] == len(all_cols):
                    imp = np.mean(np.abs(w1), axis=1); max_abs = imp.max() or 1
                    pcts = (imp / max_abs * 100).astype(int)
                    order = np.argsort(pcts)[::-1][:10]
                    st.markdown("".join(coef_bar_html(all_cols[i], int(pcts[i])) for i in order), unsafe_allow_html=True)
                else:
                    st.warning(f"W1 rows {w1.shape[0]} don't match feature columns {len(all_cols)}.")
    else:
        st.markdown("<p style='color:#9ca3af;font-size:0.88rem;'>Feature importance not available for this model.</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Confusion matrix (slicer-only models only) ────────────────────────────────
if not is_combined:
    active_metrics = lr_metrics if model_view == "Logistic Regression" else nn_metrics
    st.markdown(
        '<div class="card"><div style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:14px;">Confusion matrix</div>',
        unsafe_allow_html=True,
    )
    if active_metrics and "confusion_matrix" in active_metrics:
        cm = active_metrics["confusion_matrix"]
        try:
            tn, fp = int(cm[0][0]), int(cm[0][1])
            fn, tp = int(cm[1][0]), int(cm[1][1])
            st.markdown(confusion_matrix_html(tp, fp, fn, tn), unsafe_allow_html=True)
        except Exception:
            st.warning("Could not parse confusion_matrix.")
    else:
        st.markdown("<p style='color:#9ca3af;font-size:0.88rem;'>Confusion matrix not available.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

