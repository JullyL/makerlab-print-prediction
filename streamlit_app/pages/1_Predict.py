import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import pickle
import numpy as np
import streamlit as st

from streamlit_app.utils import (
    file_banner, section_label,
    risk_flag, conf_bar_html,
)
from src.preprocessing import preprocess_single
from src.parse_3mf import extract_features_from_bytes

_ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(_ROOT, "data", "raw", "processed")
MODELS_DIR    = os.path.join(_ROOT, "models")


@st.cache_resource(show_spinner=False)
def load_preprocessing():
    with open(os.path.join(PROCESSED_DIR, "scaler_params.json")) as f:
        scaler = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "ohe_cols.json")) as f:
        ohe_map = json.load(f)
    return scaler, feature_cols, ohe_map


def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def lr_sigmoid(weights, x):
    w = np.array(weights["w"])
    b = float(weights["b"])
    return float(1 / (1 + np.exp(-(x @ w + b))))


def nn_forward(weights, x):
    a = x.copy()
    n = weights.get("n_layers", 1)
    for i in range(1, n + 1):
        W = np.array(weights[f"W{i}"])
        b = np.array(weights[f"b{i}"])
        a = np.maximum(0, a @ W + b)
    W_out = np.array(weights["W_out"])
    b_out = np.array(weights["b_out"])
    return float(1 / (1 + np.exp(-(a @ W_out + b_out))))


def compute_risk_flags(material, speed, temp, cooling, layer_h):
    flags = []

    if material == "PETG" and speed > 55:
        flags.append((f"Print speed {speed} mm/s may cause stringing with PETG — consider ≤ 45 mm/s", "warn"))
    elif material in ("TPU", "Flex") and speed > 30:
        flags.append((f"Print speed {speed} mm/s is high for flexible material — consider ≤ 25 mm/s", "warn"))
    elif speed > 200:
        flags.append((f"Print speed {speed} mm/s is very aggressive — check for layer shifting", "warn"))

    if material == "PLA" and temp > 220:
        flags.append((f"Temperature {temp}°C is high for PLA — may cause stringing or blobs", "warn"))
    if material == "ABS" and cooling > 50:
        flags.append((f"Cooling fan at {cooling}% may cause warping with ABS — consider ≤ 30%", "warn"))
    if material == "PC" and temp < 270:
        flags.append((f"Temperature {temp}°C may be too low for PC — recommend ≥ 270°C", "warn"))

    if layer_h <= 0.12:
        flags.append((f"Layer height {layer_h:.2f} mm is very fine — print time will increase significantly", "warn"))
    elif 0.15 <= layer_h <= 0.25:
        flags.append((f"Layer height {layer_h:.2f} mm is well-suited to part geometry", "ok"))

    if material in ("PLA", "PETG") and cooling >= 70:
        flags.append((f"Cooling fan at {cooling}% provides good bridging performance for {material}", "ok"))

    if not flags:
        flags.append(("Settings look reasonable — no obvious risk factors detected", "ok"))

    return flags


with st.sidebar:
    section_label("Print Settings")

    material     = st.selectbox("Material",     options=["PLA", "PETG", "ABS", "TPU", "PC"], index=0)
    printer_type = st.selectbox("Printer Type", options=["Bambu X1C", "Bambu P1S", "Bambu A1", "Other FDM"], index=0)
    speed  = st.slider("Print speed (mm/s)", 10, 500, 75, 5)
    temp   = st.slider("Temperature (°C)",  150, 320, 210, 5)
    fan    = st.slider("Cooling fan%",        0, 100,   80, 5)
    layer  = st.slider("Layer height (mm)", 0.05, 0.40, 0.20, 0.01, format="%.2f")

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Predict ↗", type="primary")

    if st.button("Reset defaults", type="secondary"):
        st.session_state.pop("gcode_features", None)
        st.session_state.pop("uploaded_name", None)
        st.rerun()


try:
    scaler, feature_cols, ohe_map = load_preprocessing()
except FileNotFoundError as e:
    st.error(f"Preprocessing artifacts missing: {e}")
    st.stop()

lr_weights = load_model(os.path.join(MODELS_DIR, "lr_weights.pkl"))
nn_weights = load_model(os.path.join(MODELS_DIR, "nn_weights.pkl"))

uploaded = st.file_uploader(
    "Upload a .gcode or .gcode.3mf file",
    type=["gcode", "3mf"],
    label_visibility="collapsed",
)

if uploaded is not None:
    if st.session_state.get("uploaded_name") != uploaded.name:
        with st.spinner("Extracting G-code features…"):
            try:
                st.session_state["gcode_features"] = extract_features_from_bytes(uploaded.read())
                st.session_state["uploaded_name"]  = uploaded.name
            except Exception as exc:
                st.error(f"Feature extraction failed: {exc}")

if "uploaded_name" in st.session_state:
    file_banner(st.session_state["uploaded_name"])
    gcode_feat = st.session_state.get("gcode_features", {})

    if gcode_feat:
        def _dur(secs):
            if secs <= 0: return "—"
            h, m = divmod(int(secs), 3600)
            m, s = divmod(m, 60)
            return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"

        def _row(label, value, last=False):
            border = "" if last else "border-bottom:1px solid #f3f4f6;"
            return (
                f"<tr style='{border}'>"
                f"<td style='padding:7px 4px;color:#6b7280;'>{label}</td>"
                f"<td style='padding:7px 4px;text-align:right;font-weight:500;color:#374151;'>{value}</td>"
                f"</tr>"
            )

        def _group(name):
            return (
                f"<tr><td colspan='2' style='padding:8px 4px 4px;font-size:0.72rem;font-weight:600;"
                f"letter-spacing:.07em;text-transform:uppercase;color:#9ca3af;'>{name}</td></tr>"
            )

        f = gcode_feat
        rows_html = "".join([
            _group("Geometry"),
            _row("Total layers",             f.get("total_layers", "—")),
            _row("Max Z height",             f"{f.get('max_z_height_mm', '—')} mm"),
            _row("Bounding box",             f"{f.get('bbox_x_mm', 0):.1f} × {f.get('bbox_y_mm', 0):.1f} mm"),
            _row("Aspect ratio (H/W)",       f.get("aspect_ratio", "—")),
            _group("Toolpath"),
            _row("Total toolpath",           f"{f.get('total_toolpath_mm', 0):,.0f} mm"),
            _row("Extrusion / travel ratio", f.get("extrusion_travel_ratio", "—")),
            _row("Path variability",         f.get("path_variability", "—")),
            _row("First-layer toolpath",     f"{f.get('first_layer_toolpath_mm', 0):,.0f} mm"),
            _row("Max single-layer toolpath",f"{f.get('max_layer_toolpath_mm', 0):,.0f} mm"),
            _row("Sparse layer fraction",    f"{f.get('sparse_layer_fraction', 0):.1%}"),
            _group("Speed"),
            _row("Avg feedrate",             f"{f.get('avg_feedrate_mms', 0):.1f} mm/s"),
            _row("Feedrate variation (CV)",  f.get("feedrate_cv", "—")),
            _group("Retraction"),
            _row("Retraction count",         f"{f.get('retraction_count', 0):,}"),
            _row("Retraction density",       f"{f.get('retraction_density', 0):.2f} / layer"),
            _group("Timing"),
            _row("Est. print duration",      _dur(f.get("est_print_duration_sec", 0)), last=True),
        ])

        with st.expander("Extracted G-code features", expanded=False):
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>"
                f"{rows_html}</table>",
                unsafe_allow_html=True,
            )
else:
    gcode_feat = {}

raw_input = {
    "filament_type":                    material,
    "nozzle_temperature":               float(temp),
    "nozzle_temperature_initial_layer": float(temp),
    "layer_height":                     float(layer),
    "initial_layer_print_height":       float(layer),
    "inner_wall_speed":                 float(speed),
    "outer_wall_speed":                 float(max(speed * 0.7, 20)),
    "bridge_speed":                     float(min(speed * 0.6, 80)),
    "fan_max_speed":                    float(fan),
    "sparse_infill_density":            15.0,
    "bottom_shell_layers":              3.0,
    "top_shell_layers":                 5.0,
    "wall_loops":                       2.0,
    "retraction_length":                0.8,
    "retraction_speed":                 30.0,
    "enable_support":                   0.0,
}

lr_prob = nn_prob = None

if predict_btn or "uploaded_name" in st.session_state:
    try:
        X = preprocess_single(raw_input, feature_cols, scaler, ohe_map)
        if lr_weights is not None:
            lr_prob = lr_sigmoid(lr_weights, X[0]) * 100
        if nn_weights is not None:
            nn_prob = nn_forward(nn_weights, X[0:1]) * 100
        st.session_state["lr_prob"]   = lr_prob
        st.session_state["nn_prob"]   = nn_prob
        st.session_state["raw_input"] = raw_input
        st.session_state["X"]         = X
    except Exception as exc:
        st.warning(f"Preprocessing issue: {exc}")

if lr_prob is None:
    lr_prob = st.session_state.get("lr_prob")
if nn_prob is None:
    nn_prob = st.session_state.get("nn_prob")

tab_results, tab_sug, tab_fs = st.tabs(["Results", "Suggestions", "File summary"])

with tab_results:
    models_ready = lr_prob is not None or nn_prob is not None

    if not models_ready:
        col_score, col_conf = st.columns([1, 1.4])
        with col_score:
            st.markdown(
                """
                <div class="card" style="text-align:center;padding:28px 20px;">
                  <div style="font-size:0.8rem;color:#9ca3af;margin-bottom:8px;">Success score</div>
                  <div class="score-value" style="color:#d1d5db;">—</div>
                  <div style="margin-top:10px;">
                    <span style="background:#f3f4f6;color:#9ca3af;border-radius:20px;padding:3px 12px;font-size:0.82rem;">
                      Awaiting model weights
                    </span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_conf:
            st.markdown(
                """
                <div class="card" style="padding:28px 20px;">
                  <div style="font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:16px;">
                    Confidence breakdown
                  </div>
                  <div style="color:#9ca3af;font-size:0.88rem;">
                    ⌛ Neural Network — <em>weights not loaded yet</em><br>
                    ⌛ Logistic Reg. &nbsp; — <em>weights not loaded yet</em>
                  </div>
                  <div style="font-size:0.75rem;color:#d1d5db;margin-top:14px;">
                    Place <code>models/lr_weights.pkl</code> and <code>models/nn_weights.pkl</code>
                    to activate predictions.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        probs    = [p for p in [lr_prob, nn_prob] if p is not None]
        avg_fail = sum(probs) / len(probs)
        success  = 100 - avg_fail

        if success >= 60:
            badge = '<span class="score-badge">Likely to succeed</span>'
            color = "#22c55e"
        elif success >= 40:
            badge = '<span style="background:#fef3c7;color:#92400e;border-radius:20px;padding:3px 12px;font-size:0.82rem;">Uncertain</span>'
            color = "#f59e0b"
        else:
            badge = '<span class="fail-badge">High failure risk</span>'
            color = "#ef4444"

        col_score, col_conf = st.columns([1, 1.4])
        with col_score:
            st.markdown(
                f"""
                <div class="card" style="text-align:center;padding:28px 20px;">
                  <div style="font-size:0.8rem;color:#9ca3af;margin-bottom:8px;">Success score</div>
                  <div class="score-value" style="color:{color};">{success:.0f}%</div>
                  <div style="margin-top:10px;">{badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_conf:
            bars = ""
            if nn_prob is not None:
                bars += conf_bar_html("Neural Network", 100 - nn_prob, "#22c55e")
            if lr_prob is not None:
                bars += conf_bar_html("Logistic Reg.", 100 - lr_prob, "#f59e0b")
            st.markdown(
                f"""
                <div class="card" style="padding:28px 20px;">
                  <div style="font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:16px;">
                    Confidence breakdown
                  </div>
                  {bars}
                </div>
                """,
                unsafe_allow_html=True,
            )

    flags      = compute_risk_flags(material, speed, temp, fan, layer)
    flags_html = "".join(risk_flag(text, kind) for text, kind in flags)
    st.markdown(
        f'<div class="risk-section-label">Risk Flags</div>{flags_html}',
        unsafe_allow_html=True,
    )

with tab_sug:
    suggested_speed = min(speed, 45) if (material == "PETG" and speed > 45) else speed
    suggested_temp  = {"PLA": 215, "PETG": 235, "ABS": 250, "TPU": 220, "PC": 275}.get(material, temp)
    suggested_fan   = fan
    suggested_layer = layer

    def _sug_row(label, current, suggested, unit=""):
        changed  = abs(current - suggested) > 0.001
        val_html = (
            f'<span style="color:#2563eb;font-weight:700;">{suggested}{unit}</span>'
            if changed else
            f'<span class="sug-ok">{suggested}{unit} ✓</span>'
        )
        return f"<tr><td>{label}</td><td>{val_html}</td></tr>"

    sug_table = (
        _sug_row("Print Speed",   speed, suggested_speed, " mm/s")
        + _sug_row("Temperature", temp,  suggested_temp)
        + _sug_row("Cooling fan", fan,   suggested_fan, "%")
        + _sug_row("Layer height",layer, suggested_layer, " mm")
    )

    col_sug, col_alt = st.columns([1.1, 1])
    with col_sug:
        st.markdown(
            f"""
            <div class="card">
              <div style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:14px;">
                Recommended settings
              </div>
              <table class="sug-table">{sug_table}</table>
              <a class="apply-link" style="display:inline-block;margin-top:14px;">Apply suggestions ↗</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_alt:
        if gcode_feat:
            flat = gcode_feat.get("max_z_height_mm", 0) < 10 and gcode_feat.get("total_layers", 0) < 60
            suggestion_text = (
                "Flat geometry detected. This part may be better suited for laser cutting, "
                "which offers higher precision for planar designs."
                if flat else
                "Complex 3D geometry detected. FDM printing is well-suited to this part shape."
            )
        else:
            suggestion_text = "Upload a .gcode or .3mf file to receive geometry-specific fabrication suggestions."

        st.markdown(
            f"""
            <div class="card">
              <div style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:10px;">
                Alternative method
              </div>
              <p style="font-size:0.88rem;color:#4b5563;line-height:1.6;margin:0 0 12px 0;">
                {suggestion_text}
              </p>
              <a class="apply-link">Learn more ↗</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_fs:
    if not gcode_feat:
        st.markdown('<div class="card" style="color:#9ca3af;text-align:center;padding:40px;">', unsafe_allow_html=True)
        st.info("Upload a .gcode or .gcode.3mf file to see extracted geometry features here.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        def _dur(secs):
            if secs <= 0: return "—"
            h, m = divmod(int(secs), 3600)
            m, s = divmod(m, 60)
            return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"

        rows = [
            ("File",                      st.session_state.get("uploaded_name", "—")),
            ("Total Layers",              str(gcode_feat.get("total_layers", "—"))),
            ("Max Z height (mm)",         str(gcode_feat.get("max_z_height_mm", "—"))),
            ("Total toolpath length (mm)",f"{gcode_feat.get('total_toolpath_mm', 0):,.0f}"),
            ("Extrusion-to-travel ratio", str(gcode_feat.get("extrusion_travel_ratio", "—"))),
            ("Est. print duration",       _dur(gcode_feat.get("est_print_duration_sec", 0))),
        ]
        rows_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
        st.markdown(
            f'<div class="card"><table class="fs-table">{rows_html}</table></div>',
            unsafe_allow_html=True,
        )
