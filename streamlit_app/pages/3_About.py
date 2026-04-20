import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

st.markdown(
    """
    <div class="card">
      <h2 style="font-size:1.4rem;font-weight:700;color:#111827;margin-bottom:6px;">
        About this project
      </h2>
      <p style="font-size:0.9rem;color:#6b7280;margin-bottom:20px;">
        Cornell Tech MakerLAB · PAML Final Project
      </p>

      <h3 style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:8px;">Overview</h3>
      <p style="font-size:0.92rem;color:#4b5563;line-height:1.7;margin-bottom:20px;">
        3D Print Verifier predicts whether a print job will succeed or fail before
        fabrication begins. It uses machine learning models trained on a synthetic dataset
        of 5,000 simulated Bambu Lab print configurations from the Cornell MakerLAB.
      </p>

      <h3 style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:10px;">Dataset</h3>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;margin-bottom:20px;">
      <tr style="border-bottom:1px solid #e5e7eb;">
        <td style="padding:9px 4px;color:#6b7280;">Dataset size</td>
        <td style="padding:9px 4px;font-weight:600;color:#111827;">5,000 simulated print jobs</td>
      </tr>
      <tr style="border-bottom:1px solid #e5e7eb;">
        <td style="padding:9px 4px;color:#6b7280;">Class balance</td>
        <td style="padding:9px 4px;font-weight:600;color:#111827;">64.5% success · 35.5% failure</td>
      </tr>
      <tr style="border-bottom:1px solid #e5e7eb;">
        <td style="padding:9px 4px;color:#6b7280;">Features</td>
        <td style="padding:9px 4px;font-weight:600;color:#111827;">15 numeric slicer settings + filament type (one-hot)</td>
      </tr>
      <tr>
        <td style="padding:9px 4px;color:#6b7280;">Train / Val / Test split</td>
        <td style="padding:9px 4px;font-weight:600;color:#111827;">70% / 15% / 15% (stratified)</td>
      </tr>
    </table>

    <h3 style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:10px;">Models</h3>
    <p style="font-size:0.9rem;color:#4b5563;line-height:1.7;margin-bottom:8px;">
      Both models are implemented from scratch using NumPy only — no scikit-learn, no PyTorch.
      They use weighted binary cross-entropy loss to handle the class imbalance.
    </p>
    <ul style="font-size:0.9rem;color:#4b5563;line-height:1.8;padding-left:18px;margin-bottom:20px;">
      <li><strong>Logistic Regression</strong> — L2 regularisation, early stopping, feature coefficient analysis</li>
      <li><strong>Neural Network</strong> — Feedforward with ReLU hidden layer(s) and sigmoid output, tuned on validation set</li>
    </ul>

    <h3 style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:10px;">G-code Analysis</h3>
    <p style="font-size:0.92rem;color:#4b5563;line-height:1.7;margin-bottom:8px;">
      When a <code>.gcode</code> or <code>.gcode.3mf</code> file is uploaded, 16 features
      are extracted directly from the toolpath:
    </p>
    <ul style="font-size:0.9rem;color:#4b5563;line-height:1.8;padding-left:18px;margin-bottom:20px;">
      <li>Total layer count</li>
      <li>Maximum Z height</li>
      <li>Total toolpath length</li>
      <li>Extrusion-to-travel ratio</li>
      <li>Estimated print duration</li>
      <li>Path variability (std dev of per-layer move lengths)</li>
    </ul>

    <h3 style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:10px;">Evaluation</h3>
    <p style="font-size:0.92rem;color:#4b5563;line-height:1.7;">
      In addition to the held-out synthetic test set, both models are evaluated on a set of
      real MakerLAB print jobs with manually reviewed labels. The gap between synthetic test
      performance and real-print transfer accuracy is the key result of this project.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)
