"""
Home page content — rendered by app.py via st.navigation()
"""

import streamlit as st

st.markdown(
    """
    <div class="card">
      <h2 style="font-size:1.5rem;font-weight:700;color:#111827;margin-bottom:10px;">
        Welcome to 3D Print Verifier
      </h2>
      <p style="font-size:0.95rem;color:#4b5563;line-height:1.6;margin-bottom:20px;">
        A machine learning tool that predicts whether your 3D print will succeed before
        fabrication begins. Upload your model, configure your print settings, and get instant
        feedback on potential issues.
      </p>

      <h3 style="font-size:1rem;font-weight:600;color:#111827;margin-bottom:10px;">How it works</h3>
      <ul class="how-list">
        <li>Navigate to the Predict page to upload your .3mf or .gcode file</li>
        <li>Configure your print settings in the sidebar</li>
        <li>Review the success prediction and risk analysis</li>
        <li>Explore model performance metrics in the Model Explorer</li>
      </ul>

      <h3 style="font-size:1rem;font-weight:600;color:#111827;margin:20px 0 10px 0;">Features</h3>
      <ul class="how-list">
        <li>Multi-model ensemble predictions using Neural Network and Logistic Regression</li>
        <li>Real-time risk flag detection for common print failures</li>
        <li>Personalized recommendations for optimal print settings</li>
        <li>Alternative fabrication method suggestions</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
