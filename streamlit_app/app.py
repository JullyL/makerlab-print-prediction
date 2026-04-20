import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from streamlit_app.utils import inject_css

st.set_page_config(
    page_title="3D Print Verifier",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

with st.sidebar:
    st.markdown(
        "<h2 style='font-size:1.1rem;font-weight:700;margin:0 0 2px 0;color:#111827;'>"
        "🖨️ 3D Print Verifier</h2>"
        "<p style='font-size:0.75rem;color:#9ca3af;margin:0 0 8px 0;'>PAML Final Project</p>",
        unsafe_allow_html=True,
    )

_here = os.path.dirname(__file__)
pg = st.navigation([
    st.Page(os.path.join(_here, "pages", "home.py"),             title="Home",           icon="🏠"),
    st.Page(os.path.join(_here, "pages", "1_Predict.py"),        title="Predict",        icon="🎯"),
    st.Page(os.path.join(_here, "pages", "2_Model_Explorer.py"), title="Model Explorer", icon="📊"),
    st.Page(os.path.join(_here, "pages", "3_About.py"),          title="About",          icon="ℹ️"),
])
pg.run()
