import streamlit as st

GREEN     = "#22c55e"
GREEN_BG  = "#f0fdf4"
GREEN_BD  = "#86efac"
AMBER_BG  = "#fffbeb"
AMBER_BD  = "#fcd34d"
RED_BG    = "#fff1f2"
RED_TEXT  = "#e11d48"
BLUE      = "#2563eb"
GRAY_TEXT = "#6b7280"
CARD_BG   = "#ffffff"
PAGE_BG   = "#f3f4f6"

GLOBAL_CSS = f"""
<style>
#MainMenu {{ visibility: hidden; }}
footer    {{ visibility: hidden; }}
header    {{ visibility: hidden; }}

.stApp {{
    background-color: {PAGE_BG};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}

.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 900px;
}}

[data-testid="stSidebar"] {{
    background-color: #f9fafb;
    border-right: 1px solid #e5e7eb;
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 1rem !important;
    max-width: 100%;
}}
[data-testid="stSidebarNav"] {{
    padding-top: 0;
}}

.sidebar-section-label {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: {GRAY_TEXT};
    text-transform: uppercase;
    margin: 1rem 0 0.4rem 0;
    padding-left: 0.1rem;
}}

.card {{
    background: {CARD_BG};
    border-radius: 12px;
    padding: 24px 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}}

.file-banner {{
    background: {GREEN_BG};
    border: 1.5px dashed {GREEN_BD};
    border-radius: 10px;
    padding: 14px 20px;
    text-align: center;
    color: #15803d;
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 20px;
}}

.metric-row {{
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}}
.metric-card {{
    flex: 1;
    background: {CARD_BG};
    border-radius: 12px;
    padding: 22px 20px 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    text-align: center;
}}
.metric-card .metric-value {{
    font-size: 2.1rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.1;
}}
.metric-card .metric-label {{
    font-size: 0.78rem;
    color: {GRAY_TEXT};
    margin-top: 6px;
}}

.score-value {{
    font-size: 3rem;
    font-weight: 700;
    color: {GREEN};
    line-height: 1;
}}
.score-badge {{
    display: inline-block;
    background: {GREEN_BG};
    color: #166534;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 8px;
}}
.fail-badge {{
    display: inline-block;
    background: {RED_BG};
    color: {RED_TEXT};
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 8px;
}}

.conf-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}}
.conf-label {{
    font-size: 0.87rem;
    color: #374151;
    width: 130px;
    flex-shrink: 0;
}}
.conf-bar-wrap {{
    flex: 1;
    background: #e5e7eb;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}}
.conf-bar {{
    height: 8px;
    border-radius: 6px;
}}
.conf-pct {{
    font-size: 0.87rem;
    font-weight: 600;
    color: #111827;
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}}

.risk-section-label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {GRAY_TEXT};
    margin: 18px 0 8px 0;
}}
.flag {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    line-height: 1.5;
}}
.flag-warn {{
    background: {AMBER_BG};
    border: 1px solid {AMBER_BD};
    color: #92400e;
}}
.flag-ok {{
    background: {GREEN_BG};
    border: 1px solid {GREEN_BD};
    color: #166534;
}}

.sug-table {{
    width: 100%;
    border-collapse: collapse;
}}
.sug-table tr {{
    border-bottom: 1px solid #f3f4f6;
}}
.sug-table td {{
    padding: 10px 4px;
    font-size: 0.9rem;
    color: #374151;
}}
.sug-table td:last-child {{
    text-align: right;
    font-weight: 600;
    color: #111827;
}}
.sug-ok {{
    color: {GREEN};
}}
.apply-link {{
    color: {BLUE};
    font-size: 0.87rem;
    cursor: pointer;
    text-decoration: none;
}}

.fs-table {{
    width: 100%;
    border-collapse: collapse;
}}
.fs-table tr {{
    border-bottom: 1px solid #f3f4f6;
}}
.fs-table td {{
    padding: 12px 4px;
    font-size: 0.9rem;
}}
.fs-table td:first-child {{
    color: {GRAY_TEXT};
}}
.fs-table td:last-child {{
    text-align: right;
    font-weight: 600;
    color: #111827;
}}

.coef-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}}
.coef-label {{
    font-size: 0.87rem;
    color: #374151;
    width: 160px;
    flex-shrink: 0;
}}
.coef-bar-wrap {{
    flex: 1;
    background: #f3f4f6;
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
}}
.coef-bar {{
    height: 10px;
    background: #93c5fd;
    border-radius: 4px;
}}
.coef-pct {{
    font-size: 0.87rem;
    font-weight: 600;
    color: #111827;
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}}

.cm-wrap {{
    display: inline-grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 12px;
}}
.cm-cell {{
    border-radius: 10px;
    padding: 18px 24px;
    text-align: center;
    min-width: 110px;
}}
.cm-cell .cm-num {{
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}}
.cm-cell .cm-lbl {{
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 4px;
}}
.cm-tp {{ background: #dcfce7; color: #166534; }}
.cm-fp {{ background: #fee2e2; color: {RED_TEXT}; }}
.cm-fn {{ background: #fee2e2; color: {RED_TEXT}; }}
.cm-tn {{ background: #dcfce7; color: #166534; }}

button[data-baseweb="tab"] {{
    font-size: 0.9rem !important;
}}

[data-testid="stFileUploader"] > section {{
    border: 2px dashed {BLUE} !important;
    border-radius: 12px !important;
    background: #eff6ff !important;
    padding: 28px 20px !important;
    transition: border-color 0.2s, background 0.2s;
}}
[data-testid="stFileUploader"] > section:hover {{
    border-color: #1d4ed8 !important;
    background: #dbeafe !important;
}}
[data-testid="stFileUploader"] > section svg {{
    color: {BLUE} !important;
    fill: {BLUE} !important;
}}
[data-testid="stFileUploader"] > section > div > span:first-child {{
    color: #1e3a5f !important;
    font-weight: 600 !important;
}}
[data-testid="stFileUploader"] > section > div > span:last-child {{
    color: #4b7cbf !important;
}}

.page-title {{
    font-size: 1.4rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 4px;
}}
.page-subtitle {{
    font-size: 0.88rem;
    color: {GRAY_TEXT};
    margin-bottom: 20px;
}}

.how-list {{
    list-style: none;
    padding: 0;
    margin: 0;
}}
.how-list li {{
    padding: 5px 0;
    font-size: 0.92rem;
    color: #374151;
    display: flex;
    gap: 8px;
}}
.how-list li::before {{
    content: "›";
    color: {GRAY_TEXT};
    font-weight: 700;
    flex-shrink: 0;
}}

div[data-testid="stButton"] > button[kind="primary"] {{
    background-color: {BLUE} !important;
    border-color: {BLUE} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.55rem 1.5rem !important;
}}
div[data-testid="stButton"] > button[kind="primary"]:disabled {{
    background-color: #e5e7eb !important;
    border-color: #e5e7eb !important;
    color: #9ca3af !important;
    cursor: not-allowed !important;
}}
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def card(html_body: str, extra_style: str = "") -> None:
    st.markdown(
        f'<div class="card" style="{extra_style}">{html_body}</div>',
        unsafe_allow_html=True,
    )


def file_banner(filename: str) -> None:
    st.markdown(
        f'<div class="file-banner">✅ &nbsp;✓ {filename} loaded</div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(
        f'<p class="sidebar-section-label">{text}</p>',
        unsafe_allow_html=True,
    )


def risk_flag(text: str, kind: str = "warn") -> str:
    icon = "⚠️" if kind == "warn" else "✅"
    cls  = "flag-warn" if kind == "warn" else "flag-ok"
    return f'<div class="flag {cls}">{icon} &nbsp;{text}</div>'


def confusion_matrix_html(tp: int, fp: int, fn: int, tn: int) -> str:
    return (
        f'<div class="cm-wrap">'
        f'<div class="cm-cell cm-tp"><div class="cm-num">{tp}</div><div class="cm-lbl">TP</div></div>'
        f'<div class="cm-cell cm-fp"><div class="cm-num">{fp}</div><div class="cm-lbl">FP</div></div>'
        f'<div class="cm-cell cm-fn"><div class="cm-num">{fn}</div><div class="cm-lbl">FN</div></div>'
        f'<div class="cm-cell cm-tn"><div class="cm-num">{tn}</div><div class="cm-lbl">TN</div></div>'
        f'</div>'
    )


def coef_bar_html(label: str, value_pct: int, max_pct: int = 100) -> str:
    bar_width = int(value_pct / max_pct * 100)
    return (
        f'<div class="coef-row">'
        f'<div class="coef-label">{label}</div>'
        f'<div class="coef-bar-wrap"><div class="coef-bar" style="width:{bar_width}%"></div></div>'
        f'<div class="coef-pct">{value_pct}%</div>'
        f'</div>'
    )


def conf_bar_html(label: str, pct: float, color: str = "#22c55e") -> str:
    return (
        f'<div class="conf-row">'
        f'<div class="conf-label">{label}</div>'
        f'<div class="conf-bar-wrap"><div class="conf-bar" style="width:{pct:.0f}%;background:{color}"></div></div>'
        f'<div class="conf-pct">{pct:.0f}%</div>'
        f'</div>'
    )
