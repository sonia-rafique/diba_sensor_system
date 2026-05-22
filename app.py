import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.engine import DIBAOncologyEngine, LUNG_CANCER_VOC_RISK
from core.explainers import DIBAExplainableAI
from core.twin import VirtualPatientProfileTwin
from core.recommend import DIBAClinicalRecommendationEngine

st.set_page_config(
    page_title="DIBA Oncology Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body { font-family: 'Plus Jakarta Sans', sans-serif !important; }

[data-testid="stAppViewContainer"] {
    background: #F4F6FB !important;
}
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04);
}
[data-testid="stSidebar"] * { color: #374151 !important; }
section[data-testid="stSidebarContent"] { padding: 1.5rem 1.25rem !important; }

.main-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 50%, #1E40AF 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(37,99,235,0.2);
}
.main-header::after {
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
    pointer-events: none;
}

.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.card:hover { box-shadow: 0 4px 16px rgba(37,99,235,0.08); border-color: #BFDBFE; }

.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    border-left: 4px solid;
    transition: all 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.08); }

.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2563EB;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.rec-item {
    background: #F8FAFF;
    border: 1px solid #DBEAFE;
    border-left: 4px solid #2563EB;
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #374151;
    line-height: 1.5;
}
.rec-critical { border-left-color: #DC2626 !important; background: #FFF5F5 !important; border-color: #FCA5A5 !important; }
.rec-warning  { border-left-color: #D97706 !important; background: #FFFBEB !important; border-color: #FCD34D !important; }

.alert-box {
    background: #FFF5F5;
    border: 1px solid #FCA5A5;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: #991B1B;
    font-weight: 600;
}
.alert-moderate {
    background: #FFFBEB;
    border-color: #FCD34D;
    color: #92400E;
}

.model-best-badge {
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: white !important;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.dept-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}

.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 0.3rem;
    gap: 0.2rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: #6B7280 !important;
    font-weight: 500;
    padding: 0.5rem 1.1rem;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: #EFF6FF !important;
    color: #2563EB !important;
    font-weight: 700;
}

.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.25) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.35) !important;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
div[data-testid="stMetricLabel"] p { color: #6B7280 !important; font-size: 0.75rem !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"] { color: #111827 !important; font-size: 1.4rem !important; font-weight: 700 !important; }

.stSelectbox > div > div {
    background: #FFFFFF !important;
    border-color: #E2E8F0 !important;
    color: #111827 !important;
    border-radius: 8px !important;
}
.stNumberInput > div > div input {
    background: #FFFFFF !important;
    border-color: #E2E8F0 !important;
    color: #111827 !important;
}
.stTextInput > div > div input {
    background: #FFFFFF !important;
    border-color: #E2E8F0 !important;
    color: #111827 !important;
}
label { color: #374151 !important; font-size: 0.82rem !important; font-weight: 600 !important; }

.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }

[data-testid="stRadio"] label { color: #374151 !important; font-size: 0.875rem !important; }

div.stSpinner > div { border-top-color: #2563EB !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #F4F6FB; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#FAFBFF',
    font_color='#374151',
    margin=dict(l=10, r=10, t=35, b=10),
)
AXIS_STYLE = dict(gridcolor='#E2E8F0', linecolor='#E2E8F0', tickfont=dict(color='#6B7280', size=10))

def hex_to_rgba(hex_color, alpha=0.15):
    header = hex_color.lstrip('#')
    r, g, b = int(header[0:2], 16), int(header[2:4], 16), int(header[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

COLOR_MAP = {
    'Random Forest':      '#2563EB',
    'XGBoost':            '#7C3AED',
    'SVM':                '#059669',
    'Logistic Regression':'#DC2626',
    'Decision Tree':      '#D97706',
}

TIER_CONFIG = {
    'Safe Tier':       {'color': '#059669', 'bg': '#ECFDF5', 'border': '#A7F3D0', 'icon': 'SAFE'},
    'Moderate Concern':{'color': '#D97706', 'bg': '#FFFBEB', 'border': '#FCD34D', 'icon': 'MODERATE'},
    'High Risk Tier':  {'color': '#DC2626', 'bg': '#FFF5F5', 'border': '#FCA5A5', 'icon': 'HIGH RISK'},
    'Critical Alert':  {'color': '#7C3AED', 'bg': '#F5F3FF', 'border': '#DDD6FE', 'icon': 'CRITICAL'},
}


@st.cache_resource(show_spinner="Training ML classifiers on gsalc.csv...")
def load_engine():
    engine = DIBAOncologyEngine(dataset_path="Dataset/gsalc.csv")
    metrics, best_model = engine.train()
    return engine, metrics, best_model


@st.cache_data
def load_raw_dataset():
    df = pd.read_csv("Dataset/gsalc.csv", header=None)
    rows = []
    for idx, row in df.iterrows():
        gas = str(row[0]).strip()
        ppb_str = str(row[1]).replace('ppb', '').strip()
        try:
            ppb = float(ppb_str)
        except ValueError:
            ppb = 100.0
        signals = pd.to_numeric(row[2:], errors='coerce').dropna().values
        if len(signals) > 0:
            rows.append({
                'Row':               idx,
                'Gas Type':          gas.title(),
                'Concentration (PPB)': ppb,
                'Sensor Mean':       round(float(np.mean(signals)), 4),
                'Sensor Max':        round(float(np.max(signals)), 4),
                'Sensor Min':        round(float(np.min(signals)), 4),
                'Sensor Std':        round(float(np.std(signals)), 4),
                'Signal Range':      round(float(np.max(signals) - np.min(signals)), 4),
            })
    return pd.DataFrame(rows)


engine, all_metrics, best_model_name = load_engine()
df_raw = load_raw_dataset()

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

with st.sidebar:
    st.markdown("""
        <div style='margin-bottom:1.5rem;'>
            <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;'>
                <div style='width:36px; height:36px; background:linear-gradient(135deg,#2563EB,#1D4ED8);
                            border-radius:9px; display:flex; align-items:center; justify-content:center;'>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
                        <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
                        <path d="M6 18H4a2 2 0 0 0-2 2v2"/><path d="M14 21v-1a2 2 0 0 0-2-2H9"/>
                    </svg>
                </div>
                <div>
                    <div style='font-size:1.1rem; font-weight:800; color:#111827; letter-spacing:-0.02em;'>DIBA</div>
                    <div style='font-size:0.65rem; color:#2563EB; font-weight:700; text-transform:uppercase; letter-spacing:0.1em;'>Oncology AI</div>
                </div>
            </div>
            <div style='font-size:0.75rem; color:#9CA3AF; padding-left:0.1rem;'>Exhaled VOC Lung Cancer Detection</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px; background:#F3F4F6; margin:0.75rem 0 1.25rem;'></div>", unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["Command Dashboard", "Model Performance", "Dataset Explorer", "Department Matrix"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:1px; background:#F3F4F6; margin:1.25rem 0;'></div>", unsafe_allow_html=True)

    best_f1  = all_metrics[best_model_name]['F1 Score']
    best_auc = all_metrics[best_model_name]['AUC-ROC']
    best_cv  = all_metrics[best_model_name]['CV F1 Mean']

    st.markdown(f"""
        <div style='background:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px; padding:1rem;'>
            <div style='font-size:0.65rem; color:#2563EB; font-weight:800; text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom:0.6rem;'>Active Best Model</div>
            <div style='font-size:0.95rem; font-weight:800; color:#1E40AF; margin-bottom:0.6rem;'>{best_model_name}</div>
            <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.4rem;'>
                <div style='background:white; border-radius:8px; padding:0.4rem 0.5rem; text-align:center;'>
                    <div style='font-size:0.6rem; color:#9CA3AF; font-weight:700;'>F1</div>
                    <div style='font-size:0.88rem; font-weight:800; color:#059669;'>{best_f1:.3f}</div>
                </div>
                <div style='background:white; border-radius:8px; padding:0.4rem 0.5rem; text-align:center;'>
                    <div style='font-size:0.6rem; color:#9CA3AF; font-weight:700;'>AUC</div>
                    <div style='font-size:0.88rem; font-weight:800; color:#2563EB;'>{best_auc:.3f}</div>
                </div>
                <div style='background:white; border-radius:8px; padding:0.4rem 0.5rem; text-align:center;'>
                    <div style='font-size:0.6rem; color:#9CA3AF; font-weight:700;'>CV</div>
                    <div style='font-size:0.88rem; font-weight:800; color:#7C3AED;'>{best_cv:.3f}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style='margin-top:1.25rem; background:#F9FAFB; border:1px solid #E5E7EB; border-radius:10px; padding:0.85rem;'>
            <div style='font-size:0.65rem; color:#9CA3AF; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>System Status</div>
            <div style='display:flex; align-items:center; gap:0.4rem; margin-bottom:0.3rem;'>
                <div style='width:7px; height:7px; background:#059669; border-radius:50%;'></div>
                <span style='font-size:0.78rem; color:#374151; font-weight:600;'>Engine Operational</span>
            </div>
            <div style='font-size:0.72rem; color:#9CA3AF;'>Dataset: {len(df_raw)} samples loaded</div>
            <div style='font-size:0.72rem; color:#9CA3AF;'>Models trained: 5 classifiers</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.caption("DIBA Platform v3.1 · Production Build\ngsalc.csv · 6 VOC gases · 90 samples")


if nav == "Command Dashboard":

    st.markdown(f"""
        <div class="main-header">
            <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                <div>
                    <div style='display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;'>
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" style='opacity:0.9;'>
                            <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
                            <path d="M6 18H4a2 2 0 0 0-2 2v2"/><path d="M14 21v-1a2 2 0 0 0-2-2H9"/>
                        </svg>
                        <div style='font-size:1.5rem; font-weight:800; color:white; letter-spacing:-0.03em;'>
                            Lung Cancer VOC Detection — Command Dashboard
                        </div>
                    </div>
                    <div style='font-size:0.875rem; color:rgba(255,255,255,0.7);'>
                        Real-time exhaled breath biomarker classification &nbsp;·&nbsp; gsalc.csv dataset &nbsp;·&nbsp; {len(df_raw)} clinical samples &nbsp;·&nbsp; 6 VOC gas types
                    </div>
                </div>
                <div style='text-align:right; flex-shrink:0;'>
                    <div style='background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25);
                                border-radius:10px; padding:0.6rem 1rem;'>
                        <div style='font-size:0.65rem; color:rgba(255,255,255,0.7); font-weight:700; text-transform:uppercase;'>System Active</div>
                        <div style='font-size:0.82rem; color:white; font-weight:700; margin-top:0.2rem;'>All 5 Models Ready</div>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    kpi_meta = [
        ("Total Samples",     str(len(df_raw)),  "gsalc.csv validated records",  "#2563EB", "M9 3H5a2 2 0 0 0-2 2v4m6-6h10l4 4v10a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm3 3v4h4"),
        ("VOC Gas Types",     "6",               "Toluene · Hexane · Acetone+3", "#7C3AED", "M19.428 15.428a2 2 0 0 0-1.022-.547l-2.387-.477a6 6 0 0 0-3.86.517l-.318.158a6 6 0 0 1-3.86.517L6.05 15.21a2 2 0 0 0-1.806.547M8 4h8l-1 1v5.172a2 2 0 0 0 .586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 0 0 9 10.172V5L8 4z"),
        ("Concentration Levels","3",             "50 PPB · 100 PPB · 200 PPB",   "#059669", "M19.428 15.428a2 2 0 0 0-1.022-.547l-2.387-.477a6 6 0 0 0-3.86.517l-.318.158a6 6 0 0 1-3.86.517L6.05 15.21a2 2 0 0 0-1.806.547M8 4h8l-1 1v5.172a2 2 0 0 0 .586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 0 0 9 10.172V5L8 4z"),
        ("Best Model F1",     f"{best_f1:.3f}",  best_model_name,                "#DC2626", "M9 19v-6a2 2 0 0 0-2-2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h2a2 2 0 0 0 2-2v1a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v1a2 2 0 0 0 2 2h2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2h-2a2 2 0 0 0-2 2v6"),
        ("Best AUC-ROC",      f"{best_auc:.3f}", "Cross-validated score",         "#D97706", "M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2z"),
    ]

    kpi_cols = st.columns(5)
    for col, (label, val, sub, color, icon_path) in zip(kpi_cols, kpi_meta):
        with col:
            st.markdown(f"""
                <div class="kpi-card" style='border-left-color:{color};'>
                    <div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.75rem;'>
                        <div style='width:30px; height:30px; background:{color}15; border-radius:8px;
                                    display:flex; align-items:center; justify-content:center; flex-shrink:0;'>
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2">
                                <path d="{icon_path}"/>
                            </svg>
                        </div>
                        <div style='font-size:0.68rem; font-weight:700; color:#6B7280; text-transform:uppercase; letter-spacing:0.06em;'>{label}</div>
                    </div>
                    <div style='font-size:1.75rem; font-weight:800; color:{color}; line-height:1; margin-bottom:0.3rem;'>{val}</div>
                    <div style='font-size:0.7rem; color:#9CA3AF;'>{sub}</div>
                </div>
            """, unsafe_allow_html=True)

    tab_predict, tab_twin, tab_trends = st.tabs([
        "Prediction Engine",
        "Digital Twin Simulator",
        "Signal Trend Analysis"
    ])

    with tab_predict:
        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("""
                <div class="section-label">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
                    </svg>
                    Sample Configuration
                </div>
            """, unsafe_allow_html=True)

            gas_options = sorted(df_raw['Gas Type'].unique().tolist())
            selected_gas = st.selectbox("VOC Gas Type", options=gas_options)

            ppb_options = sorted(df_raw['Concentration (PPB)'].unique().tolist())
            selected_ppb = st.selectbox("Concentration (PPB)", options=ppb_options)

            filtered_rows = df_raw[
                (df_raw['Gas Type'] == selected_gas) &
                (df_raw['Concentration (PPB)'] == selected_ppb)
            ]

            selected = None
            if len(filtered_rows) > 0:
                row_choice = st.selectbox(
                    "Sample Row",
                    options=filtered_rows.index.tolist(),
                    format_func=lambda x: f"Sample #{x}  |  Mean={filtered_rows.loc[x,'Sensor Mean']:.4f}  |  Range={filtered_rows.loc[x,'Signal Range']:.4f}"
                )
                selected = filtered_rows.loc[row_choice]

                c1, c2 = st.columns(2)
                with c1:
                    s_mean = st.number_input("Sensor Mean",  value=float(selected['Sensor Mean']),  format="%.4f")
                    s_max  = st.number_input("Sensor Max",   value=float(selected['Sensor Max']),   format="%.4f")
                with c2:
                    s_min  = st.number_input("Sensor Min",   value=float(selected['Sensor Min']),   format="%.4f")
                    s_std  = st.number_input("Sensor Std",   value=float(selected['Sensor Std']),   format="%.4f")

                s_range  = s_max - s_min
                voc_info = LUNG_CANCER_VOC_RISK.get(selected_gas.lower(), {'weight': 0.5, 'risk': 'Unknown'})

                risk_color = {'Very High':'#DC2626','High':'#D97706','Moderate':'#2563EB','Lower':'#059669'}.get(voc_info.get('risk','Unknown'), '#6B7280')

                st.markdown(f"""
                    <div style='background:#F8FAFF; border:1px solid #BFDBFE; border-radius:10px;
                                padding:0.75rem 1rem; margin-top:0.5rem; display:flex; justify-content:space-between; align-items:center;'>
                        <div>
                            <div style='font-size:0.65rem; color:#2563EB; font-weight:800; text-transform:uppercase; letter-spacing:0.08em;'>VOC Oncological Profile</div>
                            <div style='font-size:0.82rem; color:#374151; margin-top:0.2rem;'>
                                {selected_gas} &nbsp;·&nbsp; Risk Weight: <strong style='color:#374151;'>{voc_info.get("weight",0.5):.2f}</strong>
                            </div>
                        </div>
                        <div style='background:{risk_color}15; border:1px solid {risk_color}40; border-radius:8px;
                                    padding:0.35rem 0.75rem; font-size:0.75rem; font-weight:800; color:{risk_color};'>
                            {voc_info.get("risk","Unknown")}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            if selected is not None:
                if st.button("Run Classification Pipeline", use_container_width=True):
                    payload = {
                        'gas_label':      selected_gas.lower(),
                        'ppb':            float(selected_ppb),
                        'sensor_mean':    s_mean,
                        'sensor_max':      s_max,
                        'sensor_min':      s_min,
                        'sensor_std':      s_std,
                        'sensor_range':    s_range,
                        'voc_risk_weight': voc_info.get('weight', 0.5),
                    }
                    with st.spinner("Running VOC classification pipeline..."):
                        result          = engine.predict(payload)
                        attribution     = DIBAExplainableAI.calculate_local_attribution(payload)
                        explanation     = DIBAExplainableAI.generate_explanation_text(result, attribution)
                        all_model_scores= engine.get_all_model_proba(payload)
                        actions, notifs = DIBAClinicalRecommendationEngine.evaluate_clinical_action(
                            result['score'], result['tier'], selected_gas
                        )
                        st.session_state.prediction_result = {
                            'result': result, 'attribution': attribution,
                            'explanation': explanation, 'all_model_scores': all_model_scores,
                            'actions': actions, 'notifs': notifs, 'payload': payload,
                        }
                    st.rerun()

        with right_col:
            if st.session_state.prediction_result is not None:
                pr    = st.session_state.prediction_result
                res   = pr['result']
                score = res['score']
                tier  = res['tier']
                tc    = TIER_CONFIG.get(tier, TIER_CONFIG['Safe Tier'])
                color = tc['color']

                st.markdown(f"""
                    <div class="card" style='border-color:{tc["border"]}; background:{tc["bg"]};'>
                        <div class="section-label" style='color:{color};'>
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                            </svg>
                            Classification Result
                        </div>
                        <div style='display:flex; align-items:center; gap:2rem; padding:0.5rem 0;'>
                            <div style='text-align:center;'>
                                <div style='font-size:3.5rem; font-weight:900; color:{color}; line-height:1;'>{score}</div>
                                <div style='font-size:0.72rem; color:#6B7280; font-weight:600;'>/ 100 Risk Score</div>
                            </div>
                            <div style='flex:1;'>
                                <div style='background:{color}20; border:2px solid {color}40; border-radius:12px;
                                            padding:0.5rem 1rem; display:inline-block; margin-bottom:0.6rem;'>
                                    <span style='color:{color}; font-weight:800; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.06em;'>{tc["icon"]}</span>
                                </div>
                                <div style='font-size:0.78rem; color:#374151;'>
                                    Model: <strong>{res['model_used']}</strong><br/>
                                    Prob: <strong>{res['prob']*100:.1f}%</strong> &nbsp;·&nbsp;
                                    VOC Risk: <strong>{res['voc_risk']}</strong>
                                </div>
                            </div>
                        </div>
                        <div style='background:{color}15; border-radius:8px; height:10px; overflow:hidden; margin-top:0.5rem;'>
                            <div style='height:100%; width:{score}%; background:{color}; border-radius:8px; transition:width 0.6s ease;'></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#9CA3AF', 'tickwidth': 1,
                                 'tickfont': {'color': '#9CA3AF', 'size': 9}},
                        'bar':  {'color': color, 'thickness': 0.22},
                        'bgcolor': '#F9FAFB',
                        'bordercolor': '#E2E8F0',
                        'steps': [
                            {'range': [0, 25],   'color': '#ECFDF5'},
                            {'range': [25, 50],  'color': '#FFFBEB'},
                            {'range': [50, 75],  'color': '#FFF5F5'},
                            {'range': [75, 100], 'color': '#F5F3FF'},
                        ],
                        'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.85, 'value': score},
                    },
                    number={'font': {'color': color, 'size': 32, 'family': 'Plus Jakarta Sans'}, 'suffix': '%'},
                ))
                gauge_fig.update_layout(
                    height=200, margin=dict(l=20, r=20, t=20, b=10),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#374151',
                )
                st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})

                st.markdown(f"""
                    <div style='background:#F8FAFF; border:1px solid #DBEAFE; border-radius:10px;
                                padding:0.85rem 1rem; font-size:0.82rem; color:#374151; line-height:1.6; margin-bottom:1rem;'>
                        <strong style='color:#1E40AF;'>AI Explanation:</strong> {pr['explanation']}
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                    <div class="section-label">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                            <rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8m-4-4v4"/>
                        </svg>
                        All Models — Risk Score Comparison
                    </div>
                """, unsafe_allow_html=True)

                model_scores = pr['all_model_scores']
                bar_colors   = [COLOR_MAP.get(k,'#2563EB') for k in model_scores.keys()]
                fig_bar = go.Figure(go.Bar(
                    x=list(model_scores.keys()),
                    y=list(model_scores.values()),
                    marker_color=bar_colors,
                    marker_line_width=0,
                    text=[f"{v:.1f}%" for v in model_scores.values()],
                    textposition='outside',
                    textfont={'color': '#374151', 'size': 10, 'family': 'Plus Jakarta Sans'},
                ))
                fig_bar.update_layout(
                    height=230,
                    **PLOT_LAYOUT,
                    xaxis=dict(**AXIS_STYLE, title=None),
                    yaxis=dict(**AXIS_STYLE, range=[0, 120], title='Risk Probability (%)'),
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

                st.markdown("""
                    <div class="section-label">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                            <path d="M9 17H5a2 2 0 0 0-2 2v0a2 2 0 0 0 2 2h4m6-18H9a2 2 0 0 0-2 2v0a2 2 0 0 0 2 2h6"/>
                            <line x1="12" y1="12" x2="12" y2="12"/><path d="M12 6v6l3 3"/>
                        </svg>
                        XAI — Feature Attribution Weights
                    </div>
                """, unsafe_allow_html=True)

                attr = pr['attribution']
                attr_colors = ['#2563EB', '#7C3AED', '#059669', '#D97706', '#DC2626', '#0891B2']
                fig_attr = go.Figure(go.Bar(
                    y=list(attr.keys()),
                    x=[v * 100 for v in attr.values()],
                    orientation='h',
                    marker_color=attr_colors[:len(attr)],
                    marker_line_width=0,
                    text=[f"{v*100:.1f}%" for v in attr.values()],
                    textposition='outside',
                    textfont={'color': '#374151', 'size': 10},
                ))
                fig_attr.update_layout(
                    height=220,
                    **{**PLOT_LAYOUT, 'margin': dict(l=10, r=70, t=10, b=10)},
                    xaxis=dict(**AXIS_STYLE, title='Attribution Weight (%)'),
                    yaxis=dict(**AXIS_STYLE),
                    showlegend=False,
                )
                st.plotly_chart(fig_attr, use_container_width=True, config={'displayModeBar': False})

                if pr['notifs']:
                    for notif in pr['notifs']:
                        cls = "alert-box" if "CRITICAL" in notif else "alert-box alert-moderate"
                        st.markdown(f"<div class='{cls}'>{notif}</div>", unsafe_allow_html=True)

                st.markdown("""
                    <div class="section-label">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                            <path d="M9 12l2 2 4-4m6 2a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"/>
                        </svg>
                        Clinical Recommendations
                    </div>
                """, unsafe_allow_html=True)

                rec_cls = "rec-critical" if score > 75 else "rec-warning" if score > 50 else ""
                for i, action in enumerate(pr['actions'], 1):
                    st.markdown(f"""
                        <div class='rec-item {rec_cls if i == 1 else ""}'>
                            <span style='color:#2563EB; font-weight:800; font-family:JetBrains Mono,monospace;'>#{i:02d}</span>
                            &nbsp; {action}
                        </div>
                    """, unsafe_allow_html=True)

            else:
                st.markdown("""
                    <div class="card" style='text-align:center; padding:3rem 2rem; border:2px dashed #E2E8F0;'>
                        <div style='width:56px; height:56px; background:#EFF6FF; border-radius:16px;
                                    display:flex; align-items:center; justify-content:center; margin:0 auto 1rem;'>
                            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2">
                                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                            </svg>
                        </div>
                        <div style='font-size:1rem; font-weight:700; color:#374151;'>No Prediction Yet</div>
                        <div style='font-size:0.82rem; color:#9CA3AF; margin-top:0.4rem;'>
                            Select a gas sample and click Run Classification Pipeline
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    with tab_twin:
        if st.session_state.prediction_result is not None:
            pr  = st.session_state.prediction_result
            res = pr['result']

            twin = VirtualPatientProfileTwin(
                patient_id=f"PT-{abs(hash(str(pr['payload']))) % 9999:04d}",
                baseline_payload=pr['payload'],
            )
            proj_score, summary, direction, curve = twin.compute_progression_trajectory(res['score'])

            dir_color = {'WORSENING': '#DC2626', 'MILD INCREASE': '#D97706', 'STABLE': '#059669'}.get(direction, '#2563EB')

            t1, t2 = st.columns([1, 1], gap="large")
            with t1:
                st.markdown(f"""
                    <div class="card">
                        <div class="section-label">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                            </svg>
                            60-Day Digital Twin Projection
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center;
                                    background:#F8FAFF; border:1px solid #BFDBFE; border-radius:10px;
                                    padding:1rem; margin-bottom:1rem;'>
                            <div style='text-align:center;'>
                                <div style='font-size:0.65rem; color:#6B7280; font-weight:700; text-transform:uppercase;'>Current Score</div>
                                <div style='font-size:2rem; font-weight:800; color:{res["color"]};'>{res["score"]}</div>
                            </div>
                            <div>
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#9CA3AF" stroke-width="2">
                                    <path d="M5 12h14M12 5l7 7-7 7"/>
                                </svg>
                            </div>
                            <div style='text-align:center;'>
                                <div style='font-size:0.65rem; color:#6B7280; font-weight:700; text-transform:uppercase;'>Projected Score</div>
                                <div style='font-size:2rem; font-weight:800; color:{dir_color};'>{proj_score}</div>
                            </div>
                            <div style='text-align:center;'>
                                <div style='font-size:0.65rem; color:#6B7280; font-weight:700; text-transform:uppercase;'>Trajectory</div>
                                <div style='background:{dir_color}15; border:1px solid {dir_color}40; border-radius:8px;
                                            padding:0.3rem 0.6rem; font-size:0.72rem; font-weight:800; color:{dir_color};'>{direction}</div>
                            </div>
                        </div>
                        <div style='background:#F9FAFB; border:1px solid #E5E7EB; border-radius:8px;
                                    padding:0.75rem; font-size:0.82rem; color:#374151; line-height:1.6;'>
                            {summary}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with t2:
                months = [pt['month'] for pt in curve]
                scores = [pt['score'] for pt in curve]

                fill_rgba = hex_to_rgba(dir_color, 0.08)

                fig_twin = go.Figure()
                fig_twin.add_trace(go.Scatter(
                    x=months, y=scores,
                    mode='lines+markers',
                    line=dict(color=dir_color, width=2.5),
                    marker=dict(size=8, color=dir_color, line=dict(color='white', width=2)),
                    fill='tozeroy',
                    fillcolor=fill_rgba,
                    name='Projected Risk',
                ))
                fig_twin.add_hline(
                    y=res['score'], line_dash='dash', line_color='#2563EB', line_width=1.5,
                    annotation_text="Current", annotation_font_color='#2563EB',
                    annotation_font_size=10,
                )
                fig_twin.update_layout(
                    height=290, title='6-Month Risk Trajectory Forecast',
                    title_font=dict(color='#6B7280', size=12, family='Plus Jakarta Sans'),
                    **PLOT_LAYOUT,
                    xaxis=dict(**AXIS_STYLE),
                    yaxis=dict(**AXIS_STYLE, range=[0, 110], title='Risk Score'),
                    showlegend=False,
                )
                st.plotly_chart(fig_twin, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Run the classification pipeline first to activate the Digital Twin simulator.")

    with tab_trends:
        col_t1, col_t2 = st.columns([1, 1], gap="large")

        with col_t1:
            st.markdown("""
                <div class="section-label">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                        <rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/>
                    </svg>
                    Sensor Mean Distribution by Gas Type
                </div>
            """, unsafe_allow_html=True)
            fig_box = px.box(
                df_raw, x='Gas Type', y='Sensor Mean', color='Gas Type',
                color_discrete_sequence=['#2563EB','#7C3AED','#059669','#D97706','#DC2626','#0891B2'],
                points='all',
            )
            fig_box.update_layout(
                height=300, **PLOT_LAYOUT,
                xaxis=dict(**AXIS_STYLE, title=None),
                yaxis=dict(**AXIS_STYLE, title='Sensor Mean Value'),
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})

        with col_t2:
            st.markdown("""
                <div class="section-label">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    </svg>
                    Signal Range vs Concentration
                </div>
            """, unsafe_allow_html=True)
            fig_scatter = px.scatter(
                df_raw,
                x='Concentration (PPB)', y='Signal Range',
                color='Gas Type',
                size='Sensor Max',
                color_discrete_sequence=['#2563EB','#7C3AED','#059669','#D97706','#DC2626','#0891B2'],
            )
            fig_scatter.update_layout(
                height=300, **PLOT_LAYOUT,
                xaxis=dict(**AXIS_STYLE, title='Concentration (PPB)'),
                yaxis=dict(**AXIS_STYLE, title='Signal Range'),
                legend=dict(font=dict(color='#6B7280', size=9), bgcolor='rgba(0,0,0,0)'),
            )
            st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

        st.markdown("""
            <div class="section-label" style='margin-top:0.5rem;'>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <path d="M3 3h18v18H3z"/><path d="M3 9h18M9 21V9"/>
                </svg>
                Mean Sensor Response — Gas vs Concentration Heatmap
            </div>
        """, unsafe_allow_html=True)

        pivot = df_raw.pivot_table(
            index='Gas Type', columns='Concentration (PPB)',
            values='Sensor Mean', aggfunc='mean'
        )
        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{int(c)} PPB" for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale='Blues',
            text=np.round(pivot.values, 4),
            texttemplate='%{text}',
            textfont=dict(size=10, color='#1E40AF'),
            showscale=True,
            colorbar=dict(tickfont=dict(color='#6B7280', size=9)),
        ))
        fig_hm.update_layout(
            height=280,
            **{**PLOT_LAYOUT, 'margin': dict(l=10, r=10, t=10, b=10)},
            xaxis=dict(tickfont=dict(color='#374151', size=10)),
            yaxis=dict(tickfont=dict(color='#374151', size=10)),
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})

        st.markdown("""
            <div class="section-label" style='margin-top:0.5rem;'>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
                Sensor Max vs Sensor Min per Gas Type
            </div>
        """, unsafe_allow_html=True)

        gas_colors_map = {
            'Ethanol':'#2563EB','Acetone':'#7C3AED','Toluene':'#DC2626',
            'Ethyl Acetate':'#059669','Isopropanol':'#D97706','Hexane':'#0891B2'
        }
        fig_line = go.Figure()
        for gas in df_raw['Gas Type'].unique():
            sub = df_raw[df_raw['Gas Type'] == gas].sort_values('Concentration (PPB)')
            c   = gas_colors_map.get(gas, '#2563EB')
            fig_line.add_trace(go.Scatter(
                x=sub['Concentration (PPB)'], y=sub['Sensor Max'],
                mode='lines+markers', name=f'{gas} (Max)',
                line=dict(color=c, width=2),
                marker=dict(size=6, color=c),
            ))
            fig_line.add_trace(go.Scatter(
                x=sub['Concentration (PPB)'], y=sub['Sensor Min'],
                mode='lines', name=f'{gas} (Min)',
                line=dict(color=c, width=1, dash='dot'),
                showlegend=False,
                fill='tonexty',
                fillcolor=hex_to_rgba(c, 0.06),
            ))
        fig_line.update_layout(
            height=300, **PLOT_LAYOUT,
            xaxis=dict(**AXIS_STYLE, title='Concentration (PPB)'),
            yaxis=dict(**AXIS_STYLE, title='Sensor Value'),
            legend=dict(font=dict(color='#6B7280', size=9), bgcolor='rgba(0,0,0,0)', orientation='h',
                        x=0, y=1.12),
        )
        st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})


elif nav == "Model Performance":

    st.markdown("""
        <div class="main-header">
            <div style='display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;'>
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                    <path d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2z"/>
                </svg>
                <div style='font-size:1.4rem; font-weight:800; color:white;'>ML Model Performance Analytics</div>
            </div>
            <div style='font-size:0.875rem; color:rgba(255,255,255,0.7);'>Comparative evaluation of 5 classifiers trained on gsalc.csv · 90 samples · 5-fold cross validation</div>
        </div>
    """, unsafe_allow_html=True)

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    model_names  = list(all_metrics.keys())

    st.markdown("""
        <div class="section-label">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <rect x="2" y="3" width="7" height="7"/><rect x="15" y="3" width="7" height="7"/>
                <rect x="2" y="14" width="7" height="7"/><rect x="15" y="14" width="7" height="7"/>
            </svg>
            Model Scorecards
        </div>
    """, unsafe_allow_html=True)
    model_cols = st.columns(len(model_names))

    for col, name in zip(model_cols, model_names):
        m      = all_metrics[name]
        is_best = (name == best_model_name)
        c      = COLOR_MAP[name]

        clean_name = name.split('<')[0].strip() if '<' in name else name
        
        metrics_html = "".join([f"""
        <div style='display:flex; justify-content:space-between; align-items:center; padding:0.2rem 0; border-bottom:1px solid #F3F4F6;'>
            <span style='font-size:0.68rem; color:#6B7280; font-weight:600;'>{metric}</span>
            <span style='font-size:0.75rem; font-weight:800; color:{"#059669" if m[metric]>=0.95 else "#374151"};'>{m[metric]:.3f}</span>
        </div>""" for metric in metric_names])
        
        with col:
            best_badge = f"<div style='margin-bottom:0.5rem;'><span class='model-best-badge'>Best Model</span></div>" if is_best else ""
            st.markdown(f"""
                <div style='background:{"#EFF6FF" if is_best else "#FFFFFF"}; border:{"2px" if is_best else "1px"} solid {c if is_best else "#E2E8F0"};
                            border-radius:12px; padding:1rem; margin-bottom:0.75rem;
                            box-shadow:{"0 0 0 3px " + c + "20" if is_best else "0 2px 6px rgba(0,0,0,0.04)"};'>
                    {best_badge}
                    <div style='font-size:0.78rem; font-weight:800; color:{c}; margin-bottom:0.6rem;'>{name}</div>
                    <div style='display:grid; gap:0.35rem;'>
                        {metrics_html}
                        <div style='display:flex; justify-content:space-between; padding-top:0.3rem;'>
                            <span style='font-size:0.68rem; color:#6B7280; font-weight:600;'>CV F1 Mean</span>
                            <span style='font-size:0.75rem; font-weight:800; color:{c};'>{m["CV F1 Mean"]:.3f}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("""
            <div class="section-label">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                </svg>
                Radar — Multi-Metric Comparison
            </div>
        """, unsafe_allow_html=True)

        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
        fig_radar  = go.Figure()

        for name in model_names:
            vals = [all_metrics[name][c] for c in categories]
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            c = COLOR_MAP[name]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=cats_closed,
                fill='toself',
                name=name,
                line=dict(color=c, width=2),
                fillcolor=hex_to_rgba(c, 0.10),
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor='#FAFBFF',
                radialaxis=dict(
                    visible=True, range=[0.8, 1.01],
                    tickfont=dict(color='#9CA3AF', size=8),
                    gridcolor='#E2E8F0', linecolor='#E2E8F0',
                    tickvals=[0.85, 0.90, 0.95, 1.00],
                    ticktext=['0.85','0.90','0.95','1.00'],
                ),
                angularaxis=dict(
                    tickfont=dict(color='#374151', size=10),
                    gridcolor='#E2E8F0', linecolor='#E2E8F0',
                ),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            height=380,
            legend=dict(font=dict(color='#374151', size=9), bgcolor='rgba(0,0,0,0)', orientation='h', x=0, y=-0.05),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        st.markdown("""
            <div class="section-label">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
                    <line x1="6" y1="20" x2="6" y2="14"/>
                </svg>
                F1 Score vs AUC-ROC Comparison
            </div>
        """, unsafe_allow_html=True)

        fig_grouped = go.Figure()
        fig_grouped.add_trace(go.Bar(
            name='F1 Score', x=model_names,
            y=[all_metrics[n]['F1 Score'] for n in model_names],
            marker_color=[COLOR_MAP[n] for n in model_names],
            marker_line_width=0,
            text=[f"{all_metrics[n]['F1 Score']:.3f}" for n in model_names],
            textposition='outside', textfont=dict(color='#374151', size=10),
        ))
        fig_grouped.add_trace(go.Bar(
            name='AUC-ROC', x=model_names,
            y=[all_metrics[n]['AUC-ROC'] for n in model_names],
            marker_color=[hex_to_rgba(COLOR_MAP[n], 0.45) for n in model_names],
            marker_line_color=[COLOR_MAP[n] for n in model_names],
            marker_line_width=1.5,
            text=[f"{all_metrics[n]['AUC-ROC']:.3f}" for n in model_names],
            textposition='outside', textfont=dict(color='#374151', size=10),
        ))
        fig_grouped.update_layout(
            barmode='group', height=380, **PLOT_LAYOUT,
            xaxis=dict(**AXIS_STYLE, tickangle=-20),
            yaxis=dict(**AXIS_STYLE, range=[0.75, 1.08], title='Score'),
            legend=dict(font=dict(color='#374151', size=10), bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig_grouped, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""
        <div class="section-label">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M21 21H4.6A1.6 1.6 0 0 1 3 19.4V3"/><path d="M7 13l4-4 4 4 4-4"/>
            </svg>
            Cross-Validation F1 Score — 5-Fold Distribution
        </div>
    """, unsafe_allow_html=True)

    fig_cv = go.Figure()
    for name in model_names:
        cv_vals = all_metrics[name]['CV F1 Scores']
        c = COLOR_MAP[name]
        fig_cv.add_trace(go.Box(
            y=cv_vals, name=name,
            marker_color=c,
            line_color=c,
            fillcolor=hex_to_rgba(c, 0.12),
            boxmean=True,
            marker_size=6,
        ))
    fig_cv.update_layout(
        height=300, **PLOT_LAYOUT,
        xaxis=dict(**AXIS_STYLE),
        yaxis=dict(**AXIS_STYLE, title='F1 Score', range=[0.4, 1.1]),
        showlegend=False,
    )
    st.plotly_chart(fig_cv, use_container_width=True, config={'displayModeBar': False})

    fi = engine.get_feature_importance()
    if fi:
        st.markdown("""
            <div class="section-label">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
                </svg>
                Feature Importance — Best Model
            </div>
        """, unsafe_allow_html=True)
        fi_colors = ['#2563EB','#7C3AED','#059669','#D97706','#DC2626','#0891B2','#0E7490','#9333EA','#15803D']
        fig_fi = go.Figure(go.Bar(
            x=list(fi.values()),
            y=list(fi.keys()),
            orientation='h',
            marker_color=fi_colors[:len(fi)],
            marker_line_width=0,
            text=[f"{v:.4f}" for v in fi.values()],
            textposition='outside',
            textfont=dict(color='#374151', size=10),
        ))
        fig_fi.update_layout(
            height=320,
            **{**PLOT_LAYOUT, 'margin': dict(l=10, r=80, t=10, b=10)},
            xaxis=dict(**AXIS_STYLE, title='Feature Importance Score'),
            yaxis=dict(**AXIS_STYLE),
            showlegend=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""
        <div class="section-label" style='margin-top:0.5rem;'>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
        </div>
    """, unsafe_allow_html=True)


    cv_comparison_html = "".join([f"""
    <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;'>
        <div style='width:8px; height:8px; background:{COLOR_MAP[n]}; border-radius:50%; flex-shrink:0;'></div>
        <div style='font-size:0.78rem; color:#374151; flex:1;'>{n}</div>
        <div style='font-size:0.78rem; font-weight:800; color:{COLOR_MAP[n]};'>{all_metrics[n]["CV F1 Mean"]:.3f}</div>
        <div style='background:#F3F4F6; border-radius:4px; height:8px; width:80px; overflow:hidden;'>
            <div style='height:100%; width:{all_metrics[n]["CV F1 Mean"]*100}%; background:{COLOR_MAP[n]}; border-radius:4px;'></div>
        </div>
    </div>""" for n in model_names])


elif nav == "Dataset Explorer":

    st.markdown("""
        <div class="main-header">
            <div style='display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;'>
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
                <div style='font-size:1.4rem; font-weight:800; color:white;'>Dataset Explorer — gsalc.csv</div>
            </div>
            <div style='font-size:0.875rem; color:rgba(255,255,255,0.7);'>
                90 exhaled breath samples · 6 VOC gases · 3 concentration levels · ~9000 sensor readings per sample
            </div>
        </div>
    """, unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        gas_filter = st.multiselect("Filter by Gas Type", options=df_raw['Gas Type'].unique().tolist(),
                                    default=df_raw['Gas Type'].unique().tolist())
    with f2:
        ppb_filter = st.multiselect("Filter by Concentration (PPB)",
                                    options=sorted(df_raw['Concentration (PPB)'].unique().tolist()),
                                    default=sorted(df_raw['Concentration (PPB)'].unique().tolist()))

    df_filtered = df_raw[
        df_raw['Gas Type'].isin(gas_filter) &
        df_raw['Concentration (PPB)'].isin(ppb_filter)
    ]

    st.markdown(f"<div style='font-size:0.78rem; color:#6B7280; margin-bottom:0.5rem; font-weight:600;'>Showing {len(df_filtered)} of {len(df_raw)} records</div>", unsafe_allow_html=True)
    st.dataframe(
        df_filtered.style.background_gradient(subset=['Sensor Mean', 'Signal Range'], cmap='Blues'),
        use_container_width=True, height=350
    )

    st.markdown("""
        <div class="section-label" style='margin-top:1rem;'>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
            VOC Oncological Risk Reference — Medical Literature Weights
        </div>
    """, unsafe_allow_html=True)

    voc_df = pd.DataFrame([
        {'Gas Type': v['label'], 'Risk Category': v['risk'], 'Risk Weight': v['weight'],
         'Clinical Significance': {
             'toluene':        'Strong carcinogen marker; highest lung cancer correlation in exhaled breath studies',
             'hexane':         'Elevated in lung cancer patients; industrial solvent with known pulmonary toxicity',
             'ethyl acetate': 'Associated with metabolic dysregulation in oncological profiles',
             'acetone':        'Elevated in cancer cell metabolism due to ketosis and lipid peroxidation',
             'isopropanol':    'Moderate correlation; associated with cellular oxidative stress',
             'ethanol':        'Lowest oncological specificity; common background VOC in breath',
         }.get(k, '')}
        for k, v in LUNG_CANCER_VOC_RISK.items()
    ]).sort_values('Risk Weight', ascending=False)

    st.dataframe(
        voc_df.style.background_gradient(subset=['Risk Weight'], cmap='Reds'),
        use_container_width=True, hide_index=True
    )


else:
    st.markdown("""
        <div class="main-header">
            <div style='display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;'>
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
                    <polyline points="9 22 9 12 15 12 15 22"/>
                </svg>
                <div style='font-size:1.4rem; font-weight:800; color:white;'>Hospital Department Control Matrix</div>
            </div>
            <div style='font-size:0.875rem; color:rgba(255,255,255,0.7);'>Real-time ward throughput and oncology department routing status</div>
        </div>
    """, unsafe_allow_html=True)

    dept_data = [
        ("Pulmonary Unit Ward",    "Active",   12, 85,  "#2563EB", "M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"),
        ("Thoracic Diagnostics",   "Alert",     8, 92,  "#D97706", "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0 1 12 2.944a11.955 11.955 0 0 1-8.618 3.04A12.02 12.02 0 0 0 3 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"),
        ("Central Pathology Lab",  "Active",   24, 34,  "#059669", "M19.428 15.428a2 2 0 0 0-1.022-.547l-2.387-.477a6 6 0 0 0-3.86.517l-.318.158a6 6 0 0 1-3.86.517L6.05 15.21a2 2 0 0 0-1.806.547M8 4h8l-1 1v5.172a2 2 0 0 0 .586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 0 0 9 10.172V5L8 4z"),
        ("Oncology MDT Clinic",    "Active",   15, 67,  "#7C3AED", "M17 20h5v-2a3 3 0 0 0-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 0 1 5.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 0 1 9.288 0M15 7a3 3 0 1 1-6 0 3 3 0 0 1 6 0z"),
        ("Radiology LDCT Suite",   "Alert",     6, 96,  "#DC2626", "M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2z"),
        ("Breath Analysis Lab",    "Active",   31, 48,  "#0891B2", "M9 3H5a2 2 0 0 0-2 2v4m6-6h10l4 4v10a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm3 3v4h4"),
    ]

    for i in range(0, len(dept_data), 3):
        dept_cols = st.columns(3)
        for col, (name, status, patients, load, color, icon_path) in zip(dept_cols, dept_data[i:i+3]):
            s_color = "#059669" if status == "Active" else "#D97706"
            s_bg    = "#ECFDF5" if status == "Active" else "#FFFBEB"
            with col:
                st.markdown(f"""
                    <div class="dept-card" style='border-left:4px solid {color};'>
                        <div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem;'>
                            <div style='display:flex; align-items:center; gap:0.6rem;'>
                                <div style='width:34px; height:34px; background:{color}15; border-radius:9px;
                                            display:flex; align-items:center; justify-content:center; flex-shrink:0;'>
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2">
                                        <path d="{icon_path}"/>
                                    </svg>
                                </div>
                                <div style='font-size:0.875rem; font-weight:700; color:#111827;'>{name}</div>
                            </div>
                            <div style='background:{s_bg}; border:1px solid {s_color}40; border-radius:6px;
                                        padding:0.2rem 0.55rem; font-size:0.65rem; font-weight:800; color:{s_color};
                                        white-space:nowrap;'>{status}</div>
                        </div>
                        <div style='font-size:0.82rem; color:#6B7280; margin-bottom:0.75rem;'>
                            Active Cases: <strong style='color:#111827;'>{patients}</strong>
                        </div>
                        <div style='display:flex; justify-content:space-between; margin-bottom:0.35rem;'>
                            <span style='font-size:0.7rem; color:#9CA3AF; font-weight:600;'>Capacity Load</span>
                            <span style='font-size:0.7rem; font-weight:800; color:{color};'>{load}%</span>
                        </div>
                        <div style='background:#F3F4F6; border-radius:6px; height:8px; overflow:hidden;'>
                            <div style='height:100%; width:{load}%; background:{color}; border-radius:6px;'></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("""
        <div style='margin-top:1.5rem; padding:1.25rem; background:white; border:1px solid #E2E8F0;
                    border-radius:12px; text-align:center; box-shadow:0 1px 4px rgba(0,0,0,0.04);'>
            <div style='font-size:0.78rem; color:#9CA3AF;'>
                DIBA Cognitive Health Networks &nbsp;·&nbsp; Exhaled VOC Lung Cancer Detection Platform &nbsp;·&nbsp; v3.1 Production
            </div>
        </div>
    """, unsafe_allow_html=True)