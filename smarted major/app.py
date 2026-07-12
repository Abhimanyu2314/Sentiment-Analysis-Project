import streamlit as st
import joblib
import nltk
import pandas as pd
import numpy as np
import time
import json
import random
import re
import os
from concurrent.futures import ThreadPoolExecutor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Set up elite wide-canvas interface configuration matching the presentation frame
st.set_page_config(
    page_title="TinySentiment - Enterprise Analytics Suite", 
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State Objects
if 'extended_audit_df' not in st.session_state:
    st.session_state.extended_audit_df = pd.DataFrame([
        {"Audit ID": 1, "Timestamp": "21:40:12", "Sequence Input": "WandaVision was an absolute masterpiece with incredible writing.", "Model Verdict": "POS", "Override Status": "Verified Accuracy", "POS Confidence %": 94.2, "Latency ms": 1.4},
        {"Audit ID": 2, "Timestamp": "21:42:05", "Sequence Input": "Supernatural had a terrible final season, completely ruined it.", "Model Verdict": "NEG", "Override Status": "Verified Accuracy", "POS Confidence %": 12.4, "Latency ms": 1.1}
    ])

if 'nav_workspace' not in st.session_state:
    st.session_state.nav_workspace = "Real-Time Engine"

if 'text_input_buffer' not in st.session_state:
    st.session_state.text_input_buffer = ""

if 'last_verdict_accent' not in st.session_state:
    st.session_state.last_verdict_accent = "#9f7aea"

# =========================================================================
# ULTRA-PREMIUM CINEMATIC DESIGN SYSTEM & KINETIC BLOOM ANIMATIONS
# =========================================================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    /* Premium Fluid Mesh Background Animation */
    @keyframes meshGradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    .stApp {{
        background: radial-gradient(circle at 90% 10%, rgba(31, 12, 64, 0.4) 0%, rgba(9, 5, 26, 1) 50%, rgba(4, 2, 10, 1) 100%) !important;
        background-size: 300% 300% !important;
        animation: meshGradient 20s ease infinite !important;
        color: #f3f0f9 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }}
    
    /* Scrollbar Artistry */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: #04020a; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(124, 58, 237, 0.2); border-radius: 20px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(139, 92, 246, 0.6); }}

    /* Cinematic Premium Bloom Transition Engine */
    @keyframes kineticBloomReveal {{
        0% {{ opacity: 0; transform: translateY(12px) scale(0.995); filter: blur(8px); }}
        100% {{ opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }}
    }}
    @keyframes pulseGlow {{
        0%, 100% {{ opacity: 0.6; box-shadow: 0 0 15px rgba(124, 58, 237, 0.1); }}
        50% {{ opacity: 1; box-shadow: 0 0 30px rgba(124, 58, 237, 0.3); }}
    }}
    @keyframes textShimmer {{
        0% {{ background-position: 0% 50%; }}
        100% {{ background-position: 100% 50%; }}
    }}

    /* Global smooth component interpolation binding */
    .element-container, .stButton, .media-card, .kpi-box, .premium-workspace-card {{
        animation: kineticBloomReveal 0.55s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    }}
    
    /* Premium Headers with Dynamic Light Sheen */
    h1, h2, h3, h4, .sidebar-brand {{
        font-family: 'Space Grotesk', sans-serif !important;
        background: linear-gradient(90deg, #ffffff 0%, #c3b4fc 50%, #ffffff 100%);
        background-size: 200% auto;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        animation: textShimmer 6s linear infinite;
        letter-spacing: -0.03em !important;
        font-weight: 700 !important;
    }}
    
    /* Sidebar Obsidian Overhaul */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(6, 3, 15, 0.98) 0%, rgba(2, 1, 5, 1) 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(30px);
    }}
    
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {{
        color: #e2e8f0 !important;
    }}
    
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        color: #f1edf7 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }}
    
    .sidebar-brand {{
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 3px;
        margin-bottom: 2.5rem;
        padding-left: 1rem;
    }}
    .sidebar-brand span {{ color: #a78bfa; -webkit-text-fill-color: initial !important; text-shadow: 0 0 15px rgba(167, 139, 250, 0.4); }}
    
    div[data-testid="stSidebarNav"] {{ display: none; }}
    
    /* Luxury Radio Group Toggles Visibility Fixes */
    div[data-testid="stRadio"] label {{
        padding: 14px 18px !important;
        border-radius: 16px;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        margin-bottom: 8px;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }}
    div[data-testid="stRadio"] label p {{
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        transition: color 0.3s ease;
    }}
    div[data-testid="stRadio"] label:hover {{
        background: rgba(255, 255, 255, 0.05) !important;
        transform: translateX(4px);
    }}
    div[data-testid="stRadio"] label:hover p {{
        color: #ffffff !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > div[data-checked="true"] label {{
        background: linear-gradient(90deg, rgba(124, 58, 237, 0.35) 0%, rgba(45, 20, 90, 0.4) 100%) !important;
        border: 1px solid rgba(167, 139, 250, 0.5) !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.25), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > div[data-checked="true"] label p {{
        color: #ffffff !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stRadio"] input[type="radio"] {{ display: none; }}
    
    /* Glassmorphic Widget Configurations */
    div[data-testid="stSlider"], div[data-testid="stExpander"], div[data-testid="stSelectbox"] > div {{
        background: rgba(13, 8, 28, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px);
    }}
    
    div[data-testid="stSlider"] div[role="slider"] {{
        background-color: #8b5cf6 !important;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.7) !important;
        border: 2px solid #ffffff !important;
    }}
    div[data-testid="stSlider"] div[data-testid="stSliderTrack"] > div {{
        background-color: #8b5cf6 !important;
    }}
    
    /* High-Fidelity Workspace Cards with Micro-Glow Transitions */
    .premium-workspace-card {{
        background: linear-gradient(145deg, rgba(18, 10, 38, 0.6) 0%, rgba(7, 3, 18, 0.9) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 24px !important;
        padding: 2.2rem !important;
        box-shadow: 0 40px 80px rgba(0,0,0,0.6), inset 0 1px 1px rgba(255, 255, 255, 0.08) !important;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(20px);
        transition: border-color 0.5s ease, box-shadow 0.5s ease;
    }}
    .premium-workspace-card:hover {{
        border-color: rgba(139, 92, 246, 0.25);
        box-shadow: 0 40px 80px rgba(0,0,0,0.65), 0 0 30px rgba(139, 92, 246, 0.05) !important;
    }}
    
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0px !important;
    }}
    
    .top-nav-bar {{
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 1rem;
    }}
    .top-nav-title {{ font-weight: 700; font-size: 1.35rem; color: #fff; letter-spacing: -0.5px; }}
    
    /* KPI Dashboards Layering */
    .kpi-container {{
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2.5rem;
        width: 100%;
    }}
    .kpi-box {{
        flex: 1;
        background: linear-gradient(135deg, rgba(22, 12, 51, 0.45) 0%, rgba(5, 2, 15, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 1.25rem 1.6rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.4s ease;
    }}
    .kpi-box:hover {{
        transform: translateY(-4px);
        border-color: rgba(139, 92, 246, 0.3);
    }}
    .kpi-label {{
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
        font-weight: 700;
    }}
    .kpi-value {{ font-size: 1.85rem; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; }}
    
    /* ==========================================================
       MEDIA CARD RESIZING & STRUCTURAL BALANCE CORRECTIONS
       ========================================================== */
    .media-card {{
        background: linear-gradient(135deg, rgba(16, 9, 33, 0.6) 0%, rgba(4, 2, 10, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 20px;
        padding: 1.5rem !important;
        margin-bottom: 1.25rem;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 20px 45px rgba(0,0,0,0.45);
        
        /* Enforces uniform layout sizing across multi-line wrapping fields */
        min-height: 165px !important; 
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }}
    .media-card h4 {{
        margin: 0 0 8px 0 !important;
        min-height: 52px !important; /* Resolves height offset differences caused by string wraps */
        display: flex !important;
        align-items: center !important;
    }}
    .media-card:hover {{
        transform: translateY(-5px) scale(1.01);
        border-color: rgba(139, 92, 246, 0.35);
        box-shadow: 0 25px 55px rgba(0,0,0,0.55), 0 0 25px rgba(139, 92, 246, 0.1);
    }}
    
    /* Kinetic Premium Action Buttons with Fixed Footprint Dimensions */
    div.stButton > button, div.stDownloadButton > button {{
        background: linear-gradient(90deg, #7c3aed 0%, #6d28d9 50%, #5b21b6 100%) !important;
        background-size: 200% auto !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 25px rgba(124, 58, 237, 0.2) !important;
        
        /* Strict sizing controls to keep layout buttons aligned on the horizon axis */
        height: 58px !important;
        padding: 0px 1.5rem !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    div.stButton > button div[data-testid="stMarkdownContainer"] p, 
    div.stDownloadButton > button div[data-testid="stMarkdownContainer"] p {{
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.3px;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.3 !important;
        text-align: center !important;
        white-space: normal !important; /* Permits beautiful internal wrapping inside bounded height */
    }}
    div.stButton > button:hover, div.stDownloadButton > button:hover {{
        background-position: right center !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 30px rgba(124, 58, 237, 0.45), 0 0 15px rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }}
    
    /* Input Fields */
    textarea {{
        background-color: rgba(3, 1, 8, 0.85) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        padding: 1.3rem !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }}
    textarea:focus {{
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.25) !important;
    }}
    
    div[data-testid="stBlock"] {{ padding: 0px !important; margin: 0px !important; }}
    
    /* Pulsing Cybernetic Loader Layout */
    .scanning-loader {{
        animation: pulseGlow 2s infinite ease-in-out;
        background: linear-gradient(90deg, #070412 0%, #1c0e38 50%, #070412 100%);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        color: #c3b4fc;
        font-weight: 600;
        border: 1px dashed rgba(139, 92, 246, 0.3);
    }}
    
    .premium-chart-card {{
        background: rgba(4, 2, 9, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 24px !important;
        padding: 2rem !important;
        margin-bottom: 1.5rem;
        box-shadow: 0 25px 55px rgba(0,0,0,0.5);
    }}
    
    div[data-testid="stDataEditor"] {{
        background-color: rgba(4, 2, 9, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        padding: 0.75rem !important;
    }}
    
    .valency-badge-row {{
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
        gap: 12px;
    }}
    .v-badge {{
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 12px 18px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 700;
        text-align: center;
        flex: 1;
        backdrop-filter: blur(5px);
    }}
    
    .status-dot {{
        height: 7px;
        width: 7px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        vertical-align: middle;
        box-shadow: 0 0 10px currentColor;
    }}
    
    #MainMenu, footer, header {{ visibility: hidden; }}
    div[data-testid="stFileUploader"] {{ 
        background-color: rgba(4, 2, 10, 0.6) !important; 
        border: 1px dashed rgba(139, 92, 246, 0.2) !important; 
        border-radius: 20px !important; 
        padding: 1rem;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# ASYNCHRONOUS ENGINE CONFIGURATION & FALLBACKS
# ==========================================
random.seed(42)

def clean_single_sequence(raw_text, negation_words, stop_words, lemmatizer):
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""
    try:
        words = word_tokenize(raw_text)
    except Exception:
        words = re.findall(r'\b\w+\b', raw_text)
        
    transformed_words, negate_counter = [], 0
    reset_words = {'.', ',', ';', '!', '?', 'but', 'and', 'however', 'although', 'yet', 'br', 'html'}
    
    for word in words:
        word_lower = word.lower()
        if word_lower in negation_words:
            negate_counter = 3
            transformed_words.append(word_lower)
        elif word_lower in reset_words or len(word_lower) > 15:
            negate_counter = 0
            transformed_words.append(word_lower)
        elif negate_counter > 0:
            transformed_words.append(f"not_{word_lower}")
            negate_counter -= 1
        else:
            transformed_words.append(word_lower)
            
    final_words = []
    for word in transformed_words:
        if word.startswith("not_"):
            base = word[4:]
            if base.isalpha():
                final_words.append(f"not_{lemmatizer.lemmatize(base)}")
        else:
            if word.isalpha() and word not in stop_words:
                final_words.append(lemmatizer.lemmatize(word))
    return ' '.join(final_words)

@st.cache_resource
def load_assets():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    log_model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    negation_words = {'not', 'no', 'never', 'neither', 'nor', 'but', 'without', 'against'}
    stop = set(stopwords.words('english')) - negation_words
    lem = WordNetLemmatizer()
    
    if not os.path.exists('svm_model.pkl'):
        if os.path.exists('my_dataset.csv'):
            df = pd.read_csv('my_dataset.csv')
            text_col = 'text' if 'text' in df.columns else 'review' if 'review' in df.columns else df.columns[0]
            label_col = 'sentiment' if 'sentiment' in df.columns else 'label' if 'label' in df.columns else df.columns[1]
            
            cleaned_texts = [clean_single_sequence(str(t), negation_words, stop, lem) for t in df[text_col]]
            X = vectorizer.transform(cleaned_texts)
            
            label_mapping = {'positive': 'pos', 'pos': 'pos', '1': 'pos', 1: 'pos', 'negative': 'neg', 'neg': 'neg', '0': 'neg', 0: 'neg'}
            y = [label_mapping.get(str(l).strip().lower(), 'neg') for l in df[label_col]]
            
            base_svm = LinearSVC(random_state=42, dual=False)
            svm_model = CalibratedClassifierCV(base_svm)
            svm_model.fit(X, y)
            joblib.dump(svm_model, 'svm_model.pkl')
        else:
            svm_model = log_model
    else:
        svm_model = joblib.load('svm_model.pkl')
        
    return log_model, svm_model, vectorizer, negation_words, stop, lem

model_log, model_svm, vectorizer, negation_words, stop_words, lemmatizer = load_assets()

# ==========================================
# SIDEBAR NAVIGATION & TUNING SWITCHER
# ==========================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">TINY<span>SENTIMENT</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color: #a78bfa; font-size: 0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin: 0 0 0.75rem 1rem;">Workspaces</div>', unsafe_allow_html=True)
    nav_select = st.radio(label="Nav", options=["Real-Time Engine", "Bulk Batch Processor"], label_visibility="collapsed")
    st.session_state.nav_workspace = nav_select
    
    st.markdown('<div style="color: #a78bfa; font-size: 0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin: 2.5rem 0 0.75rem 1rem;">Tuning Console</div>', unsafe_allow_html=True)
    confidence_threshold = st.slider(label="Certainty Cutoff Threshold (%)", min_value=50, max_value=100, value=55, step=5)
    
    active_engine = st.selectbox(
        label="🧬 Active Neural Backend Engine",
        options=["Sparse Linear Array (Logistic)", "High-Dimension Vector Core (SVM)"]
    )

classifier = model_log if active_engine == "Sparse Linear Array (Logistic)" else model_svm

# ==========================================
# GLOBAL WORKSPACE FRAGMENT INTERFACES
# ==========================================
@st.fragment
def render_realtime_engine():
    df_metrics = st.session_state.extended_audit_df
    total_ops = len(df_metrics)
    avg_latency = round(df_metrics["Latency ms"].mean(), 1) if total_ops > 0 else 0.0
    pos_count = len(df_metrics[df_metrics["Model Verdict"] == "POS"])
    valency_ratio = round((pos_count / total_ops * 100), 1) if total_ops > 0 else 0.0

    left_view, right_view = st.columns([3, 1.3], gap="large")
    with left_view:
        st.markdown('<div class="top-nav-bar"><span class="top-nav-title">🏠 Real-Time Analyzer Hub</span></div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-box"><div class="kpi-label">Total Operations</div><div class="kpi-value">{total_ops}</div></div>
                <div class="kpi-box"><div class="kpi-label">Mean Latency</div><div class="kpi-value">{avg_latency} ms</div></div>
                <div class="kpi-box"><div class="kpi-label">Positivity Valency</div><div class="kpi-value">{valency_ratio}%</div></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="premium-workspace-card">', unsafe_allow_html=True)
        st.markdown("<h2 style='margin:0; font-size: 2.1rem; font-weight:800; color:#fff;'>The Cognitive Spotlight</h2>", unsafe_allow_html=True)
        st.markdown("<p style='margin:4px 0 1.5rem 0; color:#94a3b8; font-size:0.88rem;'>Submit unstructured text sequences below to isolate raw token weights live.</p>", unsafe_allow_html=True)
        
        user_review = st.text_area(label="Inference Entry", label_visibility="collapsed", height=140, value=st.session_state.text_input_buffer, key="inference_text_area")
        st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
        
        btn_col1, btn_col2, _ = st.columns([1.5, 2.0, 3.5])
        with btn_col1:
            analyze_button = st.button("Analyze", key="exec_single_inference")
        with btn_col2:
            if st.button("Clear Console", key="flush_single_console"):
                st.session_state.text_input_buffer = ""
                st.utility_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<h3 style='font-size:1.15rem; font-weight:700; color:#fff; margin-top:2.5rem; margin-bottom:1rem;'>Popular Templates</h3>", unsafe_allow_html=True)
        card_col1, card_col2, card_col3 = st.columns(3, gap="medium")
        with card_col1:
            st.markdown('<div class="media-card"><h4>Loki</h4><p style="color:#a78bfa; font-size:0.85rem; margin:0;">⭐⭐⭐⭐⭐</p></div>', unsafe_allow_html=True)
            if st.button("Load Loki Review", use_container_width=True):
                st.session_state.text_input_buffer = "An absolute masterpiece! The chronological pacing was completely mind-blowing."
        with card_col2:
            st.markdown('<div class="media-card"><h4>Chernobyl</h4><p style="color:#a78bfa; font-size:0.85rem; margin:0;">⭐⭐⭐⭐⭐</p></div>', unsafe_allow_html=True)
            if st.button("Load Chernobyl Review", use_container_width=True):
                st.session_state.text_input_buffer = "A terrifyingly bleak experience. Not pleasant, but incredibly vital filmmaking."
        with card_col3:
            st.markdown('<div class="media-card"><h4>Rick & Morty</h4><p style="color:#a78bfa; font-size:0.85rem; margin:0;">⭐⭐❌❌❌</p></div>', unsafe_allow_html=True)
            if st.button("Load Rick Review", use_container_width=True):
                st.session_state.text_input_buffer = "The jokes are not bad, but the recent narrative choices feel exhausted."

    with right_view:
        st.markdown('<div class="top-nav-bar" style="visibility:hidden; min-height:61px;"></div>', unsafe_allow_html=True)
        
        current_text_to_process = user_review if analyze_button else st.session_state.text_input_buffer
        
        if (analyze_button and user_review.strip()) or (st.session_state.text_input_buffer and not analyze_button):
            loading_placeholder = st.empty()
            loading_placeholder.markdown('<div class="scanning-loader">🔮 Tracing sparse tensor alignments...</div>', unsafe_allow_html=True)
            start_perf = time.time()
            
            clean_review = clean_single_sequence(current_text_to_process, negation_words, stop_words, lemmatizer)
            review_vector = vectorizer.transform([clean_review])
            raw_prediction = classifier.predict(review_vector)[0]
            prob_scores = classifier.predict_proba(review_vector)[0]
            classes = list(classifier.classes_)
            pos_confidence = prob_scores[classes.index('pos')] * 100
            neg_confidence = prob_scores[classes.index('neg')] * 100
            latency_ms = (time.time() - start_perf) * 1000
            
            display_verdict = raw_prediction.upper()
            dot_color = "#a78bfa" if raw_prediction.lower() == 'pos' else "#7c3aed"
            if max(pos_confidence, neg_confidence) < confidence_threshold:
                display_verdict = "UNCERTAIN"
                dot_color = "#64748b"
            
            if analyze_button:
                new_id = len(st.session_state.extended_audit_df) + 1
                timestamp_str = time.strftime("%H:%M:%S")
                new_row = pd.DataFrame([{
                    "Audit ID": new_id, "Timestamp": timestamp_str, "Sequence Input": current_text_to_process, 
                    "Model Verdict": display_verdict, "Override Status": "Verified Accuracy",
                    "POS Confidence %": round(pos_confidence, 1), "Latency ms": round(latency_ms, 1)
                }])
                st.session_state.extended_audit_df = pd.concat([st.session_state.extended_audit_df, new_row], ignore_index=True)
            
            loading_placeholder.empty()
            
            st.markdown(f"""
                <div class="media-card" style="border-left: 4px solid {dot_color}; background: linear-gradient(135deg, rgba(30,15,65,0.4) 0%, rgba(5,2,12,0.9) 100%); min-height: unset !important; height: auto !important;">
                    <h4 style="margin:0; color:#fff; font-size:1.1rem; min-height: unset !important;"><span class="status-dot" style="background-color:{dot_color}; color:{dot_color};"></span>Verdict: {display_verdict}</h4>
                    <p style="color:#94a3b8; font-size:0.9rem; margin:8px 0 0 0; font-weight:500;">Confidence Score: {max(pos_confidence, neg_confidence):.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<h3 style='font-size:1.15rem; font-weight:700; color:#fff; margin-top:1.5rem;'>Historical Log Actions</h3>", unsafe_allow_html=True)
        if st.button("Reset Ledger", use_container_width=True):
            st.session_state.extended_audit_df = st.session_state.extended_audit_df.iloc[0:0]
            st.rerun()

    st.markdown("---")
    st.markdown("<h3 style='font-size:1.25rem; font-weight:700; color:#fff; margin-bottom:0.75rem;'>🔮 Interactive Ledger Validation Arrays</h3>", unsafe_allow_html=True)
    edited_df = st.data_editor(st.session_state.extended_audit_df, use_container_width=True)

# ==========================================
# WORKSPACE WORKER 2: BATCH INGESTION ENGINE
# ==========================================
@st.fragment
def render_bulk_engine():
    left_view, right_view = st.columns([2.5, 1.8], gap="large")
    
    with left_view:
        st.markdown('<div class="top-nav-bar"><span class="top-nav-title">🚀 Bulk Batch Pipeline Engine</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="premium-workspace-card">', unsafe_allow_html=True)
        st.markdown("<h2 style='margin:0; font-size: 2.1rem; font-weight:800; color:#fff;'>Batch Ingestion Lane</h2>", unsafe_allow_html=True)
        st.markdown("<p style='margin:4px 0 1.5rem 0; color:#94a3b8; font-size:0.88rem;'>Concurrently ingest datasets and tune linguistic cutoff filters below.</p>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(label="Upload CSV Target", label_visibility="collapsed", type=["csv"])
        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        length_bounds = st.slider(label="🔬 Linguistic Boundary Filter (Document Length Word Counts)", min_value=1, max_value=250, value=(1, 150))
        st.markdown('</div>', unsafe_allow_html=True)

    with right_view:
        st.markdown('<div class="top-nav-bar" style="visibility:hidden; min-height:61px;"></div>', unsafe_allow_html=True)
        if uploaded_file is not None:
            bulk_df = pd.read_csv(uploaded_file)
            text_col = 'text' if 'text' in bulk_df.columns else 'review' if 'review' in bulk_df.columns else bulk_df.columns[0]
            label_col = 'sentiment' if 'sentiment' in bulk_df.columns else 'label' if 'label' in bulk_df.columns else None
            
            total_raw_rows = len(bulk_df)
            start_bulk_time = time.time()
            raw_documents = bulk_df[text_col].astype(str).tolist()
            
            with ThreadPoolExecutor() as executor:
                cleaned_bulk_reviews = list(executor.map(
                    lambda doc: clean_single_sequence(doc, negation_words, stop_words, lemmatizer),
                    raw_documents
                ))
            
            document_word_counts = [len(doc.split()) for doc in cleaned_bulk_reviews]
            bulk_df['__word_count_internal__'] = document_word_counts
            
            lower_limit, upper_limit = length_bounds
            filtered_bulk_df = bulk_df[
                (bulk_df['__word_count_internal__'] >= lower_limit) & 
                (bulk_df['__word_count_internal__'] <= upper_limit)
            ].copy().reset_index(drop=True)
            
            total_rows = len(filtered_bulk_df)
            
            if total_rows > 0:
                X_bulk = vectorizer.transform(filtered_bulk_df[text_col].astype(str).tolist())
                bulk_predictions = classifier.predict(X_bulk)
                bulk_prob_scores = classifier.predict_proba(X_bulk)
                bulk_classes = list(classifier.classes_)
                bulk_latency_seconds = time.time() - start_bulk_time
                dps_throughput = total_rows / max(0.001, bulk_latency_seconds)
                
                final_batch_verdicts = []
                pos_idx = bulk_classes.index('pos')
                neg_idx = bulk_classes.index('neg')
                max_confidence_percentages = []
                
                for probs in bulk_prob_scores:
                    max_prob = max(probs) * 100
                    max_confidence_percentages.append(max_prob)
                    if max_prob < confidence_threshold:
                        final_batch_verdicts.append("UNCERTAIN")
                    else:
                        final_batch_verdicts.append(bulk_classes[np.argmax(probs)].upper())
                
                pos_count = sum(1 for v in final_batch_verdicts if v == 'POS')
                neg_count = sum(1 for v in final_batch_verdicts if v == 'NEG')
                unc_count = total_rows - (pos_count + neg_count)
                
                st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-box"><div class="kpi-label">Filtered Records</div><div class="kpi-value">{total_rows}</div></div>
                        <div class="kpi-box"><div class="kpi-label">Dropped Out-Of-Bounds</div><div class="kpi-value">{total_raw_rows - total_rows}</div></div>
                        <div class="kpi-box"><div class="kpi-label">Ingestion Throughput</div><div class="kpi-value">{int(dps_throughput)} /sec</div></div>
                    </div>
                    
                    <div class="media-card" style="min-height: unset !important; height: auto !important;">
                        <h4 style="margin:0; font-size:1.1rem; color:#fff; min-height: unset !important;">📊 Dataset Valency Proportions</h4>
                        <div class="valency-badge-row">
                            <div class="v-badge" style="border-color: rgba(167,139,250,0.3); color:#a78bfa;"><span class="status-dot" style="background:#a78bfa; color:#a78bfa;"></span>POS: {pos_count}</div>
                            <div class="v-badge" style="border-color: rgba(124,58,237,0.3); color:#7c3aed;"><span class="status-dot" style="background:#7c3aed; color:#7c3aed;"></span>NEG: {neg_count}</div>
                            <div class="v-badge" style="border-color: rgba(100,116,139,0.3); color:#94a3b8;"><span class="status-dot" style="background:#64748b; color:#64748b;"></span>UNC: {unc_count}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="premium-chart-card">', unsafe_allow_html=True)
                st.markdown("<h4 style='margin:0 0 1.25rem 0; font-size:1.05rem; color:#fff;'>📐 Document Word Length vs Model Certainty Drift</h4>", unsafe_allow_html=True)
                drift_df = pd.DataFrame({
                    'Word Count Length': filtered_bulk_df['__word_count_internal__'].tolist(), 
                    'Model Certainty Index (%)': max_confidence_percentages
                })
                st.scatter_chart(drift_df, x='Word Count Length', y='Model Certainty Index (%)', color="#8b5cf6")
                st.markdown('</div>', unsafe_allow_html=True)
                
                filtered_bulk_df['Model_Ternary_Verdict'] = final_batch_verdicts
                filtered_bulk_df['Favorable_Probability_%'] = bulk_prob_scores[:, pos_idx] * 100
                
                csv_export = filtered_bulk_df.drop(columns=['__word_count_internal__']).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Annotated Sentiment Ingestion Data (.CSV)",
                    data=csv_export,
                    file_name="bulk_sentiment_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error("⚠️ Boundary Constraint Error: No documents found inside selected limits.")

# Routing core dashboard workspace panels
if st.session_state.nav_workspace == "Real-Time Engine":
    render_realtime_engine()
else:
    render_bulk_engine()