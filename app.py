import streamlit as st
import torch
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import your trained model from inference.py
from inference import SentimentPredictor

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelAI · Multimodal Sentiment",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS – full premium skin
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── App background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0d1b3e 0%, #050d1f 55%, #000000 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f25 0%, #030916 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.12);
}
[data-testid="stSidebar"] * { color: #c8d6f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #a3b8d8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #63b3ed !important; font-weight: 700 !important; }

/* ── Hero title ── */
.hero-wrap {
    background: linear-gradient(135deg, rgba(99,179,237,0.06) 0%, rgba(129,140,248,0.06) 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 20px;
    padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(99,179,237,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 0.4rem;
}
.hero-title {
    font-size: clamp(1.8rem, 3.5vw, 3rem);
    font-weight: 900;
    line-height: 1.15;
    background: linear-gradient(110deg, #ffffff 30%, #63b3ed 65%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.6rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #7a96bc;
    font-weight: 400;
    max-width: 680px;
    line-height: 1.6;
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-blue  { background: rgba(99,179,237,0.12); color: #63b3ed; border: 1px solid rgba(99,179,237,0.25); }
.badge-purple{ background: rgba(129,140,248,0.12); color: #818cf8; border: 1px solid rgba(129,140,248,0.25); }
.badge-green { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }

/* ── Panel cards ── */
.panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
}
.panel-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Section subheaders ── */
.stApp h2, .stApp h3 {
    color: #dce8f5 !important;
    font-weight: 700 !important;
}

/* ── Text areas & inputs ── */
.stTextArea textarea, .stTextInput input {
    background: rgba(15,25,55,0.7) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    color: #dce8f5 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(99,179,237,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.1) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(15,25,55,0.5) !important;
    border: 1px dashed rgba(99,179,237,0.3) !important;
    border-radius: 12px !important;
}

/* ── Primary button – fully overridden ── */
.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.75rem 1.5rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 24px rgba(59,130,246,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(59,130,246,0.5) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

/* ── Secondary button ── */
.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #a3b8d8 !important;
    font-weight: 500 !important;
}

/* ── Verdict banner ── */
.verdict-banner {
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.verdict-banner.negative {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.06) 100%);
    border: 1px solid rgba(239,68,68,0.3);
}
.verdict-banner.positive {
    background: linear-gradient(135deg, rgba(52,211,153,0.12) 0%, rgba(16,185,129,0.06) 100%);
    border: 1px solid rgba(52,211,153,0.3);
}
.verdict-banner.neutral {
    background: linear-gradient(135deg, rgba(148,163,184,0.12) 0%, rgba(100,116,139,0.06) 100%);
    border: 1px solid rgba(148,163,184,0.3);
}
.verdict-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.3rem;
}
.verdict-word {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1;
}
.verdict-banner.negative .verdict-word { color: #f87171; }
.verdict-banner.positive .verdict-word { color: #34d399; }
.verdict-banner.neutral  .verdict-word { color: #94a3b8; }
.verdict-banner.negative .verdict-label { color: #f87171; }
.verdict-banner.positive .verdict-label { color: #34d399; }
.verdict-banner.neutral  .verdict-label { color: #94a3b8; }

/* ── Confidence bars ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.9rem;
}
.conf-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #a3b8d8;
    width: 72px;
    flex-shrink: 0;
}
.conf-track {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease;
}
.conf-fill-neg  { background: linear-gradient(90deg, #ef4444, #f87171); }
.conf-fill-neu  { background: linear-gradient(90deg, #64748b, #94a3b8); }
.conf-fill-pos  { background: linear-gradient(90deg, #10b981, #34d399); }
.conf-pct {
    font-size: 0.78rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #dce8f5;
    width: 46px;
    text-align: right;
    flex-shrink: 0;
}

/* ── XAI reasoning card ── */
.xai-card {
    background: rgba(99,179,237,0.05);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    color: #b0c8e4;
    font-size: 0.88rem;
    line-height: 1.7;
}
.xai-card strong { color: #dce8f5; }
.xai-card em { color: #93c5fd; font-style: normal; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(99,179,237,0.06) !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
    border-radius: 10px !important;
    color: #63b3ed !important;
    font-weight: 600 !important;
}

/* ── Spinners ── */
.stSpinner > div { border-color: #3b82f6 transparent transparent !important; }

/* ── Metric overrides ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
}
div[data-testid="metric-container"] label { color: #7a96bc !important; font-size: 0.75rem !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #dce8f5 !important; font-weight: 700 !important; }

/* ── selectbox / radio ── */
.stRadio > div { gap: 0.4rem; }
.stRadio label { color: #a3b8d8 !important; font-size: 0.88rem !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Success / Warning / Error ── */
.stSuccess  { background: rgba(52,211,153,0.08)  !important; border: 1px solid rgba(52,211,153,0.25)  !important; border-radius: 10px !important; color: #6ee7b7 !important; }
.stWarning  { background: rgba(251,191,36,0.08)  !important; border: 1px solid rgba(251,191,36,0.25)  !important; border-radius: 10px !important; color: #fcd34d !important; }
.stError    { background: rgba(239,68,68,0.08)   !important; border: 1px solid rgba(239,68,68,0.25)   !important; border-radius: 10px !important; color: #fca5a5 !important; }
.stInfo     { background: rgba(99,179,237,0.08)  !important; border: 1px solid rgba(99,179,237,0.25)  !important; border-radius: 10px !important; color: #93c5fd !important; }

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid rgba(99,179,237,0.15);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.25); border-radius: 999px; }

/* ── Pipeline steps ── */
.pipeline {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-top: 0.8rem;
}
.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.55rem 0.8rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.025);
    border-left: 3px solid #3b82f6;
}
.pipeline-step-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.pipeline-step-text { font-size: 0.8rem; color: #a3b8d8; line-height: 1.4; }
.pipeline-step-text strong { color: #dce8f5; font-size: 0.83rem; }

/* Large column padding */
.block-container { padding: 2rem 2.5rem 4rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CACHED MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ai():
    return SentimentPredictor("dataset/features/multimodal_sentiment_model.pth")

@st.cache_resource(show_spinner=False)
def load_vader():
    return SentimentIntensityAnalyzer()


# ─────────────────────────────────────────────
#  XAI ENGINE
# ─────────────────────────────────────────────
def generate_explanation(text, probabilities, final_verdict, vader_model):
    words = text.split()
    pos_words, neg_words = [], []
    for word in words:
        score = vader_model.polarity_scores(word)['compound']
        if score > 0.15:  pos_words.append(word)
        elif score < -0.15: neg_words.append(word)
    pos_words = list(set([w.strip('.,!?"\'()') for w in pos_words]))
    neg_words = list(set([w.strip('.,!?"\'()') for w in neg_words]))

    neg_pct = probabilities[0] * 100
    neu_pct = probabilities[1] * 100
    pos_pct = probabilities[2] * 100

    lines = [f"**Overall Verdict:** The fusion layer converged on **{final_verdict}**.\n"]

    lines.append(f"🔴 **Why {neg_pct:.1f}% Negative?**")
    if neg_words:
        lines.append(f"* **NLP Branch:** Negative textual anchors detected — *{', '.join(neg_words[:4])}*.")
    else:
        lines.append("* **NLP Branch:** No strong negative keywords found.")
    if neg_pct > 50:
        lines.append("* **Vision Branch:** ResNet-50 visual branch dominated — likely identified conflict-correlated features (military uniforms, ruins, fire).\n")
    else:
        lines.append("* **Vision Branch:** Visual conflict markers were weak, suppressing the negative score.\n")

    lines.append(f"🟢 **Why {pos_pct:.1f}% Positive?**")
    if pos_words:
        lines.append(f"* **NLP Branch:** Positive/hopeful phrasing detected — *{', '.join(pos_words[:4])}*.")
    else:
        lines.append("* **NLP Branch:** Minimal positive linguistics found.")
    if pos_pct > 50:
        lines.append("* **Vision Branch:** Image lacked aggressive conflict markers — positive text dictated the fusion output.\n")
    elif pos_words and pos_pct < 50:
        lines.append("* **Vision Branch:** Positive text suppressed — visual branch flagged the image as OOD or geopolitically tensioned (Modality Dominance).\n")
    else:
        lines.append("* **Vision Branch:** No compelling positive visual evidence found.\n")

    lines.append(f"⚪ **Why {neu_pct:.1f}% Neutral?**")
    lines.append("* **Baseline Math:** Probability of purely objective reporting.")
    if neu_pct < 10:
        lines.append("* Trained on polarized war data, the model actively penalises neutrality — mathematically negligible.")
    else:
        lines.append("* The emotional charge of text & image was statistically balanced.")

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  SCRAPERS
# ─────────────────────────────────────────────
def fetch_image_safely(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content))
    except Exception:
        pass
    return None

def scrape_news(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(r.content, 'html.parser')
        main = soup.find('article') or soup.find('main') or soup
        paras = [p.get_text().strip() for p in main.find_all('p') if len(p.get_text().strip()) > 60]
        text = " ".join(paras[:3])
        img_meta = soup.find("meta", property="og:image")
        img_url = img_meta["content"] if img_meta else None
        return text, img_url
    except Exception:
        return None, None


# ─────────────────────────────────────────────
#  ░░  SIDEBAR  ░░
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1rem;'>
        <div style='font-size:1.4rem; font-weight:800; color:#63b3ed; letter-spacing:-0.02em;'>🔮 SentinelAI</div>
        <div style='font-size:0.72rem; color:#4a6585; font-weight:500; letter-spacing:0.08em; text-transform:uppercase; margin-top:2px;'>Multimodal Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Data Source**")
    data_source = st.radio(
        "Source",
        ["🐦  Twitter / X (Manual)", "📰  News Article (URL)"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin:1.2rem 0;'>", unsafe_allow_html=True)

    st.markdown("**🏗️ Model Architecture**")
    st.markdown("""
    <div class='pipeline'>
        <div class='pipeline-step'>
            <div class='pipeline-step-icon'>📝</div>
            <div class='pipeline-step-text'><strong>NLP Branch</strong><br>DistilBERT · 768-dim CLS token</div>
        </div>
        <div class='pipeline-step'>
            <div class='pipeline-step-icon'>🖼️</div>
            <div class='pipeline-step-text'><strong>Vision Branch</strong><br>ResNet-50 · 2048-dim global pool</div>
        </div>
        <div class='pipeline-step'>
            <div class='pipeline-step-icon'>⚡</div>
            <div class='pipeline-step-text'><strong>Fusion Layer</strong><br>Mid-level concatenation → 512-dim</div>
        </div>
        <div class='pipeline-step'>
            <div class='pipeline-step-icon'>🎯</div>
            <div class='pipeline-step-text'><strong>Classifier</strong><br>2-layer MLP → 3 classes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#374e6b; line-height:1.6;'>
    🔬 Framework: <span style='color:#63b3ed;'>PyTorch</span><br>
    🧪 Training: <span style='color:#63b3ed;'>8 epochs · Adam · L2 reg</span><br>
    🏷️ Labels: <span style='color:#63b3ed;'>VADER auto-labeling</span><br>
    📊 Classes: <span style='color:#ef4444;'>Negative</span> · <span style='color:#94a3b8;'>Neutral</span> · <span style='color:#34d399;'>Positive</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ░░  HERO  ░░
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-eyebrow'>🔮 AI-Powered · Geopolitical Intelligence</div>
    <div class='hero-title'>Multimodal Sentiment Analysis</div>
    <div class='hero-sub'>
        Decode the emotional signature of geopolitical discourse by fusing
        <strong style='color:#93c5fd;'>Computer Vision (ResNet-50)</strong> with
        <strong style='color:#93c5fd;'>Natural Language Processing (DistilBERT)</strong>
        into a unified prediction engine.
    </div>
    <div class='hero-badges'>
        <span class='badge badge-blue'>🧠 Deep Learning</span>
        <span class='badge badge-purple'>🔗 Multimodal Fusion</span>
        <span class='badge badge-green'>🌐 Web-Scraped Data</span>
        <span class='badge badge-blue'>📡 Real-time Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ░░  MAIN COLUMNS  ░░
# ─────────────────────────────────────────────
col_in, col_out = st.columns([1, 1.25], gap="large")

# ─── LEFT: INPUT ────────────────────────────
with col_in:
    st.markdown("<div class='panel-title'>📥 Input Data</div>", unsafe_allow_html=True)

    text_input    = ""
    img_path      = None
    display_image = None

    is_twitter = "Twitter" in data_source

    if is_twitter:
        text_input = st.text_area(
            "Tweet / Post Text",
            height=140,
            placeholder="🚨 Breaking: Medical supplies have reached the border crossing…",
        )

        st.markdown(
            "<div style='font-size:0.78rem; color:#63b3ed; font-weight:600; "
            "letter-spacing:0.08em; text-transform:uppercase; margin:0.9rem 0 0.4rem;'>🖼️ Image Input</div>",
            unsafe_allow_html=True,
        )
        st.info("📎 **Tip:** Upload, drag & drop, or paste an image URL below.")

        uploaded_file = st.file_uploader(
            "Upload Image",
            type=['jpg', 'png', 'jpeg', 'webp'],
            label_visibility="collapsed",
        )
        tweet_img_url = st.text_input(
            "Image URL",
            placeholder="https://pbs.twimg.com/media/example.jpg",
        )

        if uploaded_file:
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption="Uploaded Image", use_container_width=True)
            img_path = "temp_image.jpg"
            display_image.convert('RGB').save(img_path)

        elif tweet_img_url:
            fetched = fetch_image_safely(tweet_img_url)
            if fetched:
                display_image = fetched
                st.image(display_image, caption="Fetched from URL", use_container_width=True)
                img_path = "temp_image.jpg"
                display_image.convert('RGB').save(img_path)
            else:
                st.error("⚠️ Could not fetch image — server may be blocking requests. Try uploading directly.")

    else:  # News URL
        news_url = st.text_input(
            "News Article URL",
            placeholder="https://www.aljazeera.com/…",
        )
        if news_url:
            with st.spinner("Scraping article and extracting lead image…"):
                scraped_text, scraped_img_url = scrape_news(news_url)

            if scraped_text and scraped_img_url:
                text_input = scraped_text
                st.success("✅ Article successfully extracted!")
                st.text_area("Extracted Text", value=text_input, height=130, disabled=True)
                fetched_news = fetch_image_safely(scraped_img_url)
                if fetched_news:
                    display_image = fetched_news
                    st.image(display_image, caption="Lead Image", use_container_width=True)
                    img_path = "temp_image.jpg"
                    display_image.convert('RGB').save(img_path)
                else:
                    st.error("⚠️ Could not load image from article. Security restrictions apply.")
            else:
                st.error("❌ Failed to extract content from this URL. Try a different article.")

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    analyze_button = st.button("🧠 Analyse Sentiment", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)


# ─── RIGHT: RESULTS ─────────────────────────
with col_out:
    st.markdown("<div class='panel-title'>📊 AI Analysis Dashboard</div>", unsafe_allow_html=True)

    if analyze_button and text_input and img_path:

        with st.spinner("⚙️ Initialising AI Brain — first run may take a moment…"):
            predictor = load_ai()
            vader     = load_vader()

        with st.spinner("🔬 Neural networks are fusing modalities…"):
            with torch.no_grad():
                inputs    = predictor.tokenizer(text_input, return_tensors="pt",
                                                truncation=True, padding='max_length', max_length=128)
                text_feat = predictor.text_model(**inputs).last_hidden_state[:, 0, :].squeeze().unsqueeze(0)

                img_tensor = predictor.img_transforms(Image.open(img_path).convert('RGB')).unsqueeze(0)
                img_feat   = predictor.img_model(img_tensor).squeeze().unsqueeze(0)

                output        = predictor.model(text_feat, img_feat)
                probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().numpy()
                predicted_idx = torch.argmax(output).item()

        labels        = ["Negative", "Neutral", "Positive"]
        final_verdict = labels[predicted_idx]
        css_class     = final_verdict.lower()

        # ── Verdict banner ──
        st.markdown(f"""
        <div class='verdict-banner {css_class}'>
            <div class='verdict-label'>AI Verdict</div>
            <div class='verdict-word'>{final_verdict.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence breakdown ──
        st.markdown("<div class='panel-title' style='margin-top:0.2rem;'>📈 Confidence Breakdown</div>",
                    unsafe_allow_html=True)

        neg_pct = probabilities[0] * 100
        neu_pct = probabilities[1] * 100
        pos_pct = probabilities[2] * 100

        st.markdown(f"""
        <div style='margin-bottom:0.5rem;'>
            <div class='conf-row'>
                <div class='conf-label'>🔴 Negative</div>
                <div class='conf-track'><div class='conf-fill conf-fill-neg' style='width:{neg_pct:.1f}%'></div></div>
                <div class='conf-pct'>{neg_pct:.1f}%</div>
            </div>
            <div class='conf-row'>
                <div class='conf-label'>⚪ Neutral</div>
                <div class='conf-track'><div class='conf-fill conf-fill-neu' style='width:{neu_pct:.1f}%'></div></div>
                <div class='conf-pct'>{neu_pct:.1f}%</div>
            </div>
            <div class='conf-row'>
                <div class='conf-label'>🟢 Positive</div>
                <div class='conf-track'><div class='conf-fill conf-fill-pos' style='width:{pos_pct:.1f}%'></div></div>
                <div class='conf-pct'>{pos_pct:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── XAI ──
        st.markdown("<div class='panel-title'>🧠 Explainable AI (XAI) Insight</div>",
                    unsafe_allow_html=True)

        with st.expander("🔍 View Architectural Reasoning", expanded=True):
            xai_text = generate_explanation(text_input, probabilities, final_verdict, vader)
            st.markdown(f"<div class='xai-card'>{xai_text}</div>", unsafe_allow_html=True)

        # Cleanup temp file
        if os.path.exists("temp_image.jpg"):
            os.remove("temp_image.jpg")

    elif analyze_button:
        st.warning("⚠️ Please provide **both** text and an image (upload, URL, or article) before analysing.")

    else:
        st.markdown("""
        <div style='
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            height:340px; text-align:center; gap:1rem;
            background: rgba(255,255,255,0.018);
            border: 1px dashed rgba(99,179,237,0.2);
            border-radius: 16px;
        '>
            <div style='font-size:3.5rem; opacity:0.4;'>🔮</div>
            <div style='color:#3a5070; font-size:1rem; font-weight:600;'>Awaiting input</div>
            <div style='color:#2a3d55; font-size:0.82rem; max-width:260px; line-height:1.5;'>
                Provide text + image on the left panel and click <strong style='color:#3b82f6;'>Analyse Sentiment</strong>
                to launch the neural fusion engine.
            </div>
        </div>
        """, unsafe_allow_html=True)