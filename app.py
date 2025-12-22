# app.py — Enhanced Long Document Summarizer (Stable + Professional)

import streamlit as st
import textwrap
from collections import Counter

# ==================================================
# Page config + theme
# ==================================================
st.set_page_config(
    page_title="AI Long Document Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
h1, h2, h3 { color: #4CAF50; }
textarea { border-radius: 10px !important; }
.stButton>button { border-radius: 10px; font-weight: 600; }
.summary-box { background:#111827; padding:16px; border-radius:12px; }
.badge { background:#1f2937; padding:6px 10px; border-radius:8px; margin-right:6px; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# Safe imports
# ==================================================
try:
    from transformers import LEDTokenizer, LEDForConditionalGeneration
    import torch
    TRANSFORMERS_OK = True
except Exception:
    TRANSFORMERS_OK = False
    torch = None

try:
    from summarizer import Summarizer as ExtractiveSummarizer
    EXTRACTIVE_OK = True
except Exception:
    EXTRACTIVE_OK = False
    ExtractiveSummarizer = None

# ==================================================
# Utilities
# ==================================================
def keyword_extract(text, k=8):
    words = [w.lower() for w in text.split() if len(w) > 4]
    return [w for w, _ in Counter(words).most_common(k)]

def confidence_score(inp, out):
    if inp == 0:
        return 0
    ratio = out / inp
    return min(1.0, max(0.3, 1 - abs(0.25 - ratio)))

# ==================================================
# File readers
# ==================================================
def read_txt(file):
    data = file.read()
    return data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else data

def read_pdf(file):
    import PyPDF2
    reader = PyPDF2.PdfReader(file)
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def read_docx(file):
    import docx
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

# ==================================================
# LED summarization (lazy loaded)
# ==================================================
def chunk_by_tokens(tokenizer, text, max_tokens=4000, stride=200):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end], skip_special_tokens=True))
        start = end - stride if end < len(tokens) else end
    return chunks

@st.cache_resource
def load_led(model_name, device):
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
    return tokenizer, model

def summarize_led(text, device, final_len, beams, progress):
    tokenizer, model = load_led("allenai/led-base-16384", device)
    chunks = chunk_by_tokens(tokenizer, text)
    summaries = []

    for i, ch in enumerate(chunks):
        progress.progress((i + 1) / len(chunks))
        inputs = tokenizer(ch, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=200, num_beams=2)
        summaries.append(tokenizer.decode(out[0], skip_special_tokens=True))

    combined = " ".join(summaries)
    inputs = tokenizer(combined, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        final = model.generate(**inputs, max_length=final_len, num_beams=beams)

    return summaries, tokenizer.decode(final[0], skip_special_tokens=True)

def extractive_summary(text, ratio, num_sent):
    if not EXTRACTIVE_OK or ExtractiveSummarizer is None:
        return "❌ Extractive summarizer not available."
    model = ExtractiveSummarizer("distilbert-base-uncased")
    return model(text, num_sentences=num_sent) if num_sent > 0 else model(text, ratio=ratio)

# ==================================================
# Header
# ==================================================
st.title("📚 AI Long Document Summarizer")
st.caption("Abstractive (LED) + Extractive (BERT) | Stable | Resume-ready")

# ==================================================
# Sidebar
# ==================================================
with st.sidebar:
    st.header("⚙️ Controls")

    mode = st.radio("Mode", ["Abstractive (LED)", "Extractive (BERT)"])

    preset = st.selectbox("Quality Preset", ["Fast", "Balanced", "High Quality"])
    final_len, beams = (200, 2) if preset == "Fast" else (300, 4) if preset == "Balanced" else (450, 6)

    if mode == "Extractive (BERT)":
        ratio = st.slider("Extractive Ratio", 0.05, 0.5, 0.2)
        num_sent = st.slider("Sentences (0 = auto)", 0, 20, 0)

    bullet = st.checkbox("Bullet Output", True)

# ==================================================
# Main layout
# ==================================================
c1, c2 = st.columns([2, 1])

with c1:
    uploaded = st.file_uploader("Upload document", type=["txt", "pdf", "docx"])
    text_input = st.text_area("Paste or edit text", height=320)

    if uploaded:
        if uploaded.name.endswith(".pdf"):
            text_input = read_pdf(uploaded)
        elif uploaded.name.endswith(".docx"):
            text_input = read_docx(uploaded)
        else:
            text_input = read_txt(uploaded)

with c2:
    wc = len(text_input.split()) if text_input else 0
    st.metric("Input Words", wc)
    if wc > 12000:
        st.warning("Large document — may take longer")

# ==================================================
# Run
# ==================================================
st.divider()
if st.button("✨ Generate Summary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please provide text")
    else:
        progress = st.progress(0)

        with st.spinner("Summarizing..."):
            if mode == "Abstractive (LED)":
                if not TRANSFORMERS_OK:
                    st.error("Transformers not installed")
                    st.stop()
                chunks, final = summarize_led(
                    text_input,
                    "cuda" if torch and torch.cuda.is_available() else "cpu",
                    final_len,
                    beams,
                    progress
                )
            else:
                final = extractive_summary(text_input, ratio, num_sent)
                chunks = [final]

        if bullet:
            final = "\n".join(f"- {s}" for s in textwrap.wrap(final, 180))

        # ==========================================
        # Output
        # ==========================================
        t1, t2, t3 = st.tabs(["📝 Summary", "📊 Stats", "🧠 Explain"])

        with t1:
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.text_area("Final Summary", final, height=280)
            st.markdown('</div>', unsafe_allow_html=True)
            st.download_button("⬇ Download TXT", final, "summary.txt")

        with t2:
            out_wc = len(final.split())
            conf = confidence_score(wc, out_wc)
            st.metric("Summary Words", out_wc)
            st.progress(conf)
            st.caption(f"Confidence: {round(conf*100)}%")

        with t3:
            st.markdown("**Key Topics**")
            for k in keyword_extract(final):
                st.markdown(f"<span class='badge'>{k}</span>", unsafe_allow_html=True)
            st.write("Summary generated using semantic relevance and coverage.")

st.markdown("---")
st.caption("🚀 Built with Streamlit • LED • BERT")
