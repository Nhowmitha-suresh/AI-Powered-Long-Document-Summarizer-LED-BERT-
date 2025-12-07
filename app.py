# app.py
import streamlit as st
import os
import io
import textwrap
from functools import partial

# Try to import heavy NLP libraries, otherwise show friendly messages and fallback behavior
try:
    from transformers import LEDTokenizer, LEDForConditionalGeneration
    import torch
    TRANSFORMERS_OK = True
except Exception as e:
    TRANSFORMERS_OK = False
    LEDTokenizer = None
    LEDForConditionalGeneration = None
    torch = None

try:
    from summarizer import Summarizer as ExtractiveSummarizer
    EXTRACTIVE_OK = True
except Exception:
    EXTRACTIVE_OK = False
    ExtractiveSummarizer = None

# File reading helpers (PDF, DOCX, CSV)
def read_txt(file) -> str:
    data = file.read()
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8")
        except:
            return data.decode("latin-1", errors="ignore")
    return data

def read_pdf(file) -> str:
    try:
        import PyPDF2
    except Exception:
        st.error("PyPDF2 not installed. Install with: pip install PyPDF2")
        return ""
    reader = PyPDF2.PdfReader(file)
    pages = []
    for p in range(len(reader.pages)):
        try:
            pages.append(reader.pages[p].extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def read_docx(file) -> str:
    try:
        import docx
    except Exception:
        st.error("python-docx not installed. Install with: pip install python-docx")
        return ""
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def read_csv_excel(file, filename) -> str:
    try:
        import pandas as pd
    except Exception:
        st.error("pandas not installed. Install with: pip install pandas openpyxl")
        return ""
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    # If there is a text-like column, try to pick it; otherwise convert all text columns
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_cols:
        # fallback: convert whole df to string
        return df.to_string()
    # join top N rows from text columns
    rows = []
    for i, row in df.iterrows():
        pieces = []
        for c in text_cols:
            val = row.get(c, "")
            if isinstance(val, float) and st.experimental_get_query_params():  # dummy check to silence linter
                pass
            if not (val is None or (isinstance(val, float) and str(val) == "nan")):
                pieces.append(str(val))
        if pieces:
            rows.append(" ".join(pieces))
        if len(rows) >= 5000:  # prevent too-large previews
            break
    return "\n\n".join(rows)

# Token-based chunker (safe for long documents)
def chunk_by_tokens(tokenizer, text, max_tokens=5000, stride=200):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    L = len(tokens)
    while start < L:
        end = min(start + max_tokens, L)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        if end == L:
            break
        start = end - stride
    return chunks

# Summarize a list of chunks using model.generate (batched)
def summarize_chunks_led(model, tokenizer, chunks, device="cpu", batch_size=1, gen_kwargs=None):
    gen_kwargs = gen_kwargs or {}
    model.to(device)
    model.eval()
    summaries = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding="longest", max_length=tokenizer.model_max_length)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        for seq in out:
            summaries.append(tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    return summaries

# High-level hierarchical pipeline
def hierarchical_led_summarize(text, model_name="allenai/led-base-16384", device="cpu",
                               chunk_tokens=5000, chunk_stride=200, batch_size=1,
                               chunk_gen_kwargs=None, final_gen_kwargs=None, progress=None):
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    model = LEDForConditionalGeneration.from_pretrained(model_name)
    chunks = chunk_by_tokens(tokenizer, text, max_tokens=chunk_tokens, stride=chunk_stride)
    if progress:
        progress.text(f"Chunked into {len(chunks)} piece(s). Summarizing chunks...")
        pbar = st.progress(0)
    if not chunk_gen_kwargs:
        chunk_gen_kwargs = {"num_beams": 2, "max_length": 200, "min_length": 40, "length_penalty": 2.0, "early_stopping": True}
    chunk_summaries = []
    for i, ch in enumerate(chunks):
        if progress:
            progress.text(f"Summarizing chunk {i+1}/{len(chunks)}")
            pbar.progress(int((i/len(chunks))*100))
        s = summarize_chunks_led(model, tokenizer, [ch], device=device, batch_size=1, gen_kwargs=chunk_gen_kwargs)
        chunk_summaries.append(s[0] if s else "")
    if progress:
        pbar.progress(100)
        progress.text("Combining chunk summaries and generating final summary...")
    combined = "\n\n".join(chunk_summaries)
    if not final_gen_kwargs:
        final_gen_kwargs = {"num_beams": 4, "max_length": 300, "min_length": 80, "length_penalty": 2.0, "early_stopping": True}
    final = summarize_chunks_led(model, tokenizer, [combined], device=device, batch_size=1, gen_kwargs=final_gen_kwargs)
    if progress:
        progress.text("Done.")
    return {
        "chunks": chunks,
        "chunk_summaries": chunk_summaries,
        "final_summary": final[0] if final else ""
    }

# Small extractive helper using bert-extractive-summarizer
def extractive_summary(text, model_name="distilbert-base-uncased", ratio=0.1, num_sentences=None):
    if not EXTRACTIVE_OK:
        return "Extractive summarizer not installed. Install 'bert-extractive-summarizer'."
    model = ExtractiveSummarizer(model_name)
    if num_sentences:
        s = model(text, num_sentences=num_sentences)
    else:
        s = model(text, ratio=ratio)
    if isinstance(s, list):
        return " ".join(s)
    return s

# UI setup
st.set_page_config(page_title="Long Document Summarizer", layout="wide", initial_sidebar_state="expanded")
st.title("📚 Long Document Summarizer — Interactive UI")
st.write("A friendly UI for **LED hierarchical summarization** + quick extractive summarization. Upload files or paste text. Supports PDF, DOCX, TXT, CSV/Excel.")

# Sidebar options
with st.sidebar:
    st.header("🛠️ Settings")
    mode = st.radio("Mode", ("Abstractive (LED)", "Extractive (BERT)"))
    if mode == "Abstractive (LED)":
        model_name = st.selectbox("LED Model", ("allenai/led-base-16384",), index=0, help="Longformer Encoder-Decoder (16k tokens).")
        device = st.selectbox("Device", ("cpu", "cuda" if torch and torch.cuda.is_available() else "cpu"))
        chunk_tokens = st.slider("Chunk size (tokens)", min_value=1024, max_value=14000, value=6000, step=512)
        chunk_stride = st.slider("Chunk overlap (tokens)", min_value=0, max_value=1500, value=200, step=50)
        chunk_batch = st.slider("Chunk batch size", 1, 4, 1)
        final_max_len = st.slider("Final summary max tokens", 100, 1000, 300)
        beams = st.slider("Beams (final)", 1, 8, 4)
    else:
        model_name = st.selectbox("Extractive model", ("distilbert-base-uncased", "paraphrase-MiniLM-L6-v2"))
        ratio = st.slider("Ratio (extractive)", 0.05, 0.5, 0.15, step=0.01)
        num_sent = st.slider("Num sentences (0 = use ratio)", 0, 20, 0)

    st.markdown("---")
    show_bullets = st.checkbox("Return as bullet points", value=False)
    show_tldr = st.checkbox("Also show one-line TL;DR", value=True)
    st.markdown("---")
    st.write("Quick tips:")
    st.write("- For huge documents, use LED with chunking (default settings are safe).")
    st.write("- If you don't have GPU, keep chunk size smaller (4k–6k tokens).")
    st.write("- Extractive is fast; abstractive is more fluent and human-like.")
    st.write("---")
    if st.button("Reset history"):
        st.session_state.pop("history", None)
        st.success("History cleared")

# Main input area
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload file (txt, pdf, docx, csv, xlsx)", type=["txt", "pdf", "docx", "csv", "xlsx"])
    text_area = st.text_area("Or paste / edit your text here:", height=320)
    if uploaded:
        filename = uploaded.name
        st.info(f"Reading {filename} ...")
        if filename.lower().endswith(".pdf"):
            text_area = read_pdf(uploaded)
        elif filename.lower().endswith(".docx"):
            text_area = read_docx(uploaded)
        elif filename.lower().endswith(".txt"):
            text_area = read_txt(uploaded)
        elif filename.lower().endswith(".csv") or filename.lower().endswith(".xlsx"):
            text_area = read_csv_excel(uploaded, filename)
        else:
            text_area = read_txt(uploaded)
        st.success("File loaded. Edit the text below if you want.")
        st.text_area("Loaded text (editable):", text_area, height=300, key="loaded_text")

with col2:
    st.header("Preview / Tools")
    st.write("Paste a few paragraphs and test quickly.")
    if st.button("Load sample article"):
        sample = ("A single person claims to have authored 113 academic papers on artificial intelligence this year, "
                  "89 of which will be presented this week at one of the world's leading conferences. "
                  "This raised questions among researchers about publication quality.")
        st.session_state["sample"] = sample
        text_area = sample
        st.experimental_rerun()
    st.write("Session history:")
    history = st.session_state.get("history", [])
    if history:
        for idx, item in enumerate(reversed(history[-5:])):
            st.markdown(f"**#{len(history)-idx}** • {item['title'][:60]} — {item['mode']}")

# Run summarization
if st.button("Generate Summary"):
    content = text_area or st.session_state.get("sample", "")
    if not content.strip():
        st.warning("Please paste some text or upload a file first.")
    else:
        # progress area
        progress = st.empty()
        with st.spinner("Running summarization... this may take a while for large text"):
            if mode == "Abstractive (LED)":
                if not TRANSFORMERS_OK:
                    st.error("transformers / torch not installed. Install: pip install transformers torch")
                else:
                    gen_chunk_kwargs = {"num_beams": 2, "max_length": 200, "min_length": 40, "length_penalty": 2.0, "early_stopping": True}
                    final_kwargs = {"num_beams": beams, "max_length": final_max_len, "min_length": 80, "length_penalty": 2.0, "early_stopping": True}
                    result = hierarchical_led_summarize(content, model_name=model_name, device=device,
                                                       chunk_tokens=chunk_tokens, chunk_stride=chunk_stride,
                                                       batch_size=chunk_batch, chunk_gen_kwargs=gen_chunk_kwargs,
                                                       final_gen_kwargs=final_kwargs, progress=progress)
                    final = result["final_summary"]
                    pieces = result["chunk_summaries"]
            else:
                # extractive
                if not EXTRACTIVE_OK:
                    st.error("Extractive summarizer not installed. Install: pip install bert-extractive-summarizer")
                    final = ""
                    pieces = []
                else:
                    if num_sent > 0:
                        final = extractive_summary(content, model_name=model_name, num_sentences=num_sent)
                    else:
                        final = extractive_summary(content, model_name=model_name, ratio=ratio)
                    pieces = [final]

        # format output
        if show_bullets:
            bullets = [f"- {s.strip()}" for s in textwrap.wrap(final, width=200)]
            display_text = "\n".join(bullets)
        else:
            display_text = final

        st.header("📝 Final Summary")
        st.text_area("Summary", display_text, height=300)

        if show_tldr:
            tldr = display_text.split(".")[0].strip()
            st.markdown(f"**TL;DR:** {tldr}.")

        # Download and save to history
        st.download_button("Download summary", data=display_text, file_name="summary.txt", mime="text/plain")

        # Save history in session
        entry = {"title": (content[:120] + "...") if len(content) > 120 else content, "summary": display_text, "mode": mode}
        st.session_state.setdefault("history", []).append(entry)
        st.success("Saved to session history")

# show history viewer
if st.session_state.get("history"):
    st.markdown("---")
    st.subheader("History (session)")
    for i, item in enumerate(reversed(st.session_state["history"][-10:])):
        with st.expander(f"{len(st.session_state['history'])-i}. {item['title'][:80]}"):
            st.write(f"Mode: {item['mode']}")
            st.write(item['summary'])
            st.download_button(f"Download #{len(st.session_state['history'])-i}", data=item['summary'],
                                file_name=f"summary_{len(st.session_state['history'])-i}.txt")
