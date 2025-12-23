# AI-Powered Long Document Summarizer (LED + BERT)

An interactive **Streamlit-based web application** for summarizing long documents using **Abstractive (LED)** and **Extractive (BERT)** summarization techniques.  
Designed to handle **large documents efficiently** with a clean, professional UI.

<img width="1919" height="1077" alt="Screenshot 2025-12-22 223341" src="https://github.com/user-attachments/assets/00c63726-0ef2-4659-b6ec-76bf5a0ea8f4" />

<img width="1919" height="1079" alt="Screenshot 2025-12-22 223349" src="https://github.com/user-attachments/assets/e712bd91-5c06-46a5-835f-774ff59db686" />



---

## Overview

This project provides an AI-powered solution for summarizing long textual documents such as research papers, articles, reports, and legal documents.  
It supports both **abstractive summarization** for human-like summaries and **extractive summarization** for fast sentence selection.

---

## Key Features

- Abstractive summarization using **Longformer Encoder-Decoder (LED)**
- Extractive summarization using **BERT**
- Handles long documents (10k+ tokens) using hierarchical chunking
- Supports multiple file formats:
  - TXT
  - PDF
  - DOCX
  - CSV / Excel
- Interactive Streamlit UI
- Adjustable chunk size, overlap, and summary length
- Word count and compression statistics
- Summary download support
- Lazy loading of models for better performance

---

## Application Architecture

1. User uploads a document or pastes text
2. Text is preprocessed and chunked using token-based splitting
3. Summarization is applied:
   - Abstractive mode → LED
   - Extractive mode → BERT
4. Final summary is generated and displayed in the UI
5. User can download or review previous summaries

---

## Models Used

### LED (Longformer Encoder-Decoder)
- Model: `allenai/led-base-16384`
- Used for abstractive summarization
- Capable of processing very long documents

### BERT Extractive Summarizer
- Model: `distilbert-base-uncased`
- Used for extractive summarization
- Fast and lightweight

---

### documentation
git push origin main


