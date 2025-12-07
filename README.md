# 📘 AI-Powered Long Document Summarizer  
### LED + BERT | Streamlit UI | Abstractive & Extractive Summaries

A modern AI application that summarizes **very long documents** using state-of-the-art Transformer models.  
It supports **LED (Longformer Encoder–Decoder)** for abstractive summarization and **BERT/SBERT** for extractive summarization — all inside a fast, elegant **Streamlit web interface**.

This project is built by **Nhowmitha Suresh**, 3rd-year AI & Data Science student passionate about NLP, deep learning models, and real-world AI applications.
<img width="1919" height="1070" alt="Screenshot 2025-12-07 111804" src="https://github.com/user-attachments/assets/957aaead-5012-483a-b34e-0c00d28d822c" />

<img width="1918" height="1079" alt="Screenshot 2025-12-07 111817" src="https://github.com/user-attachments/assets/51d4e7de-27c4-4f3b-89b4-45de5e4cc6c0" />


---

## 🚀 Why This Project?

Summarizing large documents such as:
- research papers  
- datasets  
- reports  
- articles  
- academic text  
- multi-page PDFs  

is difficult using normal models (T5, BART, etc.) because they cannot handle long text.

This app solves that using:
- **LED 16,384-token model** (long document support)  
- **Hierarchical Chunking** (splits → summarizes → merges)  
- **Streamlit UI** (easy and interactive)

---

# 🌟 Features

### ✨ 1. Streamlit Web Application  
- Modern UI with sidebar configuration  
- Paste text or upload files  
- View, copy, and download summary  
- Fast, smooth user experience  

### ✨ 2. Multiple Summarization Models  
| Model | Type | Best For |
|-------|------|----------|
| **LED (allenai/led-base-16384)** | Abstractive | Very large documents |
| **SBERT / BERT** | Extractive | Fast and accurate summaries |

### ✨ 3. Supports Multiple File Types  
- `.txt`  
- `.pdf`  
- `.docx`  
- `.csv`  
- `.xlsx`

### ✨ 4. Hierarchical Summarization  
Breaks huge text → summarizes chunks → summarizes the summaries.  
Ensures **quality + speed**.

### ✨ 5. Adjustable Controls  
- Summary length  
- Beam search  
- Chunk size  
- Overlap  
- Bullet summary  
- TL;DR mode  

### ✨ 6. Download Summary  
Download output as `.txt` or copy directly from the UI.

---

# 🛠️ Installation

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/Nhowmitha-suresh/AI-Powered-Long-Document-Summarizer-LED-BERT-.git
cd AI-Powered-Long-Document-Summarizer-LED-BERT-
2️⃣ Create a virtual environment
bash
Copy code
python -m venv .venv
3️⃣ Activate environment
Windows

bash
Copy code
.venv\Scripts\activate
Mac/Linux

bash
Copy code
source .venv/bin/activate
4️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
▶️ Run the Application
bash
Copy code
streamlit run app.py
Then open:
👉 http://localhost:8501

📁 Project Structure
bash
Copy code
📦 AI-Powered-Long-Document-Summarizer-LED-BERT-
│
├── app.py                    # Streamlit UI
├── led_summary.py            # Basic LED summarizer
├── led_hierarchical.py       # Hierarchical long-document summarizer
├── requirements.txt          # All dependencies
├── huge_text.txt             # Example input
└── README.md                 # Project documentation
🧠 Tech Stack
Python

HuggingFace Transformers

LED (Longformer Encoder Decoder)

BERT / SBERT

PyTorch

Streamlit

PyPDF2, python-docx, pandas

🧩 Use Cases
✔ Summarizing research papers
✔ Reducing long reports into short insights
✔ Academic content summarization
✔ Automating document analysis
✔ Summaries for PDFs and DOCX files
✔ Extract key points for students & professionals

✨ About the Developer
👩‍💻 Nhowmitha Suresh
3rd Year — AI & Data Science
Passionate about NLP, Deep Learning, and building real-world AI systems.

This project reflects:

Strong understanding of Transformer models

Ability to build real AI applications

Knowledge of NLP pipelines and UI frameworks

📜 License
MIT License — free to use and modify.

⭐ Support
If you find this project useful:

🌟 Star the repo
🍴 Fork it
🐞 Report issues
📬 Connect with me for collaborations
