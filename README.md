<div align="center">

# 📄 DocuMind AI

### Chat With Your PDFs — Powered by AI

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-6C3483?style=for-the-badge)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-F39C12?style=for-the-badge)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://3e5oqcgyrvyodm9pp8239u.streamlit.app/)

**Upload any PDF. Ask any question. Get answers straight from the document.**

🚀 **[Try the Live Demo →](https://3e5oqcgyrvyodm9pp8239u.streamlit.app/)**

[Features](#-features) · [Demo](#-screenshots) · [Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Tech Stack](#-tech-stack)

</div>

---

## 🧠 What Is DocuMind AI?

DocuMind AI is a Retrieval-Augmented Generation (RAG) web application built with Python and Streamlit. It lets you upload any PDF — research papers, contracts, reports, manuals — and have a natural conversation with its content.

The AI **only answers from your document**. If the information isn't there, it tells you so. No hallucinations, no guessing.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📤 **PDF Upload** | Drag-and-drop or browse to upload any text-based PDF |
| 🔍 **Semantic Search** | Finds the most relevant passages using vector embeddings |
| 💬 **Chat Interface** | Natural ChatGPT-style conversation with your document |
| 📚 **Source Citations** | Every answer shows the exact passages it was based on |
| 🚫 **Hallucination Guard** | Answers only from the PDF — says "not found" if it can't |
| 🆓 **Free by Default** | No API key needed — runs fully locally |
| 🔑 **OpenAI Upgrade** | Add an API key for GPT-3.5-Turbo quality answers |
| 🎨 **Professional UI** | Clean dark-themed interface built for portfolios |

---

## 📸 Screenshots

> **To add your own screenshots:** Run the app, take screenshots, and save them in the `screenshots/` folder.

| Main Interface | Chat in Action |
|:---:|:---:|
| *(Upload screen)* | *(Chat + sources)* |

---

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────────────┐
│                  DocuMind AI                    │
├─────────────────────────────────────────────────┤
│  Frontend       →  Streamlit                    │
│  PDF Parsing    →  pypdf                        │
│  Text Chunking  →  Custom RecursiveTextSplitter │
│  Embeddings     →  all-MiniLM-L6-v2 (ONNX)     │
│  Vector Store   →  ChromaDB (in-memory)         │
│  Free LLM       →  Extractive (no download)     │
│  Optional LLM   →  OpenAI GPT-3.5-Turbo        │
└─────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
documind-ai/
│
├── app.py                  # Main Streamlit application & UI
├── requirements.txt        # Python dependencies (lightweight)
├── .env.example            # Environment variables template
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py       # PDF text extraction via pypdf
│   ├── text_splitter.py    # Recursive character text chunking
│   ├── vector_store.py     # ChromaDB collection create & query
│   └── chatbot.py          # Answer generation (OpenAI / extractive)
│
└── screenshots/            # Add app screenshots here
```

---

## ⚡ Quick Start

### Prerequisites

- Python **3.9+**
- pip

### 1 — Clone

```bash
git clone https://github.com/IrfanGoraya442/DocuMind.git
cd documind-ai
```

### 2 — Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> Dependencies are intentionally lightweight. No PyTorch or heavy model downloads required.

### 4 — Configure (Optional)

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI key if you want GPT-3.5 answers:

```env
OPENAI_API_KEY=sk-...
```

Leave it blank to use free extractive mode.

### 5 — Run

```bash
streamlit run app.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📖 How It Works

```
 PDF Upload
     │
     ▼
 Text Extraction ──── pypdf reads all pages
     │
     ▼
 Text Chunking ─────── Split into 600-char overlapping chunks
     │
     ▼
 Embedding ─────────── all-MiniLM-L6-v2 converts chunks to vectors
     │
     ▼
 Vector Store ──────── ChromaDB stores vectors in memory
     │
     ▼
 User Question
     │
     ▼
 Semantic Search ───── Cosine similarity finds top-4 relevant chunks
     │
     ▼
 Answer Generation ─── OpenAI GPT-3.5  OR  Extractive (free)
     │
     ▼
 Answer + Sources displayed in chat
```

### The "Not Found" Guard

If the cosine distance of all retrieved chunks exceeds the relevance threshold, the app responds:

> *"I could not find this information in the uploaded document."*

This prevents the AI from making up answers.

---

## 💡 Example Use Cases

| Document | Example Question |
|---|---|
| 📄 Research Paper | *"What dataset was used for evaluation?"* |
| 📋 Legal Contract | *"What are the termination conditions?"* |
| 📊 Financial Report | *"What was the net revenue in Q3?"* |
| 📖 User Manual | *"How do I perform a factory reset?"* |
| 📝 Resume | *"What is the candidate's work experience?"* |
| 📚 Textbook Chapter | *"Summarize the main argument of this chapter."* |

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(empty)* | If set, uses GPT-3.5-Turbo for answers |

---

## 🔮 Roadmap

- [ ] Multi-PDF support with document switcher
- [ ] Persistent vector store (re-index without re-uploading)
- [ ] OCR support for scanned / image-based PDFs
- [ ] Highlighted source passages within the PDF viewer
- [ ] Export full chat history as Markdown or PDF
- [ ] Streaming responses for faster UX
- [ ] One-click deploy to Streamlit Community Cloud

---

## 🤝 Contributing

Contributions are welcome.

```bash
# Fork → clone → create branch
git checkout -b feature/your-feature

# Make changes → commit → push
git push origin feature/your-feature

# Open a Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

<div align="center">

Built with ❤️ by **Muhammad Irfan** · [Streamlit](https://streamlit.io) · [ChromaDB](https://trychroma.com) · [pypdf](https://github.com/py-pdf/pypdf)

**⭐ Star this repo if you found it useful**

</div>
