import os

import streamlit as st
from dotenv import load_dotenv

from src.chatbot import get_answer, correct_query
from src.pdf_loader import load_pdf
from src.text_splitter import split_text
from src.vector_store import create_collection, query_collection

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Cleaner chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4a4a6a;
        border-radius: 12px;
        padding: 1rem;
    }

    /* Source expander */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #888;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }

    /* Hide Streamlit footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "collection": None,
    "pdf_name": None,
    "chunk_count": 0,
    "chat_history": [],
    "questions_answered": 0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 DocuMind AI")
    st.caption("Powered by Gemini AI + TF-IDF Search")
    st.divider()

    # AI mode indicator
    st.markdown("**AI Engine**")
    if os.environ.get("GEMINI_API_KEY"):
        st.success("Google Gemini 1.5 Flash ✨")
    elif os.environ.get("OPENAI_API_KEY"):
        st.success("OpenAI GPT-3.5 Turbo")
    else:
        st.info("Extractive mode (no API key)")
        st.caption("Add `GEMINI_API_KEY` to Streamlit secrets for AI answers.")

    st.divider()

    # Stats
    if st.session_state.pdf_name:
        st.markdown("**Current Document**")
        st.markdown(f"📄 `{st.session_state.pdf_name}`")
        col1, col2 = st.columns(2)
        col1.metric("Chunks", st.session_state.chunk_count)
        col2.metric("Q&A", st.session_state.questions_answered)

        if st.button("🗑️ Clear Document", use_container_width=True):
            for key in ["collection", "pdf_name", "chunk_count", "chat_history", "questions_answered"]:
                st.session_state[key] = defaults[key]
            st.rerun()

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        """
        1. Upload a PDF
        2. AI indexes its content
        3. Ask any question
        4. Get answers from the doc
        """
    )
    st.divider()
    st.markdown(
        "<small>Built with Streamlit · Gemini AI · scikit-learn</small>",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📄 DocuMind AI")
st.markdown(
    "Upload a PDF and have a **natural conversation** with its content — "
    "answers are grounded strictly in your document."
)
st.divider()

# ── Upload section ────────────────────────────────────────────────────────────
upload_col, info_col = st.columns([3, 2], gap="large")

with upload_col:
    st.markdown("### Upload Your PDF")
    uploaded_file = st.file_uploader(
        label="Drop a PDF here or click to browse",
        type=["pdf"],
        label_visibility="collapsed",
    )

with info_col:
    st.markdown("### Supported Content")
    st.markdown(
        """
        - 📚 Research papers & articles
        - 📋 Reports & whitepapers
        - 📖 Books & manuals
        - 📝 Contracts & legal documents
        - 📊 Financial statements
        """
    )
    st.caption("Note: Scanned / image-only PDFs are not supported.")

# ── PDF Processing ────────────────────────────────────────────────────────────
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.pdf_name:
        # New file uploaded — process it
        with st.status("Processing your PDF...", expanded=True) as status:
            try:
                st.write("📖 Extracting text...")
                text = load_pdf(uploaded_file)

                st.write("✂️ Splitting into chunks...")
                chunks = split_text(text)

                st.write("🧠 Generating embeddings (first run may download ~80MB model)...")
                collection = create_collection(chunks)

                # Persist in session
                st.session_state.collection = collection
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chunk_count = len(chunks)
                st.session_state.chat_history = []
                st.session_state.questions_answered = 0

                status.update(label="✅ PDF ready! Start asking questions.", state="complete")
            except (ValueError, RuntimeError) as e:
                status.update(label="❌ Failed", state="error")
                st.error(str(e))
                st.stop()
            except Exception as e:
                status.update(label="❌ Unexpected error", state="error")
                st.error(f"Unexpected error: {e}")
                st.stop()

# ── Chat interface ────────────────────────────────────────────────────────────
if st.session_state.collection is not None:
    st.divider()
    st.markdown("### 💬 Chat with Your Document")

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 View source passages", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Passage {i}:**")
                        st.markdown(
                            f"<div style='background:#1e1e2e;padding:0.75rem;border-radius:8px;"
                            f"border-left:3px solid #6c63ff;font-size:0.9rem'>{src}</div>",
                            unsafe_allow_html=True,
                        )
                        if i < len(msg["sources"]):
                            st.markdown("---")

    # Chat input
    user_question = st.chat_input(
        f"Ask anything about {st.session_state.pdf_name}..."
    )

    if user_question:
        # Auto-correct the query
        corrected_question, was_corrected = correct_query(user_question)

        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_question, "sources": []})
        with st.chat_message("user"):
            st.write(user_question)
            if was_corrected:
                st.caption(f"🔤 Auto-corrected to: *{corrected_question}*")

        # Generate answer using corrected query
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    chunks, distances = query_collection(
                        st.session_state.collection, corrected_question
                    )

                    result = get_answer(corrected_question, chunks, distances)

                    answer = result["answer"]
                    sources = result["sources"]

                except Exception as e:
                    answer = f"An error occurred while generating the answer: {e}"
                    sources = []

            st.write(answer)

            if sources:
                with st.expander("📚 View source passages", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Passage {i}:**")
                        st.markdown(
                            f"<div style='background:#1e1e2e;padding:0.75rem;border-radius:8px;"
                            f"border-left:3px solid #6c63ff;font-size:0.9rem'>{src}</div>",
                            unsafe_allow_html=True,
                        )
                        if i < len(sources):
                            st.markdown("---")

        # Save to history
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.session_state.questions_answered += 1
        st.rerun()

else:
    # Empty state — no PDF uploaded yet
    st.divider()
    st.markdown(
        """
        <div style='text-align:center;padding:3rem;color:#555'>
            <h3>👆 Upload a PDF above to get started</h3>
            <p>DocuMind AI will read your document and answer your questions<br>
            using only the content within it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
