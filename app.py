"""
Legal Guardian — unified Streamlit entry (MVP).

Run from project root:
  streamlit run app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Legal Guardian",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Legal Guardian (MVP)")
st.markdown(
    """
Welcome. This app merges **document Q&A (RAG)** and **risk & recommendations** analysis
into one project with a **single index** under `data/raw/` + `vector_db/`.

Use the sidebar pages:

1. **Ask** — Question answering over indexed documents (FAISS + Groq), with sources and a simple confidence band.
2. **Risks** — Upload or use files in `data/raw/`, run structured risk analysis, export CSV; optionally rebuild the Q&A index.

**Setup:** copy `.env.example` to `.env`, set `GROQ_API_KEY`, add documents to `data/raw/`, then run `python scripts/build_index.py` before **Ask**.
"""
)
