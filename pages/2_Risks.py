"""Risk & recommendations analysis; optional index rebuild for Ask page."""

import importlib.util
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from config.config import Config
from src.document_loader import DocumentLoader
from src.llm_client import GroqLLMClient
from src.risk_analysis import detect_risks_and_recommendations
from src.text_splitter import OptimizedTextSplitter

logging.getLogger().setLevel(logging.WARNING)


@st.cache_resource
def get_llm():
    return GroqLLMClient()


def save_upload(name: str, data: bytes) -> Path:
    Config.create_directories()
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    path = Path(Config.RAW_DATA_DIR) / safe
    path.write_bytes(data)
    return path


def main():
    st.header("Risks & recommendations")
    st.caption(
        "Risk scan uses **RISK_CHUNK_SIZE** (larger slices, fewer API calls) — different from RAG **CHUNK_SIZE**. "
        "Rebuild index separately if you need **Ask** to match."
    )

    loader = DocumentLoader()
    splitter = OptimizedTextSplitter(
        chunk_size=Config.RISK_CHUNK_SIZE,
        chunk_overlap=Config.RISK_CHUNK_OVERLAP,
    )

    uploaded = st.file_uploader(
        "Upload a document (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
    )

    col_a, col_b = st.columns(2)
    with col_a:
        analyze_upload = st.button("Analyze uploaded file", disabled=uploaded is None)
    with col_b:
        analyze_raw = st.button("Analyze all files in data/raw/")

    results = None

    try:
        llm = get_llm()
    except Exception as e:
        st.error(f"LLM not configured: {e}")
        return

    if analyze_upload and uploaded:
        path = save_upload(uploaded.name, uploaded.getvalue())
        st.success(f"Saved to `{path}`")
        doc = loader.load_file(path)
        chunks = splitter.split_documents([doc])
        batches = (len(chunks) + Config.RISK_BATCH_SIZE - 1) // Config.RISK_BATCH_SIZE
        with st.spinner(
            f"{len(chunks)} text sections → ~{batches} API batch(es) "
            f"(size {Config.RISK_CHUNK_SIZE} chars, batch {Config.RISK_BATCH_SIZE})..."
        ):
            results = detect_risks_and_recommendations(chunks, llm)

    if analyze_raw:
        docs = loader.load_documents(Config.RAW_DATA_DIR)
        if not docs:
            st.warning("No documents in data/raw/. Upload a file or add files manually.")
        else:
            chunks = splitter.split_documents(docs)
            batches = (len(chunks) + Config.RISK_BATCH_SIZE - 1) // Config.RISK_BATCH_SIZE
            with st.spinner(
                f"{len(chunks)} text sections → ~{batches} API batch(es) "
                f"(size {Config.RISK_CHUNK_SIZE} chars, batch {Config.RISK_BATCH_SIZE})..."
            ):
                results = detect_risks_and_recommendations(chunks, llm)

    if results:
        df = pd.DataFrame(results)
        st.subheader("Results")
        st.dataframe(df, height=400, use_container_width=True)
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name="risk_analysis.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("Optional: Google Sheet")
        if st.button("Save results to Google Sheet"):
            try:
                from src.integrations import save_results_to_google_sheet

                save_results_to_google_sheet(results)
                st.success("Saved to Google Sheet.")
            except Exception as e:
                st.error(str(e))

        st.subheader("Optional: rebuild Q&A index")
        if st.button("Rebuild FAISS index (for Ask page)"):
            try:
                mod_path = ROOT / "scripts" / "build_index.py"
                spec = importlib.util.spec_from_file_location("build_index_m", mod_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                stats = mod.run_build_index()
                st.success(f"Index rebuilt: {stats.get('vector_store', {})}")
                st.info("Reload the **Ask** page to refresh the cached pipeline.")
            except Exception as e:
                st.error(str(e))

    st.sidebar.markdown("### Risk scan (env)")
    st.sidebar.caption(
        f"RISK_CHUNK_SIZE={Config.RISK_CHUNK_SIZE}, "
        f"MAX_RISK_CHUNKS={Config.MAX_RISK_CHUNKS}, "
        f"BATCH={Config.RISK_BATCH_SIZE}, "
        f"delay={Config.RISK_CHUNK_DELAY_SEC}s/batch"
    )


main()
