"""RAG Q&A: indexed library, session file uploads, optional web context."""

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config.config import Config
from src.rag_pipeline import RAGPipeline
from src.upload_docs import documents_from_uploaded_files

logging.getLogger().setLevel(logging.WARNING)

KNOWLEDGE_LIBRARY = "Indexed library"
KNOWLEDGE_UPLOAD = "Upload files (this session only)"

# DuckDuckGo `region` codes — default us-en avoids Chinese snippets when IP/locale is Asia-Pacific.
WEB_SEARCH_REGION_LABELS = {
    "us-en": "English (United States)",
    "uk-en": "English (United Kingdom)",
    "wt-wt": "Worldwide (mixed languages)",
    "cn-zh": "中文 (China)",
}


def _vector_index_cache_key() -> str:
    """Invalidate cache when index is built/rebuilt (avoids stale FAISS errors)."""
    ip, mp = Config.FAISS_INDEX_PATH, Config.METADATA_PATH
    if not os.path.isfile(ip) or not os.path.isfile(mp):
        return "missing"
    try:
        return f"{os.path.getmtime(ip):.6f}:{os.path.getmtime(mp):.6f}"
    except OSError:
        return "missing"


@st.cache_resource
def initialize_pipeline(index_cache_key: str):
    try:
        pipeline = RAGPipeline()
        pipeline.initialize()
        return pipeline, None
    except Exception as e:
        return None, str(e)


def display_sources(sources):
    if sources:
        with st.expander(f"Document sources ({len(sources)})", expanded=False):
            for source in sources:
                st.markdown(
                    f"**{source['file_name']}** — similarity `{source['similarity_score']}`"
                )
                st.caption(source["chunk_preview"])
                st.divider()


def display_web_sources(web_sources):
    if not web_sources:
        return
    with st.expander(f"Web sources ({len(web_sources)})", expanded=False):
        for w in web_sources:
            title = w.get("title") or "Result"
            url = w.get("url") or ""
            snip = w.get("snippet") or ""
            if url:
                st.markdown(f"**[{title}]({url})**")
            else:
                st.markdown(f"**{title}**")
            st.caption(snip)
            st.divider()


def _metrics_row(stats: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks", stats.get("chunks_retrieved", 0))
    c2.metric("Time (s)", f"{stats.get('processing_time', 0):.2f}")
    c3.metric("Avg similarity", f"{stats.get('avg_similarity', 0):.3f}")
    c4.metric("Confidence", stats.get("confidence_band", "—"))
    if "web_hits" in stats:
        st.caption(f"Web search rows retrieved: **{stats['web_hits']}** (0 means DDG/Bing returned nothing).")


def _append_assistant_message(content, sources, web_sources, stats, include_sources):
    entry = {
        "role": "assistant",
        "content": content,
        "stats": stats,
    }
    if include_sources:
        entry["sources"] = sources or []
        entry["web_sources"] = web_sources or []
    else:
        entry["sources"] = []
        entry["web_sources"] = []
    st.session_state.chat_messages.append(entry)


def main():
    st.header("Ask")
    st.caption(
        "Answers are grounded in **retrieved chunks** from your chosen knowledge source. "
        "Optional **web search** adds public snippets (verify sources — not legal advice)."
    )

    st.session_state.setdefault("upload_pipeline", None)

    knowledge_source = st.radio(
        "Knowledge source",
        [KNOWLEDGE_LIBRARY, KNOWLEDGE_UPLOAD],
        horizontal=True,
    )

    pipeline = None
    disk_error = None
    if knowledge_source == KNOWLEDGE_LIBRARY:
        pipeline, disk_error = initialize_pipeline(_vector_index_cache_key())

    with st.sidebar:
        st.subheader("Augmentation")
        use_web = st.checkbox(
            "Include web search (DuckDuckGo)",
            value=False,
            help="Adds public web snippets to the prompt. No API key. Verify links yourself.",
        )
        web_region = Config.WEB_SEARCH_REGION
        if use_web:
            codes = list(WEB_SEARCH_REGION_LABELS.keys())
            default_idx = 0
            if web_region in codes:
                default_idx = codes.index(web_region)
            web_region = st.selectbox(
                "Web search region",
                options=codes,
                index=default_idx,
                format_func=lambda c: WEB_SEARCH_REGION_LABELS.get(c, c),
                help=(
                    "If snippets appear in Chinese or another language, choose **English (United States)** "
                    "or **United Kingdom**. DuckDuckGo follows this instead of your network locale."
                ),
            )
        st.caption(f"Up to {Config.WEB_SEARCH_MAX_RESULTS} results (WEB_SEARCH_MAX_RESULTS).")

        st.divider()
        st.subheader("Retrieval")
        top_k = st.slider("Chunks (top-k)", 1, 10, Config.TOP_K_RETRIEVAL)
        similarity_threshold = st.slider(
            "Similarity threshold",
            0.0,
            1.0,
            float(Config.SIMILARITY_THRESHOLD),
            0.05,
        )
        include_sources = st.checkbox("Show document + web sources", value=True)
        enable_streaming = st.checkbox("Streaming", value=Config.ENABLE_STREAMING)

        st.divider()
        if knowledge_source == KNOWLEDGE_LIBRARY:
            st.caption("Index cache")
            if st.button("Reload pipeline (after build_index)"):
                initialize_pipeline.clear()
                st.rerun()

    active_pipeline = None
    if knowledge_source == KNOWLEDGE_LIBRARY:
        if disk_error:
            st.error("Indexed library failed to load")
            st.code(str(disk_error), language="text")
            st.markdown(
                "**Typical fix:** add documents under `data/raw/`, then run `python scripts/build_index.py`, "
                "then click **Reload pipeline** in the sidebar. Ensure `GROQ_API_KEY` is set in `.env`."
            )
            st.info(
                "You can switch to **Upload files** above to chat against attachments without building the disk index."
            )
            return
        active_pipeline = pipeline
    else:
        st.markdown("Upload PDF, TXT, or DOCX. Files stay in this browser session; nothing is written to `data/raw/`.")
        uploaded = st.file_uploader(
            "Files",
            type=["pdf", "txt", "docx", "doc"],
            accept_multiple_files=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            load_clicked = st.button("Index uploads for this chat", type="primary")
        with c2:
            if st.button("Clear uploaded index"):
                st.session_state.upload_pipeline = None
                st.rerun()

        if load_clicked:
            if not uploaded:
                st.warning("Choose one or more files first.")
            else:
                try:
                    docs = documents_from_uploaded_files(uploaded)
                    if not docs:
                        st.warning("No supported files loaded. Use PDF, TXT, or DOCX.")
                    else:
                        p = RAGPipeline()
                        p.init_from_documents(docs)
                        st.session_state.upload_pipeline = p
                        st.success(f"Indexed {len(docs)} file(s). You can chat below.")
                except Exception as e:
                    st.error(str(e))

        active_pipeline = st.session_state.upload_pipeline
        if active_pipeline is None:
            st.info("Index at least one file to enable grounded answers in this mode.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if include_sources and message.get("sources"):
                    display_sources(message["sources"])
                if include_sources and message.get("web_sources"):
                    display_web_sources(message["web_sources"])
                if message.get("stats"):
                    _metrics_row(message["stats"])

    prompt = st.chat_input("Ask a question...")
    if prompt:
        if active_pipeline is None:
            st.warning("No knowledge base ready. Build the library index or index uploads first.")
            st.stop()

        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if enable_streaming and Config.ENABLE_STREAMING:
                placeholder = st.empty()
                full = ""
                try:
                    result = active_pipeline.answer_query(
                        prompt,
                        stream=True,
                        include_sources=include_sources,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        use_web=use_web,
                        web_region=web_region,
                    )
                    for chunk in result["answer_stream"]:
                        full += chunk
                        placeholder.markdown(full + "▌")
                    placeholder.markdown(full)
                    if include_sources and result.get("sources"):
                        display_sources(result["sources"])
                    if include_sources and result.get("web_sources"):
                        display_web_sources(result["web_sources"])
                    stats = result.get("query_stats", {})
                    _metrics_row(stats)
                    _append_assistant_message(
                        full,
                        result.get("sources", []),
                        result.get("web_sources", []),
                        stats,
                        include_sources,
                    )
                except Exception as e:
                    st.error(str(e))
            else:
                try:
                    with st.spinner("Thinking..."):
                        result = active_pipeline.answer_query(
                            prompt,
                            stream=False,
                            include_sources=include_sources,
                            top_k=top_k,
                            similarity_threshold=similarity_threshold,
                            use_web=use_web,
                            web_region=web_region,
                        )
                    st.markdown(result["answer"])
                    if include_sources and result.get("sources"):
                        display_sources(result["sources"])
                    if include_sources and result.get("web_sources"):
                        display_web_sources(result["web_sources"])
                    stats = result.get("query_stats", {})
                    _metrics_row(stats)
                    _append_assistant_message(
                        result["answer"],
                        result.get("sources", []),
                        result.get("web_sources", []),
                        stats,
                        include_sources,
                    )
                except Exception as e:
                    st.error(str(e))

    if st.sidebar.button("Clear chat"):
        st.session_state.chat_messages = []
        st.rerun()


main()
