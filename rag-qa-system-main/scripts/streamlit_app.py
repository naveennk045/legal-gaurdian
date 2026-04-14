import streamlit as st
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_pipeline import RAGPipeline
from config.config import Config

# Configure logging to reduce noise in Streamlit
logging.getLogger().setLevel(logging.WARNING)


@st.cache_resource
def initialize_pipeline():
    """Initialize and cache the RAG pipeline"""
    try:
        pipeline = RAGPipeline()
        pipeline.initialize()
        return pipeline, None
    except Exception as e:
        return None, str(e)


def display_sources(sources):
    """Display sources in an expandable format"""
    if sources:
        with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
            for source in sources:
                st.write(f"**{source['file_name']}** (Similarity: {source['similarity_score']})")
                st.write(f"*Preview:* {source['chunk_preview']}")
                st.divider()


def display_stats(stats):
    """Display query statistics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Chunks Retrieved", stats.get('chunks_retrieved', 0))
    with col2:
        st.metric("Processing Time", f"{stats.get('processing_time', 0):.2f}s")
    with col3:
        st.metric("Avg Similarity", f"{stats.get('avg_similarity', 0):.3f}")
    with col4:
        st.metric("Sources", len(stats.get('sources', [])))


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="RAG Q&A System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 RAG Q&A System")
    st.markdown("Ask questions about your documents and get AI-powered answers with source citations!")

    # Initialize pipeline
    pipeline, error = initialize_pipeline()

    if error:
        st.error(f"Failed to initialize pipeline: {error}")
        st.markdown("### 🔧 Setup Instructions:")
        st.markdown("1. Make sure you've built the vector index: `python scripts/build_index.py`")
        st.markdown("2. Set your GROQ_API_KEY in the `.env` file")
        st.markdown("3. Install all requirements: `pip install -r requirements.txt`")
        return

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        top_k = st.slider("Number of chunks to retrieve", 1, 10, Config.TOP_K_RETRIEVAL)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, Config.SIMILARITY_THRESHOLD, 0.05)
        include_sources = st.checkbox("Show sources", value=True)
        enable_streaming = st.checkbox("Enable streaming", value=Config.ENABLE_STREAMING)

        st.divider()

        # Pipeline stats
        if st.button("Show Pipeline Stats"):
            stats = pipeline.get_pipeline_stats()
            st.json(stats)

    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message["sources"])
            if message["role"] == "assistant" and "stats" in message:
                display_stats(message["stats"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            if enable_streaming:
                # Streaming response
                message_placeholder = st.empty()
                full_response = ""

                try:
                    result = pipeline.answer_query(
                        query=prompt,
                        stream=True,
                        include_sources=include_sources
                    )

                    # Stream the response
                    for chunk in result["answer_stream"]:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                    # Display sources and stats
                    if include_sources and result.get("sources"):
                        display_sources(result["sources"])

                    display_stats(result.get("query_stats", {}))

                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": result.get("sources", []) if include_sources else [],
                        "stats": result.get("query_stats", {})
                    })

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

            else:
                # Non-streaming response
                try:
                    with st.spinner("Thinking..."):
                        result = pipeline.answer_query(
                            query=prompt,
                            stream=False,
                            include_sources=include_sources
                        )

                    # Display response
                    st.markdown(result["answer"])

                    # Display sources and stats
                    if include_sources and result.get("sources"):
                        display_sources(result["sources"])

                    display_stats(result.get("query_stats", {}))

                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []) if include_sources else [],
                        "stats": result.get("query_stats", {})
                    })

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
