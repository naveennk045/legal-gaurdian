import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from config.config import Config
from src.embeddings import EmbeddingGenerator
from src.llm_client import GroqLLMClient
from src.query_processor import QueryProcessor
from src.text_splitter import OptimizedTextSplitter
from src.vector_store import FAISSVectorStore


def confidence_band(avg_similarity: float) -> str:
    """MVP confidence label from retrieval similarity (no knowledge graph)."""
    if avg_similarity >= 0.7:
        return "High"
    if avg_similarity >= 0.5:
        return "Moderate"
    if avg_similarity >= 0.3:
        return "Low"
    return "Very low"


class RAGPipeline:
    """RAG orchestrator: FAISS retrieval + Groq generation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embedding_generator = None
        self.vector_store = None
        self.query_processor = None
        self.llm_client = None

    def initialize(self):
        self.logger.info("Initializing RAG pipeline...")

        self.embedding_generator = EmbeddingGenerator(
            model_name=Config.EMBEDDING_MODEL,
            cache_dir=Config.MODELS_DIR,
        )
        self.embedding_generator.initialize_model()

        dimension = self.embedding_generator.get_embedding_dimension()
        self.vector_store = FAISSVectorStore(
            dimension=dimension,
            index_path=Config.FAISS_INDEX_PATH,
            metadata_path=Config.METADATA_PATH,
        )

        Config.create_directories()
        index_path = Config.FAISS_INDEX_PATH
        meta_path = Config.METADATA_PATH
        if not os.path.isfile(index_path) or not os.path.isfile(meta_path):
            raise FileNotFoundError(
                "No FAISS index yet. Add PDF/TXT/DOCX files under data/raw/, then run:\n"
                "  python scripts/build_index.py"
            )

        try:
            self.vector_store.load_index()
            self.logger.info("Loaded existing vector index")
        except Exception as e:
            self.logger.error("Failed to load vector index: %s", e)
            raise

        self.query_processor = QueryProcessor(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
        )

        self.llm_client = GroqLLMClient()

        if not self.llm_client.check_connection():
            raise RuntimeError("Failed to connect to Groq API")

        self.logger.info("RAG pipeline initialized successfully")

    def init_from_documents(self, documents: List[Document]) -> None:
        """Build an in-memory FAISS index from uploaded/extracted documents (session scope)."""
        if not documents:
            raise ValueError("No documents provided.")

        self.logger.info("Building ephemeral RAG index from %s document(s)", len(documents))

        self.embedding_generator = EmbeddingGenerator(
            model_name=Config.EMBEDDING_MODEL,
            cache_dir=Config.MODELS_DIR,
        )
        self.embedding_generator.initialize_model()

        splitter = OptimizedTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError(
                "No text could be extracted from the file(s). Try PDF/TXT/DOCX with selectable text."
            )

        texts = [c.page_content for c in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        dimension = self.embedding_generator.get_embedding_dimension()

        self.vector_store = FAISSVectorStore(
            dimension=dimension,
            index_path=None,
            metadata_path=None,
        )
        self.vector_store.add_embeddings(embeddings, chunks)

        self.query_processor = QueryProcessor(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
        )

        self.llm_client = GroqLLMClient()
        if not self.llm_client.check_connection():
            raise RuntimeError("Failed to connect to Groq API")

        self.logger.info("Ephemeral index ready (%s chunks)", len(chunks))

    def answer_query(
        self,
        query: str,
        stream: bool = False,
        include_sources: bool = True,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_web: bool = False,
        web_region: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        try:
            web_block = ""
            web_sources: List[Dict[str, Any]] = []
            if use_web:
                from src.web_search import fetch_web_context

                web_block, web_sources = fetch_web_context(
                    query,
                    max_results=Config.WEB_SEARCH_MAX_RESULTS,
                    region=web_region,
                )

            retrieved_chunks = self.query_processor.process_query(
                query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

            doc_part = ""
            if retrieved_chunks:
                doc_part = self.query_processor.prepare_context(retrieved_chunks)

            # Do not require `web_sources`: if the block has text but the list is empty (parse quirks,
            # empty snippets), we still inject web into the prompt so search is not silently ignored.
            has_web = bool(use_web and web_block and str(web_block).strip())
            has_docs = bool(retrieved_chunks)

            if not has_docs and not has_web:
                return {
                    "answer": "I couldn't find relevant information in your documents for this question. "
                    "Enable **web search** in the sidebar for public-web context, or rephrase.",
                    "sources": [],
                    "web_sources": [],
                    "query_stats": {
                        "processing_time": time.time() - start_time,
                        "confidence_band": "N/A",
                        "chunks_retrieved": 0,
                    },
                }

            if has_web and has_docs:
                context = (
                    f"[Public web — verify; not your files]\n{web_block}\n\n"
                    f"[Your documents]\n{doc_part}"
                )
            elif has_web:
                context = f"[Public web — verify sources]\n{web_block}"
            else:
                context = doc_part

            doc_only = has_docs and not has_web
            prompt = self._create_rag_prompt(query, context, use_web=has_web, doc_only=doc_only)
            system_prompt = self._get_system_prompt(use_web=has_web, doc_only=doc_only)

            if retrieved_chunks:
                stats = self.query_processor.get_query_stats(query, retrieved_chunks)
                stats["confidence_band"] = confidence_band(float(stats.get("avg_similarity", 0)))
            else:
                stats = {
                    "query_length": len(query),
                    "chunks_retrieved": 0,
                    "avg_similarity": 0.0,
                    "sources": [],
                    "confidence_band": "Web-only",
                }
            if use_web:
                stats["web_hits"] = len(web_sources)

            if stream and Config.ENABLE_STREAMING:

                def stream_with_metadata():
                    for chunk in self.llm_client.generate_response(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        stream=True,
                    ):
                        yield chunk

                return {
                    "answer_stream": stream_with_metadata(),
                    "sources": self._format_sources(retrieved_chunks) if include_sources else [],
                    "web_sources": web_sources if use_web else [],
                    "query_stats": {
                        **stats,
                        "processing_time": time.time() - start_time,
                    },
                }

            answer = self.llm_client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                stream=False,
            )

            processing_time = time.time() - start_time

            return {
                "answer": answer,
                "sources": self._format_sources(retrieved_chunks) if include_sources else [],
                "web_sources": web_sources if use_web else [],
                "query_stats": {
                    **stats,
                    "processing_time": processing_time,
                },
            }

        except Exception as e:
            self.logger.error("Error in RAG pipeline: %s", e)
            return {
                "answer": f"I encountered an error while processing your question: {e}",
                "sources": [],
                "web_sources": [],
                "query_stats": {
                    "processing_time": time.time() - start_time,
                    "error": str(e),
                },
            }

    def _create_rag_prompt(
        self,
        query: str,
        context: str,
        use_web: bool = False,
        doc_only: bool = True,
    ) -> str:
        extra = ""
        if use_web and not doc_only:
            extra = (
                " Web results are third-party snippets — they are not legal advice; verify URLs.\n"
            )
        return f"""Based on the following context, answer the question.{extra}

Context:
{context}

Question: {query}

Answer clearly. If the context includes both [Public web] and [Your documents], use documents for agreement-specific wording; add short, labeled points from web where they add definitions or background. Do not dismiss web snippets as adding nothing when they are present in the context.

Answer:"""

    def _get_system_prompt(self, use_web: bool = False, doc_only: bool = True) -> str:
        base = """You are a professional legal-document assistant.

Guidelines:
- When both user documents and web snippets are in the context: prioritize documents for contract-specific clauses; still incorporate relevant web points (e.g. general definitions, public rules) in 1–2 sentences when helpful, and label them as from the web.
- Cite file names from the document context when relevant.
- Do not invent contract clauses or citations not supported by the context.
- If information is missing, say so clearly."""
        if use_web and not doc_only:
            base += (
                "\n- Web snippets are unverified public search results — label them as web and remind the user to verify URLs."
            )
        return base

    def _format_sources(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        seen_sources = set()

        for i, chunk in enumerate(retrieved_chunks):
            source_name = chunk["metadata"].get("file_name", "Unknown")
            source_key = f"{source_name}_{chunk['metadata'].get('chunk_id', i)}"

            if source_key not in seen_sources:
                preview = chunk["content"]
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                sources.append(
                    {
                        "source_id": i + 1,
                        "file_name": source_name,
                        "similarity_score": round(chunk["similarity_score"], 3),
                        "chunk_preview": preview,
                    }
                )
                seen_sources.add(source_key)

        return sources

    def get_pipeline_stats(self) -> Dict[str, Any]:
        return {
            "vector_store_stats": self.vector_store.get_stats() if self.vector_store else {},
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.GROQ_MODEL,
            "top_k_retrieval": Config.TOP_K_RETRIEVAL,
            "similarity_threshold": Config.SIMILARITY_THRESHOLD,
        }
