import logging
from typing import Any, Dict, List

import numpy as np

from config.config import Config
from src.embeddings import EmbeddingGenerator
from src.vector_store import FAISSVectorStore


class QueryProcessor:
    """Query processing and context retrieval."""

    def __init__(self, vector_store: FAISSVectorStore, embedding_generator: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.logger = logging.getLogger(__name__)

    def process_query(
        self,
        query: str,
        top_k: int = None,
        similarity_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        top_k = top_k if top_k is not None else Config.TOP_K_RETRIEVAL
        similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else Config.SIMILARITY_THRESHOLD
        )

        self.logger.info("Processing query: %s...", query[:100])

        query_embedding = self.embedding_generator.encode_single(query)
        results = self.vector_store.similarity_search(query_embedding, k=top_k)

        filtered_results = [
            result for result in results if result["similarity_score"] >= similarity_threshold
        ]

        self.logger.info(
            "Found %s relevant chunks above threshold %s",
            len(filtered_results),
            similarity_threshold,
        )

        return filtered_results

    def prepare_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_length: int = None,
    ) -> str:
        max_context_length = max_context_length or Config.MAX_CONTEXT_LENGTH

        if not retrieved_chunks:
            return "No relevant context found."

        context_parts = []
        total_length = 0

        for i, chunk in enumerate(retrieved_chunks):
            content = chunk["content"].strip()
            source = chunk["metadata"].get("file_name", "Unknown")

            chunk_text = f"[Source {i + 1}: {source}]\n{content}\n"

            if total_length + len(chunk_text) > max_context_length:
                remaining_space = max_context_length - total_length - 50
                if remaining_space > 100:
                    truncated = content[:remaining_space] + "..."
                    chunk_text = f"[Source {i + 1}: {source}]\n{truncated}\n"
                    context_parts.append(chunk_text)
                break

            context_parts.append(chunk_text)
            total_length += len(chunk_text)

        context = "\n".join(context_parts)

        self.logger.info(
            "Prepared context with %s chunks (%s characters)",
            len(context_parts),
            len(context),
        )

        return context

    def get_query_stats(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "query_length": len(query),
            "chunks_retrieved": len(retrieved_chunks),
            "avg_similarity": (
                np.mean([chunk["similarity_score"] for chunk in retrieved_chunks])
                if retrieved_chunks
                else 0
            ),
            "sources": list(
                {chunk["metadata"].get("file_name", "Unknown") for chunk in retrieved_chunks}
            ),
        }
