import logging
import numpy as np
from typing import List, Dict, Any, Optional
from src.embeddings import EmbeddingGenerator
from src.vector_store import FAISSVectorStore
from config.config import Config


class QueryProcessor:
    """Handles query processing and context retrieval"""

    def __init__(self, vector_store: FAISSVectorStore, embedding_generator: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.logger = logging.getLogger(__name__)

    def process_query(
            self,
            query: str,
            top_k: int = None,
            similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Process a query and return relevant context"""
        top_k = top_k or Config.TOP_K_RETRIEVAL
        similarity_threshold = similarity_threshold or Config.SIMILARITY_THRESHOLD

        self.logger.info(f"Processing query: {query[:100]}...")

        # Generate query embedding
        query_embedding = self.embedding_generator.encode_single(query)

        # Perform similarity search
        results = self.vector_store.similarity_search(query_embedding, k=top_k)

        # Filter by similarity threshold
        filtered_results = [
            result for result in results
            if result['similarity_score'] >= similarity_threshold
        ]

        self.logger.info(f"Found {len(filtered_results)} relevant chunks above threshold {similarity_threshold}")

        return filtered_results

    def prepare_context(
            self,
            retrieved_chunks: List[Dict[str, Any]],
            max_context_length: int = None
    ) -> str:
        """Prepare context string from retrieved chunks"""
        max_context_length = max_context_length or Config.MAX_CONTEXT_LENGTH

        if not retrieved_chunks:
            return "No relevant context found."

        context_parts = []
        total_length = 0

        for i, chunk in enumerate(retrieved_chunks):
            content = chunk['content'].strip()
            source = chunk['metadata'].get('file_name', 'Unknown')

            chunk_text = f"[Source {i + 1}: {source}]\n{content}\n"

            # Check if adding this chunk would exceed max length
            if total_length + len(chunk_text) > max_context_length:
                # Try to fit a truncated version
                remaining_space = max_context_length - total_length - 50  # Leave space for truncation notice
                if remaining_space > 100:  # Only if we have meaningful space
                    truncated = content[:remaining_space] + "..."
                    chunk_text = f"[Source {i + 1}: {source}]\n{truncated}\n"
                    context_parts.append(chunk_text)
                break

            context_parts.append(chunk_text)
            total_length += len(chunk_text)

        context = "\n".join(context_parts)

        self.logger.info(f"Prepared context with {len(context_parts)} chunks ({len(context)} characters)")

        return context

    def get_query_stats(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about query processing"""
        return {
            "query_length": len(query),
            "chunks_retrieved": len(retrieved_chunks),
            "avg_similarity": np.mean(
                [chunk['similarity_score'] for chunk in retrieved_chunks]) if retrieved_chunks else 0,
            "sources": list(set([chunk['metadata'].get('file_name', 'Unknown') for chunk in retrieved_chunks]))
        }