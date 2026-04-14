import json
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from langchain.schema import Document


class FAISSVectorStore:
    """FAISS-based vector store for similarity search"""

    def __init__(self, dimension: int, index_path: str = None, metadata_path: str = None):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.logger = logging.getLogger(__name__)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        self.id_to_metadata = {}

    def add_embeddings(self, embeddings: np.ndarray, documents: List[Document]):
        """Add embeddings and their corresponding metadata to the vector store"""
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        # Add vectors to FAISS index
        self.index.add(embeddings.astype('float32'))

        # Store metadata
        start_id = len(self.metadata)
        for i, doc in enumerate(documents):
            doc_metadata = {
                "id": start_id + i,
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            self.metadata.append(doc_metadata)
            self.id_to_metadata[start_id + i] = doc_metadata

        self.logger.info(f"Added {len(embeddings)} embeddings. Total vectors: {self.index.ntotal}")

    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Perform similarity search and return top-k results"""
        if self.index.ntotal == 0:
            return []

        # Ensure query_embedding is the right shape and type
        query_vector = query_embedding.reshape(1, -1).astype('float32')

        # Search
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))

        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 indicates no match found
                result = {
                    "id": int(idx),
                    "distance": float(distances[0][i]),
                    "similarity_score": 1 / (1 + distances[0][i]),  # Convert distance to similarity
                    "content": self.metadata[idx]["content"],
                    "metadata": self.metadata[idx]["metadata"]
                }
                results.append(result)

        return results

    def save_index(self, index_path: str = None, metadata_path: str = None):
        """Save FAISS index and metadata to disk"""
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path

        if not index_path or not metadata_path:
            raise ValueError("Index path and metadata path must be provided")

        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")

    def load_index(self, index_path: str = None, metadata_path: str = None):
        """Load FAISS index and metadata from disk"""
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # Rebuild id_to_metadata mapping
        self.id_to_metadata = {item["id"]: item for item in self.metadata}

        self.logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_metadata": len(self.metadata),
            "index_type": type(self.index).__name__
        }