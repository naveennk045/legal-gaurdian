import json
import logging
import os
from typing import Any, Dict, List

import faiss
import numpy as np
from langchain_core.documents import Document


class FAISSVectorStore:
    """FAISS-based vector store for similarity search."""

    def __init__(self, dimension: int, index_path: str = None, metadata_path: str = None):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.logger = logging.getLogger(__name__)

        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        self.id_to_metadata = {}

    def add_embeddings(self, embeddings: np.ndarray, documents: List[Document]):
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        self.index.add(embeddings.astype("float32"))

        start_id = len(self.metadata)
        for i, doc in enumerate(documents):
            doc_metadata = {
                "id": start_id + i,
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            self.metadata.append(doc_metadata)
            self.id_to_metadata[start_id + i] = doc_metadata

        self.logger.info(
            "Added %s embeddings. Total vectors: %s",
            len(embeddings),
            self.index.ntotal,
        )

    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        query_vector = query_embedding.reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                result = {
                    "id": int(idx),
                    "distance": float(distances[0][i]),
                    "similarity_score": 1 / (1 + distances[0][i]),
                    "content": self.metadata[idx]["content"],
                    "metadata": self.metadata[idx]["metadata"],
                }
                results.append(result)

        return results

    def save_index(self, index_path: str = None, metadata_path: str = None):
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path

        if not index_path or not metadata_path:
            raise ValueError("Index path and metadata path must be provided")

        faiss.write_index(self.index, index_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        self.logger.info("Saved index to %s and metadata to %s", index_path, metadata_path)

    def load_index(self, index_path: str = None, metadata_path: str = None):
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path

        if not index_path or not metadata_path:
            raise ValueError("Index path and metadata path must be provided")
        if not os.path.isfile(index_path) or not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                "Vector index files are missing. Add documents to data/raw/, then run:\n"
                "  python scripts/build_index.py"
            )

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.id_to_metadata = {item["id"]: item for item in self.metadata}

        self.logger.info(
            "Loaded index with %s vectors and %s metadata entries",
            self.index.ntotal,
            len(self.metadata),
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_metadata": len(self.metadata),
            "index_type": type(self.index).__name__,
        }
