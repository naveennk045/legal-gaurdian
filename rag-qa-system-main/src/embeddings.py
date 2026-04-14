import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import os


class EmbeddingGenerator:
    """Handles text embedding generation using sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        self.model = None

    def initialize_model(self):
        """Initialize the sentence transformer model"""
        if self.model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir
            )
            self.logger.info("Embedding model loaded successfully")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.model is None:
            self.initialize_model()

        self.logger.info(f"Generating embeddings for {len(texts)} texts...")

        # Generate embeddings in batches to manage memory
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self.logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        if self.model is None:
            self.initialize_model()
        return self.model.get_sentence_embedding_dimension()

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string"""
        if self.model is None:
            self.initialize_model()
        return self.model.encode([text])[0]