import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Text embedding generation using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        self.model = None

    def initialize_model(self):
        if self.model is None:
            self.logger.info("Loading embedding model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            self.logger.info("Embedding model loaded successfully")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            self.initialize_model()

        self.logger.info("Generating embeddings for %s texts...", len(texts))

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        self.logger.info("Generated embeddings shape: %s", embeddings.shape)
        return embeddings

    def get_embedding_dimension(self) -> int:
        if self.model is None:
            self.initialize_model()
        return self.model.get_sentence_embedding_dimension()

    def encode_single(self, text: str) -> np.ndarray:
        if self.model is None:
            self.initialize_model()
        return self.model.encode([text])[0]
