import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config
from src.embeddings import EmbeddingGenerator


def test_embeddings():
    """Test embedding generation"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Testing embedding generation...")

    # Initialize embedding generator
    embedding_gen = EmbeddingGenerator(
        model_name=Config.EMBEDDING_MODEL,
        cache_dir=Config.MODELS_DIR
    )

    # Test texts
    test_texts = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Python is a programming language.",
        "The weather is sunny today."
    ]

    # Generate embeddings
    embeddings = embedding_gen.generate_embeddings(test_texts)

    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    logger.info(f"Embedding dimension: {embedding_gen.get_embedding_dimension()}")

    # Test single encoding
    single_embedding = embedding_gen.encode_single("Test single encoding")
    logger.info(f"Single embedding shape: {single_embedding.shape}")

    logger.info("✅ Embedding test completed successfully!")


if __name__ == "__main__":
    test_embeddings()
