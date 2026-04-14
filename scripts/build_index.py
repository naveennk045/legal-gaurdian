"""Build FAISS index from files in data/raw/."""

import logging
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config import Config
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingGenerator
from src.text_splitter import OptimizedTextSplitter
from src.vector_store import FAISSVectorStore


def setup_logging():
    Config.create_directories()
    log_file = os.path.join(
        Config.LOGS_DIR,
        f"build_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def run_build_index() -> dict:
    """
    Load data/raw, chunk, embed, save FAISS index.
    Returns stats dict or raises on failure.
    """
    logger = setup_logging()
    logger.info("=== Starting index build ===")

    loader = DocumentLoader()
    documents = loader.load_documents(Config.RAW_DATA_DIR)

    if not documents:
        raise FileNotFoundError("No documents in data/raw/. Add PDF, TXT, or DOCX files.")

    splitter = OptimizedTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    stats = splitter.get_chunk_stats(chunks)
    logger.info("Chunk statistics: %s", stats)

    embedding_generator = EmbeddingGenerator(
        model_name=Config.EMBEDDING_MODEL,
        cache_dir=Config.MODELS_DIR,
    )
    texts = [c.page_content for c in chunks]
    embeddings = embedding_generator.generate_embeddings(texts)

    dimension = embedding_generator.get_embedding_dimension()
    vector_store = FAISSVectorStore(
        dimension=dimension,
        index_path=Config.FAISS_INDEX_PATH,
        metadata_path=Config.METADATA_PATH,
    )
    vector_store.add_embeddings(embeddings, chunks)
    vector_store.save_index()

    store_stats = vector_store.get_stats()
    logger.info("=== Index build complete: %s ===", store_stats)
    return {"chunk_stats": stats, "vector_store": store_stats}


def main():
    logger = setup_logging()
    try:
        run_build_index()
    except Exception as e:
        logger.error("Index build failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
