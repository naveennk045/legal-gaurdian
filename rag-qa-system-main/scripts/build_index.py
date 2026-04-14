import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config
from src.document_loader import DocumentLoader
from src.text_splitter import OptimizedTextSplitter
from src.embeddings import EmbeddingGenerator
from src.vector_store import FAISSVectorStore


def setup_logging():
    """Setup logging configuration"""
    Config.create_directories()

    log_file = os.path.join(Config.LOGS_DIR, f'build_index_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def main():
    """Main pipeline for building the vector index"""
    logger = setup_logging()
    logger.info("=== Starting RAG Index Building Pipeline (Phase 1) ===")

    try:
        # Step 1: Load Documents
        logger.info("Step 1: Loading documents...")
        loader = DocumentLoader()
        documents = loader.load_documents(Config.RAW_DATA_DIR)

        if not documents:
            logger.error("No documents found! Please add documents to data/raw/ directory")
            return

        logger.info(f"Loaded {len(documents)} documents")

        # Step 2: Split Documents into Chunks
        logger.info("Step 2: Splitting documents into chunks...")
        splitter = OptimizedTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(documents)

        # Log chunk statistics
        stats = splitter.get_chunk_stats(chunks)
        logger.info(f"Chunk Statistics: {stats}")

        # Step 3: Generate Embeddings
        logger.info("Step 3: Generating embeddings...")
        embedding_generator = EmbeddingGenerator(
            model_name=Config.EMBEDDING_MODEL,
            cache_dir=Config.MODELS_DIR
        )

        # Extract text content from chunks
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_generator.generate_embeddings(texts)

        # Step 4: Create and Populate Vector Store
        logger.info("Step 4: Creating vector store...")
        dimension = embedding_generator.get_embedding_dimension()

        vector_store = FAISSVectorStore(
            dimension=dimension,
            index_path=Config.FAISS_INDEX_PATH,
            metadata_path=Config.METADATA_PATH
        )

        vector_store.add_embeddings(embeddings, chunks)

        # Step 5: Save Vector Store
        logger.info("Step 5: Saving vector store...")
        vector_store.save_index()

        # Final Statistics
        store_stats = vector_store.get_stats()
        logger.info(f"Vector Store Statistics: {store_stats}")

        logger.info("=== Index Building Complete! ===")
        logger.info(f"Created index with {store_stats['total_vectors']} vectors")
        logger.info(f"Index saved to: {Config.FAISS_INDEX_PATH}")
        logger.info(f"Metadata saved to: {Config.METADATA_PATH}")

    except Exception as e:
        logger.error(f"Error in index building pipeline: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

