import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_pipeline import RAGPipeline


def test_pipeline():
    """Test the complete RAG pipeline"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Testing RAG pipeline...")

    try:
        # Initialize pipeline
        pipeline = RAGPipeline()
        pipeline.initialize()
        logger.info("✅ Pipeline initialized successfully")

        # Test queries
        test_queries = [
            "What is the main topic of the documents?",
            "Can you provide a summary?",
            "What are the key points discussed?"
        ]

        for i, query in enumerate(test_queries, 1):
            logger.info(f"Testing query {i}: {query}")

            result = pipeline.answer_query(query, stream=False, include_sources=True)

            print(f"\n{'=' * 50}")
            print(f"Query {i}: {query}")
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Sources found: {len(result.get('sources', []))}")
            print(f"Processing time: {result['query_stats'].get('processing_time', 0):.2f}s")

        # Get pipeline stats
        stats = pipeline.get_pipeline_stats()
        print(f"\n{'=' * 50}")
        print("Pipeline Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        logger.info("✅ All tests completed successfully!")

    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_pipeline()
