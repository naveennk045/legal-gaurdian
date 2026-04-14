import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_pipeline import RAGPipeline
from config.config import Config


def setup_logging():
    """Setup logging for CLI"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def print_separator():
    """Print a nice separator"""
    print("=" * 80)


def print_response(result):
    """Print the response in a formatted way"""
    print_separator()
    print("🤖 ANSWER:")
    print(result["answer"])

    if result.get("sources"):
        print("\n📚 SOURCES:")
        for source in result["sources"]:
            print(f"  [{source['source_id']}] {source['file_name']} (similarity: {source['similarity_score']})")
            print(f"      Preview: {source['chunk_preview']}")

    stats = result.get("query_stats", {})
    print(f"\n⚡ STATS: {stats.get('chunks_retrieved', 0)} chunks retrieved in {stats.get('processing_time', 0):.2f}s")
    print_separator()


def interactive_mode(pipeline):
    """Run in interactive mode"""
    print("🚀 RAG Q&A System - Interactive Mode")
    print("Type 'quit' to exit, 'stats' to see pipeline statistics")
    print_separator()

    while True:
        try:
            query = input("\n💬 Your question: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if query.lower() == 'stats':
                stats = pipeline.get_pipeline_stats()
                print("\n📊 PIPELINE STATISTICS:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue

            if not query:
                continue

            print("\n🔍 Processing your query...")
            result = pipeline.answer_query(query, stream=False, include_sources=True)
            print_response(result)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


def single_query_mode(pipeline, query):
    """Handle a single query"""
    result = pipeline.answer_query(query, stream=False, include_sources=True)
    print_response(result)


def main():
    """Main CLI function"""
    logger = setup_logging()

    # Parse command line arguments
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        mode = "single"
    else:
        mode = "interactive"

    try:
        # Initialize pipeline
        print("🔧 Initializing RAG pipeline...")
        pipeline = RAGPipeline()
        pipeline.initialize()
        print("✅ Pipeline ready!")

        # Run based on mode
        if mode == "single":
            single_query_mode(pipeline, query)
        else:
            interactive_mode(pipeline)

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        print(f"❌ Error: {str(e)}")
        print("\n🔧 Make sure you have:")
        print("  1. Built the vector index (python scripts/build_index.py)")
        print("  2. Set your GROQ_API_KEY in .env file")
        print("  3. Installed all requirements (pip install -r requirements.txt)")
        sys.exit(1)


if __name__ == "__main__":
    main()
