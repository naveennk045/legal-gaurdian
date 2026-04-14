"""CLI for RAG queries."""

import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rag_pipeline import RAGPipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def print_response(result):
    print("=" * 80)
    print("ANSWER:")
    print(result["answer"])

    if result.get("sources"):
        print("\nSOURCES:")
        for source in result["sources"]:
            print(
                f"  [{source['source_id']}] {source['file_name']} "
                f"(similarity: {source['similarity_score']})"
            )
            print(f"      Preview: {source['chunk_preview']}")

    stats = result.get("query_stats", {})
    print(
        f"\nSTATS: {stats.get('chunks_retrieved', 0)} chunks in "
        f"{stats.get('processing_time', 0):.2f}s "
        f"(confidence: {stats.get('confidence_band', 'n/a')})"
    )
    print("=" * 80)


def interactive_mode(pipeline):
    print("Legal Guardian — interactive Q&A (quit to exit, stats for pipeline info)")
    print("=" * 80)

    while True:
        try:
            query = input("\nYour question: ").strip()

            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break

            if query.lower() == "stats":
                print(pipeline.get_pipeline_stats())
                continue

            if not query:
                continue

            result = pipeline.answer_query(query, stream=False, include_sources=True)
            print_response(result)

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    logger = setup_logging()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        mode = "single"
    else:
        mode = "interactive"

    try:
        print("Initializing RAG pipeline...")
        pipeline = RAGPipeline()
        pipeline.initialize()
        print("Ready.\n")

        if mode == "single":
            print_response(pipeline.answer_query(query, stream=False, include_sources=True))
        else:
            interactive_mode(pipeline)

    except Exception as e:
        logger.error("Failed: %s", e)
        print(f"Error: {e}")
        print("Build index: python scripts/build_index.py")
        print("Set GROQ_API_KEY in .env")
        sys.exit(1)


if __name__ == "__main__":
    main()
