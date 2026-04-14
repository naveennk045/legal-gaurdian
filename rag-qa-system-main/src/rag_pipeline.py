import logging
from typing import Dict, Any, Optional, Iterator, List
import time
from src.embeddings import EmbeddingGenerator
from src.vector_store import FAISSVectorStore
from src.query_processor import QueryProcessor
from src.llm_client import GroqLLMClient
from config.config import Config


class RAGPipeline:
    """Complete RAG pipeline orchestrator"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embedding_generator = None
        self.vector_store = None
        self.query_processor = None
        self.llm_client = None

    def initialize(self):
        """Initialize all components"""
        self.logger.info("Initializing RAG pipeline...")

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            model_name=Config.EMBEDDING_MODEL,
            cache_dir=Config.MODELS_DIR
        )
        self.embedding_generator.initialize_model()

        # Initialize vector store
        dimension = self.embedding_generator.get_embedding_dimension()
        self.vector_store = FAISSVectorStore(
            dimension=dimension,
            index_path=Config.FAISS_INDEX_PATH,
            metadata_path=Config.METADATA_PATH
        )

        # Load existing index
        try:
            self.vector_store.load_index()
            self.logger.info("Loaded existing vector index")
        except Exception as e:
            self.logger.error(f"Failed to load vector index: {e}")
            raise

        # Initialize query processor
        self.query_processor = QueryProcessor(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator
        )

        # Initialize LLM client
        self.llm_client = GroqLLMClient()

        # Test LLM connection
        if not self.llm_client.check_connection():
            raise Exception("Failed to connect to Groq API")

        self.logger.info("RAG pipeline initialized successfully")

    def answer_query(
            self,
            query: str,
            stream: bool = False,
            include_sources: bool = True
    ) -> Dict[str, Any]:
        """Answer a query using the complete RAG pipeline"""
        start_time = time.time()

        try:
            # Step 1: Process query and retrieve context
            retrieved_chunks = self.query_processor.process_query(query)

            if not retrieved_chunks:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "sources": [],
                    "query_stats": {"processing_time": time.time() - start_time}
                }

            # Step 2: Prepare context
            context = self.query_processor.prepare_context(retrieved_chunks)

            # Step 3: Create prompt
            prompt = self._create_rag_prompt(query, context)
            system_prompt = self._get_system_prompt()

            # Step 4: Generate response
            if stream and Config.ENABLE_STREAMING:
                # Return iterator for streaming
                def stream_with_metadata():
                    response_chunks = []
                    for chunk in self.llm_client.generate_response(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            stream=True
                    ):
                        response_chunks.append(chunk)
                        yield chunk

                    # After streaming is complete, you might want to log the full response
                    full_response = ''.join(response_chunks)
                    self.logger.info(f"Completed streaming response ({len(full_response)} chars)")

                return {
                    "answer_stream": stream_with_metadata(),
                    "sources": self._format_sources(retrieved_chunks) if include_sources else [],
                    "query_stats": self.query_processor.get_query_stats(query, retrieved_chunks)
                }
            else:
                answer = self.llm_client.generate_response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    stream=False
                )

                processing_time = time.time() - start_time

                result = {
                    "answer": answer,
                    "sources": self._format_sources(retrieved_chunks) if include_sources else [],
                    "query_stats": {
                        **self.query_processor.get_query_stats(query, retrieved_chunks),
                        "processing_time": processing_time
                    }
                }

                self.logger.info(f"Query answered in {processing_time:.2f}s")
                return result

        except Exception as e:
            self.logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "query_stats": {"processing_time": time.time() - start_time, "error": str(e)}
            }

    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create the prompt for the LLM with context"""
        prompt = f"""Based on the following context information, please answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please say so and provide what information you can based on the available context.

Answer:"""

        return prompt

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return """You are a helpful AI assistant that answers questions based on provided context. 

Guidelines:
- Answer based primarily on the provided context
- Be accurate and factual
- If information is not in the context, acknowledge this
- Provide clear, well-structured responses
- Use specific details from the context when relevant
- If asked about sources, refer to the document names mentioned in the context"""

    def _format_sources(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information for the response"""
        sources = []
        seen_sources = set()

        for i, chunk in enumerate(retrieved_chunks):
            source_name = chunk['metadata'].get('file_name', 'Unknown')
            source_key = f"{source_name}_{chunk['metadata'].get('chunk_id', i)}"

            if source_key not in seen_sources:
                sources.append({
                    "source_id": i + 1,
                    "file_name": source_name,
                    "similarity_score": round(chunk['similarity_score'], 3),
                    "chunk_preview": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                })
                seen_sources.add(source_key)

        return sources

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline"""
        return {
            "vector_store_stats": self.vector_store.get_stats() if self.vector_store else {},
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.GROQ_MODEL,
            "top_k_retrieval": Config.TOP_K_RETRIEVAL,
            "similarity_threshold": Config.SIMILARITY_THRESHOLD
        }
