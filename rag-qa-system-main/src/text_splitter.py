import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class OptimizedTextSplitter:
    """Optimized text splitter for Q&A applications"""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)

        # Initialize the splitter with optimized settings for Q&A
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into optimized chunks for Q&A"""
        self.logger.info(f"Splitting {len(documents)} documents...")

        chunks = self.text_splitter.split_documents(documents)

        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content),
                "chunk_index": i
            })

        self.logger.info(f"Created {len(chunks)} chunks (avg size: {self._get_average_chunk_size(chunks)} chars)")

        return chunks

    def _get_average_chunk_size(self, chunks: List[Document]) -> int:
        """Calculate average chunk size"""
        if not chunks:
            return 0
        return sum(len(chunk.page_content) for chunk in chunks) // len(chunks)

    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """Get statistics about the chunks"""
        if not chunks:
            return {"count": 0, "avg_size": 0, "min_size": 0, "max_size": 0}

        sizes = [len(chunk.page_content) for chunk in chunks]
        return {
            "count": len(chunks),
            "avg_size": sum(sizes) // len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_chars": sum(sizes)
        }