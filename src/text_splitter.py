import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class OptimizedTextSplitter:
    """Text splitter aligned with RAG indexing settings."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        self.logger.info("Splitting %s documents...", len(documents))

        chunks = self.text_splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "chunk_id": i,
                    "chunk_size": len(chunk.page_content),
                    "chunk_index": i,
                }
            )

        self.logger.info(
            "Created %s chunks (avg size: %s chars)",
            len(chunks),
            self._get_average_chunk_size(chunks),
        )

        return chunks

    def _get_average_chunk_size(self, chunks: List[Document]) -> int:
        if not chunks:
            return 0
        return sum(len(chunk.page_content) for chunk in chunks) // len(chunks)

    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        if not chunks:
            return {"count": 0, "avg_size": 0, "min_size": 0, "max_size": 0}

        sizes = [len(chunk.page_content) for chunk in chunks]
        return {
            "count": len(chunks),
            "avg_size": sum(sizes) // len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_chars": sum(sizes),
        }
