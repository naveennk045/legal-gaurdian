import logging
from pathlib import Path
from typing import List, Union

import docx2txt
import pypdf
from langchain_core.documents import Document


class DocumentLoader:
    """Handles loading of different document types (PDF, TXT, DOCX)."""

    def __init__(self):
        self.supported_extensions = {".pdf", ".txt", ".docx", ".doc"}
        self.logger = logging.getLogger(__name__)

    def load_documents(self, data_dir: str) -> List[Document]:
        """Load all supported documents from a directory."""
        documents = []
        data_path = Path(data_dir)

        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        files = []
        for ext in self.supported_extensions:
            files.extend(data_path.glob(f"*{ext}"))

        self.logger.info("Found %s documents to process", len(files))

        for file_path in files:
            try:
                doc = self._load_single_document(file_path)
                if doc:
                    documents.append(doc)
                    self.logger.info("Loaded: %s", file_path.name)
            except Exception as e:
                self.logger.error("Error loading %s: %s", file_path.name, e)

        return documents

    def load_file(self, file_path: Union[str, Path]) -> Document:
        """Load a single file from disk (used after upload)."""
        return self._load_single_document(Path(file_path))

    def _load_single_document(self, file_path: Path) -> Document:
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return self._load_pdf(file_path)
        if extension == ".txt":
            return self._load_txt(file_path)
        if extension in [".docx", ".doc"]:
            return self._load_docx(file_path)
        raise ValueError(f"Unsupported file type: {extension}")

    def _load_pdf(self, file_path: Path) -> Document:
        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

        return Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_type": "pdf",
                "file_name": file_path.name,
            },
        )

    def _load_txt(self, file_path: Path) -> Document:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_type": "txt",
                "file_name": file_path.name,
            },
        )

    def _load_docx(self, file_path: Path) -> Document:
        text = docx2txt.process(str(file_path))

        return Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_type": "docx",
                "file_name": file_path.name,
            },
        )
