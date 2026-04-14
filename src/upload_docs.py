"""Load LangChain Documents from Streamlit UploadedFile objects."""

import os
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.document_loader import DocumentLoader


def documents_from_uploaded_files(files) -> List[Document]:
    """Write uploads to temp files and reuse DocumentLoader (PDF/TXT/DOCX)."""
    if not files:
        return []

    loader = DocumentLoader()
    docs: List[Document] = []
    allowed = {".pdf", ".txt", ".docx", ".doc"}

    for f in files:
        name = getattr(f, "name", "upload") or "upload"
        suffix = Path(name).suffix.lower()
        if suffix not in allowed:
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getbuffer())
            path = tmp.name
        try:
            docs.append(loader.load_file(path))
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    return docs
