import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for Legal Guardian MVP (single source of truth paths)."""

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

    FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index.bin")
    METADATA_PATH = os.path.join(VECTOR_DB_DIR, "faiss_metadata.json")
    DOCUMENT_MAPPING_PATH = os.path.join(VECTOR_DB_DIR, "document_mapping.json")

    # Default: Groq-supported model (see https://console.groq.com/docs/deprecations)
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

    # Groq 429 / TPM: retry and backoff (on_demand ~6000 TPM needs spacing + patience)
    GROQ_RETRY_MAX_ATTEMPTS = int(os.getenv("GROQ_RETRY_MAX_ATTEMPTS", "12"))
    GROQ_RETRY_BASE_DELAY = float(os.getenv("GROQ_RETRY_BASE_DELAY", "2.0"))
    GROQ_RETRY_MIN_WAIT = float(os.getenv("GROQ_RETRY_MIN_WAIT", "5.0"))
    GROQ_RETRY_MAX_DELAY = float(os.getenv("GROQ_RETRY_MAX_DELAY", "120.0"))

    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))

    ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

    # Risk analysis: larger chunks + batching = fewer API calls (indexing still uses CHUNK_SIZE)
    RISK_CHUNK_SIZE = int(os.getenv("RISK_CHUNK_SIZE", "8000"))
    RISK_CHUNK_OVERLAP = int(os.getenv("RISK_CHUNK_OVERLAP", "200"))
    MAX_RISK_CHUNKS = int(os.getenv("MAX_RISK_CHUNKS", "12"))
    RISK_BATCH_SIZE = int(os.getenv("RISK_BATCH_SIZE", "3"))
    RISK_MAX_TOKENS = int(os.getenv("RISK_MAX_TOKENS", "1200"))
    RISK_CHUNK_DELAY_SEC = float(os.getenv("RISK_CHUNK_DELAY_SEC", "2.0"))

    # Optional web augmentation (Ask page — DuckDuckGo text search)
    WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
    # DuckDuckGo region code — biases snippet language (e.g. us-en = US English, cn-zh = Chinese)
    WEB_SEARCH_REGION = os.getenv("WEB_SEARCH_REGION", "us-en")

    # Optional integrations
    GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "")
    GOOGLE_SPREADSHEET_ID = os.getenv("GOOGLE_SPREADSHEET_ID", "")

    @staticmethod
    def create_directories():
        dirs = [
            Config.DATA_DIR,
            Config.RAW_DATA_DIR,
            Config.PROCESSED_DATA_DIR,
            Config.VECTOR_DB_DIR,
            Config.MODELS_DIR,
            Config.LOGS_DIR,
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
