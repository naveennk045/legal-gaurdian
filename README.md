# Legal Guardian

A Streamlit app for **legal-style document work**: ask questions over your files with **RAG** (FAISS + local embeddings + Groq), and run a separate **risk and recommendations** pass with export options. Answers are grounded in retrieved text; this is an assistant tool, not a substitute for professional legal advice.

## What you get

| Area | What it does |
|------|----------------|
| **Ask** | Chat with **indexed files** under `data/raw/` *or* **upload PDF/TXT/DOCX** for a **session-only** index (nothing is written to `data/raw/`). Shows citations, simple metrics, and optional streaming. |
| **Web search (Ask)** | Optional **DuckDuckGo**-backed snippets (no API key). Choose region (e.g. US English) so results match your language. Shown separately from document sources. |
| **Risks** | Analyze uploads or files in `data/raw/`, structured risks table, CSV export, optional **Google Sheets** push, optional **rebuild FAISS** for Ask. |

Subfolders `rag-qa-system-main/` and `legal_document_analyzer-main/` are legacy references; use the **repository root** layout for day-to-day work.

## Requirements

- Python 3.10+ recommended  
- A **[Groq](https://console.groq.com/)** API key (`GROQ_API_KEY`)  
- Disk space for sentence-transformer cache under `models/` (first run downloads the embedding model)

## Quick start

```bash
cd legal-gaurdian
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set **`GROQ_API_KEY`** (required). Optionally adjust `GROQ_MODEL` and `EMBEDDING_MODEL` (see `.env.example`).

### CPU-only PyTorch (optional, Linux)

If you want a smaller CPU wheel before pulling the rest of the stack:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Indexing for Ask (disk library)

Put PDF, TXT, or DOCX files in **`data/raw/`**, then build the FAISS index:

```bash
python scripts/build_index.py
```

After adding or removing files, run the script again. On the **Ask** page, use **Indexed library** and use **Reload pipeline** in the sidebar if the app was already open.

## Run the app

```bash
streamlit run app.py
```

Open the **Ask** and **Risks** pages from the sidebar.

**Ask — two knowledge modes**

1. **Indexed library** — Uses `vector_db/` built from `data/raw/` (requires a successful `build_index.py` run).  
2. **Upload files (this session only)** — Upload files and click **Index uploads for this chat**. Chat uses only those files until you clear the session index.

Enable **Include web search** when you want public-web snippets alongside retrieval. If you see **“Web search rows retrieved: 0”**, the search backend returned no rows (network, rate limits, or blocking); document Q&A still works.

## Command-line query (optional)

```bash
python scripts/query_cli.py
python scripts/query_cli.py "Your question here"
```

Uses the same disk index as **Ask** (indexed library path).

## Configuration

All tunables live in **`.env`**; **`.env.example`** lists every variable with comments. Highlights:

- **Retrieval:** `TOP_K_RETRIEVAL`, `SIMILARITY_THRESHOLD`, `CHUNK_SIZE`, `CHUNK_OVERLAP`  
- **LLM:** `GROQ_MODEL`, `MAX_TOKENS`, `TEMPERATURE`, `ENABLE_STREAMING`  
- **Web (Ask):** `WEB_SEARCH_MAX_RESULTS`, `WEB_SEARCH_REGION` (e.g. `us-en` for English)  
- **Risks:** `RISK_CHUNK_*`, `RISK_BATCH_SIZE`, Groq retry knobs for rate limits  
- **Google Sheets (Risks):** `GOOGLE_SERVICE_ACCOUNT_FILE`, `GOOGLE_SPREADSHEET_ID`

## Optional integrations

- **Google Sheets:** set the two variables above, then use the control on the **Risks** page.

## Troubleshooting

- **Long `torch` / `torchvision` tracebacks at startup** — Install full dependencies: `pip install -r requirements.txt` so optional Streamlit imports resolve.  
- **Ask fails: no FAISS index** — Add files under `data/raw/`, run `python scripts/build_index.py`, reload the pipeline or restart the app. Or switch to **Upload files** and index in-session.  
- **Web search always empty** — Check connectivity; some environments block automated search. Document-only mode does not require web.
