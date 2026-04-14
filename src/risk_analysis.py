import json
import logging
import re
import time
from typing import Any, Dict, List

from langchain_core.documents import Document

from config.config import Config
from src.llm_client import GroqLLMClient


def _extract_json_array(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM output into a list of {risks, recommendations}."""
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Expected JSON array")
    return data


def detect_risks_and_recommendations(
    chunks: List[Document],
    llm_client: GroqLLMClient,
    max_chunks: int = None,
) -> List[Dict[str, Any]]:
    """
    Risk / recommendation analysis with batched Groq calls (fewer round trips than one-call-per-chunk).
    """
    logger = logging.getLogger(__name__)
    max_chunks = max_chunks if max_chunks is not None else Config.MAX_RISK_CHUNKS
    batch_size = max(1, Config.RISK_BATCH_SIZE)
    delay = Config.RISK_CHUNK_DELAY_SEC
    risk_max_tokens = Config.RISK_MAX_TOKENS

    if len(chunks) > max_chunks:
        logger.warning(
            "Using first %s chunks (of %s) for risk analysis — increase MAX_RISK_CHUNKS in .env for more coverage",
            max_chunks,
            len(chunks),
        )
        chunks = chunks[:max_chunks]

    results: List[Dict[str, Any]] = []
    system_prompt = (
        "You are a legal risk analyst. Reply with ONLY valid JSON as specified, no other text."
    )

    batches: List[List[Document]] = [
        chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)
    ]

    for bi, batch in enumerate(batches):
        if bi > 0 and delay > 0:
            time.sleep(delay)

        numbered = []
        for j, doc in enumerate(batch):
            numbered.append(f"[{j + 1}]\n{doc.page_content}")

        block = "\n\n---\n\n".join(numbered)
        prompt = (
            f"You are given {len(batch)} labeled text sections below.\n"
            f"Return ONLY a JSON array of exactly {len(batch)} objects, in the same order as the sections.\n"
            f'Each object must have keys "risks" and "recommendations" (strings, markdown bullet lists ok).\n\n'
            f"Sections:\n\n{block}"
        )

        raw = llm_client.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=risk_max_tokens,
            stream=False,
        ).strip()

        try:
            parsed = _extract_json_array(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("JSON parse failed (%s), falling back to single-block parse", e)
            results.append(
                {
                    "context": "\n\n".join(c.page_content for c in batch),
                    "risks": raw[:8000],
                    "recommendations": "See risks column — model returned non-JSON; retry or lower RISK_BATCH_SIZE.",
                }
            )
            continue

        if len(parsed) != len(batch):
            logger.warning(
                "Expected %s JSON objects, got %s — padding or truncating",
                len(batch),
                len(parsed),
            )
            while len(parsed) < len(batch):
                parsed.append({"risks": "—", "recommendations": "—"})
            parsed = parsed[: len(batch)]

        for doc, obj in zip(batch, parsed):
            risks = (obj.get("risks") or "No risks identified.").strip()
            recs = (obj.get("recommendations") or "No recommendations provided.").strip()
            results.append(
                {
                    "context": doc.page_content,
                    "risks": risks,
                    "recommendations": recs,
                }
            )

    return results
