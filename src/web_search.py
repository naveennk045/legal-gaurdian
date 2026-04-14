"""Lightweight web context for RAG (DuckDuckGo text search — no API key)."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from config.config import Config

logger = logging.getLogger(__name__)


def fetch_web_context(
    query: str,
    max_results: int = 5,
    region: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (plain-text block for the prompt, structured rows for UI).
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        msg = "Install duckduckgo-search: pip install duckduckgo-search"
        logger.warning(msg)
        return msg, []

    lines: List[str] = []
    web_sources: List[Dict[str, Any]] = []

    reg = region if region is not None else Config.WEB_SEARCH_REGION
    try:
        with DDGS() as ddgs:
            # region= e.g. us-en / uk-en for English; omitting often follows IP/locale (e.g. Chinese).
            results = list(ddgs.text(query, region=reg, max_results=max_results))
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return f"(Web search failed: {e})", []

    for i, r in enumerate(results):
        title = (r.get("title") or "").strip()
        body = (
            (r.get("body") or r.get("snippet") or r.get("description") or "") or ""
        ).strip()
        href = (r.get("href") or r.get("url") or "").strip()
        if not body and title:
            body = title  # Bing/DuckDuckGo sometimes omit body; keep row usable
        snippet = body[:400] + ("..." if len(body) > 400 else "")
        lines.append(f"{i + 1}. {title}\n   {snippet}\n   Link: {href}")
        web_sources.append(
            {
                "title": title or "Result",
                "url": href,
                "snippet": snippet,
            }
        )

    if not lines:
        return "(No web results.)", []

    block = (
        "External web results (public web — not your documents; verify sources):\n"
        + "\n\n".join(lines)
    )
    return block, web_sources
