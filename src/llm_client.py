import logging
import re
import time
from typing import Iterator, Optional

from groq import Groq, RateLimitError

from config.config import Config


def _error_text(err: Exception) -> str:
    """Groq puts the human message inside `body`; include it for regex/parsing."""
    parts = [str(err)]
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and inner.get("message"):
            parts.append(str(inner["message"]))
    return " ".join(parts)


def _is_rate_limit_error(err: Exception) -> bool:
    if isinstance(err, RateLimitError):
        return True
    code = getattr(err, "status_code", None)
    if code == 429:
        return True
    msg = _error_text(err).lower()
    return "429" in msg or "rate_limit" in msg or "tokens per minute" in msg


def _retry_sleep_seconds(err: Exception, attempt: int) -> float:
    """Prefer server hint 'try again in Xs', else exponential backoff."""
    msg = _error_text(err)
    m = re.search(r"try again in ([\d.]+)\s*s", msg, re.I)
    if m:
        wait = float(m.group(1)) + 1.0
    else:
        base = Config.GROQ_RETRY_BASE_DELAY * (2**attempt)
        wait = base
    wait = max(wait, Config.GROQ_RETRY_MIN_WAIT)
    return min(wait, Config.GROQ_RETRY_MAX_DELAY)


class GroqLLMClient:
    """Groq LLM client with optional streaming and 429 retries."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or Config.GROQ_API_KEY
        self.model = model or Config.GROQ_MODEL
        self.logger = logging.getLogger(__name__)

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=self.api_key)
        self.logger.info("Initialized Groq client with model: %s", self.model)

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
        stream: bool = False,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_tokens = max_tokens or Config.MAX_TOKENS
        temperature = temperature if temperature is not None else Config.TEMPERATURE

        self.logger.info("Generating response with %s", self.model)

        if stream:
            return self._stream_response(messages, max_tokens, temperature)
        return self._generate_complete_response(messages, max_tokens, temperature)

    def _generate_complete_response(self, messages, max_tokens, temperature) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(Config.GROQ_RETRY_MAX_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_err = e
                if not _is_rate_limit_error(e):
                    self.logger.error("Error generating response: %s", e)
                    raise
                if attempt >= Config.GROQ_RETRY_MAX_ATTEMPTS - 1:
                    self.logger.error(
                        "Groq rate limit persists after %s attempts: %s",
                        Config.GROQ_RETRY_MAX_ATTEMPTS,
                        e,
                    )
                    raise
                wait = _retry_sleep_seconds(e, attempt)
                self.logger.warning(
                    "Groq rate limited (429), sleeping %.1fs then retry %s/%s",
                    wait,
                    attempt + 2,
                    Config.GROQ_RETRY_MAX_ATTEMPTS,
                )
                time.sleep(wait)
        assert last_err is not None
        raise last_err

    def _stream_response(self, messages, max_tokens, temperature) -> Iterator[str]:
        stream = None
        last_err: Optional[Exception] = None
        for attempt in range(Config.GROQ_RETRY_MAX_ATTEMPTS):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                break
            except Exception as e:
                last_err = e
                if not _is_rate_limit_error(e):
                    raise
                if attempt >= Config.GROQ_RETRY_MAX_ATTEMPTS - 1:
                    raise
                wait = _retry_sleep_seconds(e, attempt)
                self.logger.warning(
                    "Groq rate limited on stream create, sleeping %.1fs (attempt %s)",
                    wait,
                    attempt + 1,
                )
                time.sleep(wait)
        if stream is None:
            assert last_err is not None
            raise last_err

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def check_connection(self) -> bool:
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
            return True
        except Exception as e:
            self.logger.error("Connection test failed: %s", e)
            return False
