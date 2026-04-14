import logging
from typing import Iterator, Optional, Dict, Any
import time
from groq import Groq
from config.config import Config


class GroqLLMClient:
    """Groq LLM client for text generation with streaming support"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or Config.GROQ_API_KEY
        self.model = model or Config.GROQ_MODEL
        self.logger = logging.getLogger(__name__)

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=self.api_key)
        self.logger.info(f"Initialized Groq client with model: {self.model}")

    def generate_response(
            self,
            prompt: str,
            system_prompt: str = None,
            max_tokens: int = None,
            temperature: float = None,
            stream: bool = False
    ) -> str:
        """Generate a response from the LLM"""
        try:
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Add user prompt
            messages.append({
                "role": "user",
                "content": prompt
            })

            # Set parameters
            max_tokens = max_tokens or Config.MAX_TOKENS
            temperature = temperature or Config.TEMPERATURE

            self.logger.info(f"Generating response with {self.model}")

            if stream:
                return self._stream_response(messages, max_tokens, temperature)
            else:
                return self._generate_complete_response(messages, max_tokens, temperature)

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    def _generate_complete_response(self, messages, max_tokens, temperature) -> str:
        """Generate complete response (non-streaming)"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )

        return response.choices[0].message.content

    def _stream_response(self, messages, max_tokens, temperature) -> Iterator[str]:
        """Generate streaming response"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def check_connection(self) -> bool:
        """Test connection to Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False
