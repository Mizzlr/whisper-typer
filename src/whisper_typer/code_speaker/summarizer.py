"""Ollama-based text summarization for TTS output."""

import logging
import re
import time

from ..ollama_client import OllamaClient

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """Summarize this in 1-2 short sentences suitable for text-to-speech. Be concise and conversational. Output ONLY the summary, nothing else.

Text: {text}

Summary:"""

MAX_SUMMARIZE_CHARS = 2000


class OllamaSummarizer:
    """Summarize long text via Ollama for concise TTS output."""

    def __init__(
        self, model: str = "qwen2.5:1.5b", host: str = "http://127.0.0.1:11434"
    ):
        self.model = model
        self._client = OllamaClient(host=host)

    async def summarize(self, text: str) -> tuple[str, float]:
        """Summarize text via Ollama. Returns (summary, latency_ms)."""
        t0 = time.perf_counter()

        result = await self._client.generate(
            model=self.model,
            prompt=SUMMARIZE_PROMPT.format(text=text[:MAX_SUMMARIZE_CHARS]),
            temperature=0.3,
            num_predict=200,
        )

        latency = (time.perf_counter() - t0) * 1000

        if result:
            logger.debug(f"Ollama summary ({latency:.0f}ms): {result}")
            return result, latency

        return self._fallback_truncate(text), latency

    @staticmethod
    def _fallback_truncate(text: str) -> str:
        """Extract first 2 sentences as a fallback summary."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return " ".join(sentences[:2])

    async def close(self):
        await self._client.close()
