"""Ollama-based text summarization for TTS output."""

import logging
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """Summarize this in 1-2 short sentences suitable for text-to-speech. Be concise and conversational. Output ONLY the summary, nothing else.

Text: {text}

Summary:"""


class OllamaSummarizer:
    """Summarize long text via Ollama for concise TTS output."""

    def __init__(self, model: str = "qwen2.5:1.5b", host: str = "http://127.0.0.1:11434"):
        self.model = model
        self.host = host
        self.client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self):
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=30.0)

    async def summarize(self, text: str) -> tuple[str, float]:
        """Summarize text via Ollama.

        Returns (summary, latency_ms). Falls back to truncation if Ollama unavailable.
        """
        await self._ensure_client()

        import time
        t0 = time.perf_counter()

        try:
            response = await self.client.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": SUMMARIZE_PROMPT.format(text=text[:2000]),
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            summary = result.get("response", "").strip()
            latency = (time.perf_counter() - t0) * 1000

            if summary:
                logger.debug(f"Ollama summary ({latency:.0f}ms): {summary}")
                return summary, latency

        except httpx.ConnectError:
            logger.warning(f"Cannot connect to Ollama at {self.host}, using fallback")
        except httpx.TimeoutException:
            logger.warning("Ollama timeout, using fallback")
        except Exception as e:
            logger.warning(f"Ollama summarization failed: {e}, using fallback")

        # Fallback: extract first 2 sentences
        latency = (time.perf_counter() - t0) * 1000
        return self._fallback_truncate(text), latency

    @staticmethod
    def _fallback_truncate(text: str) -> str:
        """Extract first 2 sentences as a fallback summary."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return " ".join(sentences[:2])

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None
