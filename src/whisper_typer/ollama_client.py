"""Shared async Ollama API client."""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async client for Ollama /api/generate calls with shared error handling."""

    def __init__(self, host: str = "http://localhost:11434", timeout: float = 30.0):
        self.host = host
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.1,
        num_predict: int = 500,
    ) -> Optional[str]:
        """Call Ollama /api/generate. Returns response text, or None on any failure."""
        await self._ensure_client()

        try:
            response = await self._client.post(
                f"{self.host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": num_predict,
                    },
                },
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            return text if text else None

        except httpx.ConnectError:
            logger.warning(f"Cannot connect to Ollama at {self.host}")
            return None
        except httpx.TimeoutException:
            logger.warning("Ollama request timed out")
            return None
        except Exception as e:
            logger.warning(f"Ollama request failed: {e}")
            return None

    async def list_models(self) -> list[str]:
        """List available model names from Ollama."""
        await self._ensure_client()

        try:
            response = await self._client.get(f"{self.host}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m.get("name", "") for m in models]
        except Exception as e:
            logger.warning(f"Cannot check Ollama models: {e}")
            return []

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
