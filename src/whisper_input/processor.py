"""Ollama text processing for grammar and punctuation."""

import logging
from typing import Optional

import httpx

from .config import OllamaConfig

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Fix any typos, grammar, and punctuation in the following transcribed speech.
Add proper capitalization and punctuation marks.
Preserve the original meaning and tone.
Output ONLY the corrected text, nothing else.

Text to fix:
{text}"""


class OllamaProcessor:
    """Process transcribed text with Ollama for grammar/punctuation fixes."""

    def __init__(self, config: OllamaConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self):
        """Create HTTP client if needed."""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=30.0)

    async def process(self, text: str) -> str:
        """Process text through Ollama for grammar fixes."""
        if not self.config.enabled:
            return text

        if not text.strip():
            return text

        await self._ensure_client()

        prompt = PROMPT_TEMPLATE.format(text=text)

        try:
            logger.debug(f"Sending to Ollama model '{self.config.model}': {text}")
            response = await self.client.post(
                f"{self.config.host}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low for deterministic output
                        "num_predict": 500,  # Limit output length
                    },
                },
            )
            logger.debug(f"Ollama response status: {response.status_code}")
            response.raise_for_status()

            result = response.json()
            logger.debug(f"Ollama raw response: {result}")
            processed_text = result.get("response", "").strip()

            if processed_text:
                logger.info(f"Ollama input:  '{text}'")
                logger.info(f"Ollama output: '{processed_text}'")
                return processed_text
            else:
                logger.warning("Ollama returned empty response, using original text")
                return text

        except httpx.ConnectError:
            logger.warning(
                f"Cannot connect to Ollama at {self.config.host}. "
                "Make sure Ollama is running. Using original text."
            )
            return text
        except httpx.TimeoutException:
            logger.warning("Ollama request timed out. Using original text.")
            return text
        except Exception as e:
            logger.warning(f"Ollama processing failed: {e}. Using original text.")
            return text

    async def check_model_available(self) -> bool:
        """Check if the configured model is available in Ollama."""
        await self._ensure_client()

        try:
            response = await self.client.get(f"{self.config.host}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Check for exact match or match without tag
            model_base = self.config.model.split(":")[0]
            return any(
                self.config.model in name or model_base in name
                for name in model_names
            )
        except Exception as e:
            logger.warning(f"Cannot check Ollama models: {e}")
            return False

    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
