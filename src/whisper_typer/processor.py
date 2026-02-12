"""Ollama text processing for grammar and punctuation."""

import logging

from .config import OllamaConfig
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Fix this speech transcription. Correct:
- Grammar and punctuation
- Misspelled names
- Technical terms
- Every sentence must end with a full stop or question mark

Output ONLY the corrected text, nothing else.

Text: {text}

Corrected:"""


class OllamaProcessor:
    """Process transcribed text with Ollama for grammar/punctuation fixes."""

    def __init__(self, config: OllamaConfig):
        self.config = config
        self._client = OllamaClient(host=config.host)

    async def process(self, text: str) -> str:
        """Process text through Ollama for grammar fixes."""
        if not self.config.enabled or not text.strip():
            return text

        prompt = PROMPT_TEMPLATE.format(text=text)
        logger.debug(f"Sending to Ollama model '{self.config.model}': {text}")

        result = await self._client.generate(
            model=self.config.model,
            prompt=prompt,
            temperature=0.1,
            num_predict=500,
        )

        if result:
            logger.debug(f"Ollama output: '{result}'")
            return result

        logger.warning("Ollama returned no result, using original text")
        return text

    async def check_model_available(self) -> bool:
        """Check if the configured model is available in Ollama."""
        model_names = await self._client.list_models()
        model_base = self.config.model.split(":")[0]
        return any(
            self.config.model in name or model_base in name for name in model_names
        )

    async def close(self):
        """Close HTTP client."""
        await self._client.close()
