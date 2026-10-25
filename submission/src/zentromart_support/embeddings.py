"""Gemini text embeddings used by hybrid evidence retrieval."""

from __future__ import annotations

import os
import time
import unicodedata


class GeminiEmbeddingModel:
    """Create normalized retrieval embeddings through the Gemini API."""

    REQUESTS_PER_MINUTE_BUDGET = 80

    def __init__(self, model_name: str | None = None) -> None:
        from dotenv import load_dotenv

        load_dotenv()
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise RuntimeError("Set GEMINI_API_KEY before using embedding retrieval.")

        from google import genai
        from google.genai import types

        self._types = types
        self.model_name = model_name or os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        )
        self._client = genai.Client(
            http_options=types.HttpOptions(
                timeout=60_000,
                retry_options=types.HttpRetryOptions(attempts=3),
            )
        )
        self._last_request_at = 0.0
        self._last_batch_size = 0

    def embed_many(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        """Embed a list with the retrieval-specific document/query task type."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            # Gemini's quota appears to account for each text in a batch, not
            # only the HTTP call. Pace by input count, including across the
            # document and query embedding stages.
            interval = self._last_batch_size * 60 / self.REQUESTS_PER_MINUTE_BUDGET
            wait = interval - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            batch = texts[start:start + 32]
            result = self._client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=self._types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=768,
                ),
            )
            self._last_request_at = time.monotonic()
            self._last_batch_size = len(batch)
            for embedding in result.embeddings or []:
                raw = [float(value) for value in embedding.values]
                norm = sum(value * value for value in raw) ** 0.5
                vectors.append([value / norm for value in raw] if norm else raw)
            if len(vectors) != min(start + len(batch), len(texts)):
                raise RuntimeError("Gemini returned a different number of embeddings than inputs.")
        return vectors


def normalize_cache_key(text: str) -> str:
    """Normalize only Unicode representation; preserve words and identifiers."""
    return unicodedata.normalize("NFC", text)
