"""Small Gemini API adapter used by the agent."""

from __future__ import annotations

import os
import time


class GeminiTextModel:
    """Text generation client; credentials are read from the environment."""

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError(
                "Project dependencies are missing. Run `python -m pip install -e .`."
            ) from exc
        load_dotenv()

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support is not installed. Run `python -m pip install -e .`."
            ) from exc

        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise RuntimeError(
                "Set GEMINI_API_KEY in your environment before calling Gemini."
            )

        # Keep smoke tests and batch runs bounded when the service stalls.
        self._types = types
        self._client = genai.Client(
            http_options=types.HttpOptions(
                timeout=60_000,
                retry_options=types.HttpRetryOptions(attempts=3),
            )
        )
        self.model_name = model_name or os.getenv(
            "GEMINI_MODEL", "gemini-3.5-flash-lite"
        )
        self._min_request_interval = float(
            os.getenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "4.5")
        )
        self._last_request_at = 0.0

    def generate(self, *, system_instruction: str, prompt: str) -> str:
        wait = self._min_request_interval - (time.monotonic() - self._last_request_at)
        if self._last_request_at and wait > 0:
            time.sleep(wait)
        response = self._client.models.generate_content(
            model=self.model_name,
            config=self._types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
            contents=prompt,
        )
        self._last_request_at = time.monotonic()
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty text response.")
        return text
