"""Gnani-only speech input/output wrapper for the conversation agent."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .conversation import ConversationAgent, ConversationState, TurnResult, _redact_proofs


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass
class VoiceTurn:
    transcript: str
    result: TurnResult
    audio_bytes: bytes


def _spoken_text(reply: str) -> str:
    """Prepare a concise reply for speech without changing the saved text reply."""
    text = re.sub(r"\b(\d{4})(\d{4})\b", r"\1 \2", reply)
    return text.strip()


class GnaniVoiceIO:
    """Use Gnani STT and TTS; no alternate speech provider is configured."""

    def __init__(
        self,
        *,
        stt_language: str | None = None,
        tts_language: str | None = None,
        voice: str | None = None,
        model: str | None = None,
    ) -> None:
        load_dotenv()
        if not os.getenv("GNANI_API_KEY"):
            raise RuntimeError("Set GNANI_API_KEY in your local .env file before using voice features.")
        try:
            from gnani.stt import GnaniSTTClient
            from gnani.tts import GnaniTTSClient
        except ImportError as exc:
            raise RuntimeError(
                "Gnani voice support is not installed. Run `python -m pip install -e .`."
            ) from exc

        self._stt = GnaniSTTClient()
        self._tts = GnaniTTSClient()
        self.stt_language = stt_language or os.getenv("GNANI_STT_LANGUAGE", "en-IN")
        self.tts_language = tts_language or os.getenv("GNANI_TTS_LANGUAGE", "en-IN")
        self.voice = voice or os.getenv("GNANI_TTS_VOICE", "Kaveri")
        self.model = model or os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")

    def transcribe(self, audio_path: str | Path) -> str:
        path = Path(audio_path)
        if path.suffix.casefold() not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(f"Unsupported audio format: {path.suffix}")
        result = self._stt.transcribe(str(path), language_code=self.stt_language)
        return str(result.get("transcript", "")).strip()

    def synthesize(self, text: str) -> bytes:
        audio = self._tts.synthesize(
            text,
            voice=self.voice,
            model=self.model,
            language=self.tts_language,
        )
        return bytes(audio)

    def turn(
        self,
        audio_path: str | Path,
        agent: ConversationAgent,
        state: ConversationState,
    ) -> VoiceTurn:
        transcript = self.transcribe(audio_path)
        if not transcript:
            result = TurnResult(
                "Sorry, I didn’t catch that. Could you say it again?",
                "clarify",
            )
        else:
            result = agent.respond(state, transcript)
        speech = self._spoken_text(result.reply)
        return VoiceTurn(
            transcript=transcript,
            result=result,
            audio_bytes=self.synthesize(speech),
        )


def safe_transcript(transcript: str) -> str:
    """Mask any phone suffix or pincode before writing a batch metadata log."""
    return _redact_proofs(transcript)
