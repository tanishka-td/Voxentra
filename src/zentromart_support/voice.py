"""Gnani-only speech input/output wrapper for the conversation agent."""

from __future__ import annotations

import asyncio
import io
import os
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .conversation import ConversationAgent, ConversationState, TurnResult, _redact_proofs


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_MONTHS = {
    "january": "January", "february": "February", "march": "March",
    "april": "April", "may": "May", "june": "June", "july": "July",
    "august": "August", "september": "September", "october": "October",
    "november": "November", "december": "December",
}
_SMALL_NUMBERS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _under_thousand_words(number: int) -> str:
    parts: list[str] = []
    if number >= 100:
        parts.extend((_SMALL_NUMBERS[number // 100], "hundred"))
        number %= 100
    if number >= 20:
        tens, ones = divmod(number, 10)
        parts.append(_TENS[tens] + (f"-{_SMALL_NUMBERS[ones]}" if ones else ""))
    elif number:
        parts.append(_SMALL_NUMBERS[number])
    return " ".join(parts)


def _number_words(number: int) -> str:
    if number < 1000:
        return _under_thousand_words(number)
    for size, label in ((10_000_000, "crore"), (100_000, "lakh"), (1000, "thousand")):
        if number >= size:
            leading, remainder = divmod(number, size)
            tail = f" {_number_words(remainder)}" if remainder else ""
            return f"{_number_words(leading)} {label}{tail}"
    return _under_thousand_words(number)


@dataclass
class VoiceTurn:
    transcript: str
    result: TurnResult
    audio_bytes: bytes


def _spoken_text(reply: str) -> str:
    """Prepare a concise reply for speech without changing the saved text reply."""
    month_pattern = "|".join(_MONTHS)

    def date_with_month(match: re.Match[str]) -> str:
        day = int(match[1])
        suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        year = f" {match[3]}" if match[3] else ""
        return f"the {day}{suffix} of {_MONTHS[match[2].casefold()]}{year}"

    text = re.sub(
        rf"\b(\d{{1,2}})\s+({month_pattern})(?:\s+(\d{{4}}))?\b",
        date_with_month,
        reply,
        flags=re.IGNORECASE,
    )

    def spoken_amount(match: re.Match[str]) -> str:
        amount = int(match[1].replace(",", ""))
        return f"{_number_words(amount)} rupees"

    text = re.sub(r"(?:₹|INR\s*|Rs\.?\s*)([0-9][0-9,]*)", spoken_amount, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d{4})(\d{4})\b", r"\1 \2", text)
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
            from gnani.stt import GnaniSTTStreamClient
            from gnani.tts import AudioConfig, GnaniTTSRealtimeClient
        except ImportError as exc:
            raise RuntimeError(
                "Gnani voice support is not installed. Run `python -m pip install -e .`."
            ) from exc

        self._stt_client = GnaniSTTStreamClient
        self._tts_client = GnaniTTSRealtimeClient
        self._audio_config = AudioConfig
        self.stt_language = stt_language or os.getenv("GNANI_STT_LANGUAGE", "en-IN")
        self.tts_language = tts_language or os.getenv("GNANI_TTS_LANGUAGE", "en-IN")
        self.voice = voice or os.getenv("GNANI_TTS_VOICE", "Kaveri")
        self.model = model or os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")

    def transcribe(self, audio_path: str | Path) -> str:
        path = Path(audio_path)
        if path.suffix.casefold() not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(f"Unsupported audio format: {path.suffix}")
        pcm = _read_streaming_pcm(path)
        return asyncio.run(self._transcribe_pcm(pcm))

    async def _transcribe_pcm(self, pcm: bytes) -> str:
        async with self._stt_client(language_code=self.stt_language, sample_rate=16000) as stream:
            transcripts = await stream.stream_audio(io.BytesIO(pcm), realtime_pace=True)
        return " ".join(str(item.text).strip() for item in transcripts if item.text).strip()

    def synthesize(self, text: str) -> bytes:
        return asyncio.run(self._synthesize_stream(text))

    async def _synthesize_stream(self, text: str) -> bytes:
        async with self._tts_client() as client:
            options = {
                "voice": self.voice,
                "model": self.model,
                "audio_config": self._audio_config(
                    sample_rate=48000,
                    encoding="linear_pcm",
                    container="wav",
                ),
            }
            if self.model == "timbre-v2.5":
                options["language"] = self.tts_language
            audio = await client.synthesize_and_collect(text, **options)
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
        speech = _spoken_text(result.reply)
        return VoiceTurn(
            transcript=transcript,
            result=result,
            audio_bytes=self.synthesize(speech),
        )


def safe_transcript(transcript: str) -> str:
    """Mask any phone suffix or pincode before writing a batch metadata log."""
    return _redact_proofs(transcript)


def _read_streaming_pcm(path: Path) -> bytes:
    """Decode an input file to the raw PCM format required by Gnani STT WebSocket."""
    if path.suffix.casefold() == ".wav":
        try:
            with wave.open(str(path), "rb") as source:
                if (
                    source.getnchannels() == 1
                    and source.getsampwidth() == 2
                    and source.getframerate() == 16000
                    and source.getcomptype() == "NONE"
                ):
                    return source.readframes(source.getnframes())
        except (wave.Error, OSError):
            pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError(
            "Gnani streaming STT needs 16 kHz mono 16-bit PCM audio. "
            "Provide that as WAV, or install ffmpeg to convert other audio formats."
        )
    converted = subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", str(path), "-f", "s16le",
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if converted.returncode != 0 or not converted.stdout:
        detail = converted.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Could not convert audio for Gnani streaming STT: {detail or 'empty audio'}")
    return converted.stdout
