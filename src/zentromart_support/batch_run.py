"""Run the agent on every customer audio file in a folder."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .agent import GroundedTextAgent
from .capstone_tools import CapstoneToolGateway
from .conversation import ConversationAgent, ConversationState
from .retrieval import KnowledgeRetriever
from .voice import GnaniVoiceIO, SUPPORTED_AUDIO_SUFFIXES, safe_transcript


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_folder", type=Path, help="Folder containing audio files to process.")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "voice_batch")
    parser.add_argument("--dataset", type=Path, default=Path(os.getenv("DATASET_DIR", root / "Capstone_Project_Dataset_FTTT_2026")))
    parser.add_argument("--stt-language", help="Gnani STT locale; defaults to GNANI_STT_LANGUAGE.")
    parser.add_argument("--tts-language", help="Gnani TTS locale; defaults to GNANI_TTS_LANGUAGE.")
    args = parser.parse_args()

    audio_folder = args.audio_folder.resolve()
    if not audio_folder.is_dir():
        parser.error(f"Audio folder does not exist: {audio_folder}")
    audio_files = sorted(
        path for path in audio_folder.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_AUDIO_SUFFIXES
    )
    if not audio_files:
        parser.error(f"No supported audio files found in {audio_folder}")

    try:
        text_agent = GroundedTextAgent(KnowledgeRetriever(args.dataset))
        agent = ConversationAgent(text_agent, CapstoneToolGateway(args.dataset), args.dataset)
        voice = GnaniVoiceIO(stt_language=args.stt_language, tts_language=args.tts_language)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for audio_path in audio_files:
        conversation_id = f"audio-{audio_path.stem}-{uuid.uuid4().hex[:6]}"
        state = ConversationState(conversation_id=conversation_id)
        base = output_dir / audio_path.stem
        audio_output = output_dir / f"{audio_path.stem}.wav"
        text_output = output_dir / f"{audio_path.stem}.txt"
        record = {
            "conversation_id": conversation_id,
            "input_file": audio_path.name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            turn = voice.turn(audio_path, agent, state)
            _atomic_write(audio_output, turn.audio_bytes)
            _atomic_write(text_output, (turn.result.reply + "\n").encode("utf-8"))
            record.update({
                "transcript": safe_transcript(turn.transcript),
                **turn.result.to_dict(),
                "reply_audio": audio_output.name,
                "reply_text": text_output.name,
            })
            print(f"Processed {audio_path.name}")
        except Exception as exc:
            record["error"] = type(exc).__name__
            print(f"Failed {audio_path.name}: {type(exc).__name__}", file=sys.stderr)
        with base.with_suffix(".json").open("w", encoding="utf-8") as target:
            json.dump(record, target, ensure_ascii=False, indent=2)
            target.write("\n")

    print(f"Saved text, audio, and evidence records under {output_dir}")


if __name__ == "__main__":
    main()
