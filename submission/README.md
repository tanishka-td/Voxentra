# ZentroMart VoiceOps

A voice based support assistant for ZentroMart. It combines Gemini for grounded replies, local policy and catalog data, verified order tools, and Gnani for speech recognition and speech synthesis.

## Included

- `src/zentromart_support/` — application code.
- `requirements.txt` — Python dependencies.
- `.env.example` — configuration template with empty secret values.
- `report.pdf` — project report.
- `sample_outputs/` — sample voice conversation with transcripts, replies, evidence records, and generated audio.
- `demo.mp4` — add your recorded walkthrough here (maximum 2 minutes).

The competition dataset and API keys are intentionally not included. Keep your real `.env` outside this package. Set `DATASET_DIR` to the extracted dataset path on your machine.

## Setup

Use Python 3.10 or later. From this folder, install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then add your `GEMINI_API_KEY` and `GNANI_API_KEY`. Set `DATASET_DIR` to the local folder containing the extracted `Capstone_Project_Dataset_FTTT_2026` dataset.

## Run the voice agent

Place supported audio files (`.wav`, `.mp3`, `.flac`, `.ogg`, or `.m4a`) in an input folder. Then run:

```powershell
python run.py path\to\audio_folder --dataset path\to\Capstone_Project_Dataset_FTTT_2026 --output outputs\voice_batch
```

To keep ordered recordings in one conversation, use `--single-conversation`; name them in turn order, for example `01_question.wav`, `02_followup.wav`. A `--language-manifest path\to\languages.json` option can assign Gnani STT and TTS language settings per file. Without it, the command defaults or `.env` settings apply.

For a text-only question, use:

```powershell
python -m zentromart_support.chat "How long does a UPI refund take?" --evidence-json
```

## Output files

For each input recording, the voice runner writes a `.txt` reply, `.wav` spoken reply, and `.json` record containing the transcript, evidence IDs, and speech settings. A failed audio request exits with a nonzero status and records the error in that file's JSON record.

