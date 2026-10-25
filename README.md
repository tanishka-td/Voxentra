# ZentroMart VoiceOps support agent

A staged implementation of the VoiceOps 2026 customer-support agent. It combines
Gemini-based grounded responses, local policy and catalog data, identity-checked
order tools, and Gnani speech recognition and synthesis.

The capstone dataset is required locally for development. It is not included in
this repository or in a submission package.

## Setup

1. Install Python 3.10 or newer.
2. Install the project dependencies:

   ```powershell
   python -m pip install -e .
   ```

   If running directly from the source folder in PowerShell, set the import path
   for the current session with `$env:PYTHONPATH = "$PWD\src"`.

3. Create a local environment file and add your `GEMINI_API_KEY`:

   ```powershell
   Copy-Item .env.example .env
   ```

   Add `GNANI_API_KEY` to `.env` to use voice features. Keep both keys private.
4. Extract `Capstone_Project_Dataset_FTTT_2026` into the project folder, or set
   `DATASET_DIR` to its location.

The text model defaults to `gemini-3.5-flash-lite`; embeddings default to
`gemini-embedding-001`. Voice language defaults to `en-IN`. Speech language,
voice, and model can be configured in `.env`.

## Run the text agent

Start an interactive prompt-only baseline:

```powershell
python -m zentromart_support.baseline
```

Run it against the 52 practice questions:

```powershell
python -m zentromart_support.baseline --evaluate
```

Try the grounded agent with one question:

```powershell
python -m zentromart_support.chat "How long does a UPI refund take?" --evidence-json
```

The grounded agent retrieves policy evidence and uses deterministic lookups for
supported product, refund, late Express credit, return-window, and pincode
questions. It records the evidence IDs used for each response.

## Evaluate retrieval and answers

Evaluate hybrid retrieval against the evidence labels:

```powershell
python -m zentromart_support.retrieval_eval --top-k 5
```

Use `--lexical-only` to skip embedding API calls and evaluate the BM25 fallback.
Retrieval rankings and cached vectors are written under `outputs/` and `.cache/`.

Generate grounded answers for the practice questions:

```powershell
python -m zentromart_support.question_eval
```

Use `--asr-text` to evaluate the supplied speech-like transcripts. This is text
evaluation; the dataset does not contain audio recordings. Score a JSONL result:

```powershell
python -m zentromart_support.score_practice outputs/step2_rag.jsonl --output outputs/step2_rag_score.json
```

The local scorer is approximate, not the organizers' official scorer. Practice
answer generation uses Gemini requests and may be interrupted by API quotas; the
runner can resume from valid responses already in the output file. Use
`--overwrite` to regenerate a result after changing the agent.

## Run multi-turn conversations

Start an interactive text conversation:

```powershell
python -m zentromart_support.conversation_cli --output outputs/local_turns.jsonl
```

The conversation controller verifies a caller before sharing order facts, allows
general policy questions while verification is pending, and limits verification
to two attempts. Turn logs redact verification numbers.

Evaluate the supplied conversation scenarios:

```powershell
python -m zentromart_support.conversation_eval --lexical-only
```

This writes per-turn decisions, tool calls, evidence IDs, and approximate checks to
`outputs/step3_conversations.jsonl`. It can create synthetic handover tickets in
the dataset's tools folder. Use `--limit 1` for a small run, `--overwrite` to
replace an existing report, or repeat `--conversation-id` to select specific
scenarios.

## Process voice messages

Put WAV, MP3, FLAC, OGG, or M4A files in a folder and run:

```powershell
python -m zentromart_support.batch_run path/to/audio_folder --output outputs/voice_batch
```

Each input file starts a separate, single-turn conversation. The interactive
conversation interface accepts text follow-up turns. The batch runner saves a
transcript, reply text, evidence record, and reply audio for each file. Gnani is
the only configured speech provider. A 16 kHz, mono, 16-bit PCM WAV works without
conversion; other input audio requires `ffmpeg` to convert it for Gnani STT.

## Validate the dataset

From the project folder, run:

```powershell
python Capstone_Project_Dataset_FTTT_2026/validate_dataset.py
```

## Local-only files

`.env`, the dataset, generated `outputs/`, embedding cache, local voice samples,
and Python build metadata are ignored by Git. Keep API keys and personal voice
recordings out of the submission package.
