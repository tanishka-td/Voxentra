# ZentroMart VoiceOps support agent

This project will build the VoiceOps 2026 support agent in stages. The local
capstone dataset is kept in `Capstone_Project_Dataset_FTTT_2026/` for development
and is excluded from Git and submission packages.

## Current stages: text baseline, retrieval, conversations, and voice wrapper

The Step 1 baseline asks Gemini to answer a customer question without company
documents or order tools. It saves each practice response and its rubric metadata
as JSONL so we can compare it with the grounded agent. The grounded agent uses
Gemini embeddings and BM25 together, keeps section IDs and document metadata, and
records the exact evidence chunks with each reply. A multi-turn controller wraps
the same agent with the supplied identity-checked order and ticket tools.
The Gnani wrapper uses WebSocket streaming for speech recognition and real-time
streaming for speech synthesis. It sends the transcription through the same
conversation controller and saves the reply as WAV audio.

### Setup

1. Install Python 3.10 or newer.
2. Install the project: `python -m pip install -e .`
3. Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`. The text model
   defaults to `gemini-3.5-flash-lite`; embeddings use `gemini-embedding-001`.

Never commit your API key. `.env` files are ignored; `.env.example` contains only
variable names and safe defaults.

### Run one question

`python -m zentromart_support.baseline`

### Capture the Step 1 practice responses

`python -m zentromart_support.baseline --evaluate`

Responses are saved to `outputs/step1_baseline.jsonl`. The dataset must be present
at the project root, or set `DATASET_DIR` to its extracted folder.

### Evaluate hybrid retrieval

`python -m zentromart_support.retrieval_eval --top-k 5`

This embeds the knowledge base and practice questions, combines semantic ranking
with BM25, and compares evidence IDs with the practice set's gold labels. Document
vectors are cached under `.cache/`; per-question rankings are saved to
`outputs/retrieval_eval.jsonl`. Use `--lexical-only` to evaluate the offline BM25
fallback without making embedding API calls.

### Evaluate grounded answers

`python -m zentromart_support.question_eval`

This sends all 52 practice questions to the grounded agent and saves replies,
retrieved evidence, and rubric fields to `outputs/step2_rag.jsonl`. Score them with
`python -m zentromart_support.score_practice outputs/step2_rag.jsonl --output outputs/step2_rag_score.json`.
The runner resumes valid responses already in the output file; pass `--overwrite`
to regenerate them after changing the agent. Product warranty and return-window
facts, refund timelines and dates, and pincode availability use deterministic
lookups/rules where the question contains enough information. Refund business-day
calculations exclude Sundays and the dataset holidays; the dataset reference date
is 20 October 2026.

### Try the grounded agent

`python -m zentromart_support.chat "How long does a UPI refund take?" --evidence-json`

### Try a multi-turn support conversation

`python -m zentromart_support.conversation_cli --output outputs/local_turns.jsonl`

The conversation controller asks for order verification before it shares order
facts. Its optional turn log redacts verification numbers and omits them from tool
call records. While verification is pending, general policy questions can still be
answered; order-specific follow-ups stay behind the verification gate. After two
failed attempts, the order remains locked for the rest of that conversation.

### Evaluate multi-turn conversations

`python -m zentromart_support.conversation_eval --lexical-only`

This runs the supplied 23 conversation scenarios and saves per-turn decisions,
tool calls, evidence IDs, and approximate local checks to
`outputs/step3_conversations.jsonl`. It may create synthetic handover tickets in
the dataset tool folder. Use `--limit 1` for a small first run, or `--overwrite`
to replace an existing report. To rerun only selected scenarios without
recreating tickets from the rest of the set, repeat `--conversation-id`, for
example `--conversation-id PC003 --conversation-id PC004`.

### Process a folder of voice messages

`python -m zentromart_support.batch_run path/to/audio_folder --output outputs/voice_batch`

The runner accepts WAV, MP3, FLAC, OGG, and M4A inputs. Each file is treated as
one conversation; use the interactive conversation runner to try follow-up turns.
Configure the speech languages, voice, and model in `.env` or pass the language
flags to the batch command. Set `GNANI_API_KEY` in your ignored local `.env`; do
not commit or share it. A 16 kHz, mono, 16-bit PCM WAV works without conversion.
Other supported input formats need `ffmpeg` installed so they can be converted to
the PCM frames required by Gnani's STT WebSocket. The only configured speech
provider is Gnani.

## Dataset validation

The supplied dataset validator can be run from its folder:

`python validate_dataset.py`

The dataset itself must not be included in the final submission ZIP.
