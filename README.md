# ZentroMart VoiceOps support agent

This project will build the VoiceOps 2026 support agent in stages. The local
capstone dataset is kept in `Capstone_Project_Dataset_FTTT_2026/` for development
and is excluded from Git and submission packages.

## Current stages: text baseline, retrieval, conversations, and voice wrapper

The Step 1 baseline asks Gemini to answer a customer question without company
documents or order tools. It saves each practice response and its metadata as
JSONL so we can compare it with the grounded agent. The Step 2 agent retrieves
section-level policy evidence and catalog rows using BM25 and records the exact
retrieved chunks with each reply. A multi-turn controller wraps the same agent
with the supplied identity-checked order and ticket tools.
The Gnani wrapper uses the supplied speech API for transcription and synthesis.

### Setup

1. Install Python 3.10 or newer.
2. Install the project: `python -m pip install -e .`
3. Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`. `GEMINI_MODEL` is
   optional and defaults to `gemini-3.8-flash`.

Never commit your API key. `.env` files are ignored; `.env.example` contains only
variable names and safe defaults.

### Run one question

`python -m zentromart_support.baseline`

### Capture the Step 1 practice responses

`python -m zentromart_support.baseline --evaluate`

Responses are saved to `outputs/step1_baseline.jsonl`. The dataset must be present
at the project root, or set `DATASET_DIR` to its extracted folder.

### Evaluate retrieval without API calls

`python -m zentromart_support.retrieval_eval --top-k 5`

This compares retrieved evidence IDs with the practice set's gold evidence and
saves per-question results to `outputs/retrieval_eval.jsonl`.

### Try the grounded agent

`python -m zentromart_support.chat "How long does a UPI refund take?" --evidence-json`

### Try a multi-turn support conversation

`python -m zentromart_support.conversation_cli --output outputs/local_turns.jsonl`

The conversation controller asks for order verification before it shares order
facts. Its optional turn log redacts verification numbers and omits them from tool
call records.

### Process a folder of voice messages

`python -m zentromart_support.batch_run path/to/audio_folder --output outputs/voice_batch`

The runner accepts WAV, MP3, FLAC, OGG, and M4A inputs. Each file is treated as
one conversation; use the interactive conversation runner to try follow-up turns.
Configure the speech languages, voice, and model in `.env` or pass the language
flags to the batch command. The only configured speech provider is Gnani.

## Dataset validation

The supplied dataset validator can be run from its folder:

`python validate_dataset.py`

The dataset itself must not be included in the final submission ZIP.
