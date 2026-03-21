# Trileaf

**Trileaf** starts from a simple premise: a writing optimizer should not be locked to a single model. Instead of forcing you onto one built-in rewriter, it lets you bring the model you already trust and use that model to improve its own draft through a standardized local optimisation pipeline.

It generates an ensemble of rewrite candidates, scores each one on AI-detection probability and semantic fidelity, and then uses Pareto-based selection to keep the strongest revision for every chunk. With optional double-pass optimisation, Trileaf can push the same text through the pipeline twice to make the final writing feel less uniform, more natural, and closer to human rhythm.

Its scoring layer is built on two public Hugging Face models: [`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01) for AI-generated-text probability estimation, and [`sentence-transformers/paraphrase-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) for chunk-level semantic similarity and sentence-alignment checks.

## Preview

![Trileaf dashboard — input and chunk view](screenshot/Trileafscreenshot1.png)

![Trileaf dashboard — output and scoring view](screenshot/Trileafscreenshot2.png)

---

## 1. Getting Started

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | 3.12 recommended |
| Git | For cloning |
| Internet connection | Required for first-time model download (~0.9 GB) and API calls |
| CUDA GPU (optional) | Detection models run on CPU, MPS, or CUDA — GPU not required |

### One-liner install (macOS / Linux / WSL)

```bash
curl -fsSL https://raw.githubusercontent.com/Rebas9512/Trileaf/main/install.sh | bash
```

The installer prompts for a clone target directory first. Press Enter to accept the default: `~/trileaf`.

One-liner layout:

- Source checkout + `.venv/` live in the install directory you choose.
- If the directory you choose already exists and is not empty, the installer falls back to a `trileaf/` subdirectory inside it.
- User config lives in `~/.trileaf/`.
- The public command is registered as `~/.local/bin/trileaf`.

**Options** (environment variables, set before the pipe):
```bash
TRILEAF_DIR=~/tools/trileaf  curl -fsSL … | bash   # custom install path
```

### Windows

```powershell
irm https://raw.githubusercontent.com/Rebas9512/Trileaf/main/install.ps1 | iex
```

```cmd
curl -fsSL https://raw.githubusercontent.com/Rebas9512/Trileaf/main/install.cmd -o install.cmd && install.cmd && del install.cmd
```

On Windows, the one-liner follows the same layout:

- Install directory prompt first (default: `%USERPROFILE%\trileaf`)
- If the selected directory already exists and is not empty, falls back to `trileaf\` subdirectory inside it
- Source checkout + `.venv\` inside that install directory
- JSON config files in `%USERPROFILE%\.trileaf`
- `trileaf.exe` exposed through the venv `Scripts\` directory on PATH

### Manual install (clone-and-run)

If you prefer to manage the clone location yourself:

**macOS / Linux / WSL**

```bash
git clone https://github.com/Rebas9512/Trileaf.git trileaf
cd trileaf
chmod +x setup.sh && ./setup.sh
```

**Windows**

```powershell
git clone https://github.com/Rebas9512/Trileaf.git trileaf
cd trileaf
powershell -ExecutionPolicy Bypass -File setup.ps1
```

The manual setup script creates an isolated `.venv/` inside the cloned directory. After it completes:

```bash
source .venv/bin/activate   # macOS / Linux / WSL — once per terminal session
.venv\Scripts\Activate.ps1  # Windows
trileaf run
```

---

## 2. Setup & First Use

### What happens during install

The installer (`install.sh` / `install.ps1`) and setup script (`setup.sh` / `setup.ps1`) run through six steps automatically:

| Step | What it does |
|------|--------------|
| 1. Platform | Detect OS, check requirements |
| 2. Python | Find Python 3.10+, validate version |
| 3. Virtual environment | Create `.venv`, install all Python dependencies |
| 4. CLI registration | Register `trileaf` command on your PATH |
| 5. **LeafHub integration** | Install LeafHub (if needed) and register this project |
| 6. Detection models | Prompt to download the two scoring models (~0.9 GB) |

Step 5 is fully automatic — you don't need to install LeafHub separately. See [LeafHub Integration](#3-leafhub-integration) for details.

### After install

```bash
trileaf run       # start the dashboard
```

Open **http://127.0.0.1:8001** in your browser.

If you skipped model download during setup, download them now:

```bash
trileaf setup     # download detection models (desklib + mpnet)
```

Verify everything is working:

```bash
trileaf doctor    # full environment and configuration health check
```

### CLI commands

| Command | What it does |
|---------|-------------|
| `trileaf run` | Start the dashboard server |
| `trileaf setup` | Download detection models (first-time or re-download) |
| `trileaf config` | Show LeafHub status, project binding, and credential info |
| `trileaf weight` | Show or update Pareto utility weights |
| `trileaf update` | Pull the latest version from git and refresh packages |
| `trileaf doctor` | Environment and model health check |
| `trileaf stop` | Stop a running server and release GPU memory |
| `trileaf remove` | Remove Trileaf, generated files, and installer PATH side effects |

Run `trileaf <command> --help` for per-command options.

### Setup script flags

| Flag | Effect |
|------|--------|
| `--reinstall` | Delete and recreate `.venv` from scratch |
| `--headless` | Non-interactive CI mode — no prompts |
| `--doctor` | Run environment check only, then exit |
| `--from-installer` | Internal flag set by `install.sh` (adjusts banner only) |

### Uninstall / clean removal

```bash
trileaf remove
```

Removes the install directory, `~/.trileaf/`, the `trileaf` symlink / PATH entry, downloaded models, and config.

For a manual source checkout, also removes generated files (`.venv`, downloaded models, build artefacts, caches):

```bash
trileaf remove --purge-source    # also delete the source checkout
```

---

## 3. LeafHub Integration

Trileaf uses [LeafHub](https://github.com/Rebas9512/Leafhub) for secure API key management. LeafHub is a local encrypted vault — your API keys never appear in plaintext files or shell history.

### How the integration works

During `setup.sh` (Step 5), Trileaf automatically:

1. Detects whether LeafHub is installed; installs it if not.
2. Registers this Trileaf project in LeafHub (`leafhub register trileaf --path <dir>`).
3. If no API providers are configured, opens the provider setup wizard.
4. Binds a provider to the project under the alias `"rewrite"`.
5. Copies `leafhub_probe.py` into the project root for runtime detection.

After setup, a `.leafhub` token file lives in the project root (chmod 600, git-ignored). On every startup, Trileaf reads this file to retrieve your API key from the encrypted vault — no key ever stored in any config file.

### What the vault manages for you

```
LeafHub vault  →  Trileaf runtime  →  external rewrite API
     ↑
     │  leafhub register trileaf
     │  (runs automatically during setup.sh)
```

The vault stores:
- Your API key (AES-256-GCM encrypted)
- Provider config: `base_url`, `model`, `api_format`, `auth_mode`

At startup, Trileaf resolves all of these from the vault automatically.

### Managing credentials

```bash
trileaf config         # show current LeafHub status and binding info
leafhub manage         # open the Web UI to add/edit providers at http://localhost:8765
```

To switch providers, add a new one in LeafHub and re-bind:

```bash
leafhub provider add --name "Anthropic" --key "sk-ant-..." --base-url https://api.anthropic.com
leafhub project bind trileaf --alias rewrite --provider "Anthropic" --model claude-3-5-haiku-20241022
```

### Credential resolution order

At runtime, Trileaf resolves credentials in this priority order:

```
1. LeafHub vault (.leafhub token)  →  API key + base_url + model + auth config
2. Environment variables           →  REWRITE_API_KEY, or provider-specific
                                       (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
```

LeafHub is always tried first. Env vars serve as a fallback for CI or advanced usage.

### Supported rewrite providers

Any OpenAI-compatible API endpoint works, including:

| Category | Examples |
|----------|---------|
| Cloud API | OpenAI, Anthropic, Google Gemini, Groq, Mistral, xAI |
| Self-hosted | Ollama, vLLM, LiteLLM |
| Regional | MiniMax, Moonshot/Kimi, OpenRouter, NVIDIA NIM |

---

## 4. Detection Models

Two local models score every rewrite candidate. They are required and always run locally (no external API):

| Model | Size | Role |
|-------|------|------|
| [`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01) | ~0.5 GB | AI-content probability scorer |
| [`sentence-transformers/paraphrase-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) | ~0.4 GB | Semantic similarity measurement |

Both models run on CPU, Apple Silicon MPS, or CUDA — no GPU is required for typical use.

Download (runs automatically during setup, or manually):

```bash
trileaf setup
```

The models are stored in `models/` inside the install directory and are git-ignored.

---

## 5. How the Pipeline Works

### Core idea

Most AI-detection tools exploit statistical patterns characteristic of LLM output: overly uniform sentence length, predictable phrasing, lack of idiomatic variation, and low perplexity. Trileaf attacks those patterns directly — but the key idea is that **the optimizer is the pipeline, not the rewrite model**.

If your preferred model can already write, it can also refine its own writing more effectively when wrapped in a disciplined system: diverse rewrite prompts, standardized scoring, hard semantic gates, and deterministic candidate selection. Rather than trusting one rewrite attempt, Trileaf turns each chunk into a controlled competition and picks the version that best trades off detectability reduction against meaning preservation.

### The ensemble strategy

Each chunk goes through three parallel rewrites at different temperatures and aggressiveness levels:

| Style | Temperature | What it changes |
|-------|-------------|-----------------|
| **Conservative** | 0.45 | Word and phrase substitution only. Sentence structure is frozen. |
| **Balanced** | 0.70 | Clause reordering, sentence merging/splitting, burstiness injection, anti-AI phrasing. |
| **Aggressive** | 0.92 | Deep restructuring — conversational register, free reordering, varied rhythm, rhetorical devices. |

All styles enforce hard factual constraints: facts, numbers, named entities, and core claims must remain unchanged.

### Selection via Pareto optimisation

Generating multiple candidates is only useful if selection is principled. The pipeline uses a two-stage selection process:

1. **Hard gate** — candidates that regress on any quality dimension are dropped:
   - AI score must be lower than the original chunk's score
   - Semantic similarity to the original must exceed a configurable threshold (default: 0.65)

2. **Pareto front + utility score** — among candidates that pass the gate, non-dominated sorting is applied across the two objectives (lower AI probability, higher semantic similarity). Among Pareto-optimal candidates, a weighted utility score picks the winner:

   ```
   U = W_AI × ai_gain_z + W_SEM × sem_z − W_RISK × risk_penalty
   ```

   Default weights: `W_AI = 0.60`, `W_SEM = 0.35`, `W_RISK = 0.05`. Adjustable via `trileaf weight`.

If no candidate passes the gate, the original chunk is kept unchanged — the optimizer never silently degrades quality.

### Short text vs long text mode

| Mode | Chunk size | Paragraph strategy | Best for |
|------|------------|-------------------|----------|
| **Short text** | ~200 chars | Each paragraph is its own chunk; large paragraphs split at sentence boundaries | Texts up to ~3 000 chars; fine-grained control; tends to produce the largest AI-score reduction |
| **Long text** | ~400 chars | Consecutive short paragraphs are merged until the target size is reached | Texts of ~2 000–8 000 chars; preserves rhetorical flow and style consistency |

### Two-pass optimization

| Mode | Description |
|------|-------------|
| **Single Run** | One standard optimization pass — default for most tasks |
| **Double Run** | The text passes through the full pipeline twice; the first-pass output becomes the input for the second pass |

In Double Run mode the original textarea text is never modified. Final AI-score deltas are reported relative to the original input from Pass 1.

---

## 6. Pipeline Architecture

### Topology overview

```
Input text
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Chunker                                            │
│  clean_text() → split_text() → chunks               │
│  (paragraph-aware; ~200 chars short / ~400 long)    │
└───────────────────┬─────────────────────────────────┘
                    │  [chunk₀, chunk₁, … chunkₙ]
                    │
                    ▼  (sequential, one chunk at a time)
┌─────────────────────────────────────────────────────┐
│  Per-chunk pipeline                                 │
│                                                     │
│  Step 0 ── Baseline scoring                         │
│            Desklib(chunk) → orig_ai_score           │
│                                                     │
│  Step 1 ── Ensemble rewrite (parallel × 3)          │
│            rewrite(chunk, "conservative") ──┐       │
│            rewrite(chunk, "balanced")     ──┼──→ candidates[]
│            rewrite(chunk, "aggressive")  ──┘       │
│                                                     │
│  Step 2 ── Batch scoring (per candidate)            │
│            • Desklib      → ai_score                │
│            • MPNet cosine → sem_score               │
│                                                     │
│  Step 3 ── Hard gate                                │
│            drop if ai_score ≥ orig_ai               │
│            drop if sem_score < SEM_GATE (0.65)      │
│                                  │                  │
│                    ┌─────────────┴──────────┐       │
│                 pass                      all fail  │
│                    │                         │      │
│  Step 4 ──  Pareto front             fallback: keep │
│             non-dominated sort         original     │
│             on (−ai_score, sem_score)               │
│                    │                                │
│             Utility score U                         │
│             = W_AI·ai_gain_z + W_SEM·sem_z          │
│               − W_RISK·risk_penalty                 │
│                    │                                │
│             select argmax(U)                        │
│             → best_rewrite                          │
└───────────────────┬─────────────────────────────────┘
                    │  [best_rewrite₀, …, best_rewriteₙ]
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  Reassembly                                         │
│  join chunks → output_text                          │
│  Desklib(output_text)   → final_ai_score            │
│  MPNet(input, output)   → final_sem_score           │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
          Dashboard (WebSocket)
          run_done event with scores + per-chunk trace
```

### Real-time feedback

The dashboard receives a WebSocket event stream as the pipeline runs. Each chunk emits:

| Event | Payload |
|-------|---------|
| `chunk_baseline` | Original AI score for this chunk |
| `ensemble_candidates` | The 3 rewrite texts before scoring |
| `chunk_stage` | Progress within the chunk (rewrite → batch_score → pareto) |
| `pareto_selection` | Which candidate won and why (utility scores, gate results) |
| `chunk_done` | Final selected text + scores |
| `run_done` | Aggregate scores across the full document |

### Key configuration parameters

Thresholds and weights are stored in `~/.trileaf/config.json`. Utility weights can be updated at any time:

```bash
trileaf weight                                        # show current weights
trileaf weight --ai 0.60 --sem 0.35 --risk 0.05      # update (must sum to 1.0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEM_GATE` | 0.65 | Minimum cosine similarity to original (hard gate) |
| `W_AI` | 0.60 | Weight for AI-score reduction in utility |
| `W_SEM` | 0.35 | Weight for semantic preservation in utility |
| `W_RISK` | 0.05 | Penalty weight for borderline candidates |
| `MAX_CHUNK_CHARS` | 200 | Target maximum characters per chunk |

---

## 7. Project Structure

```
Trileaf/
├── trileaf_cli.py                     # CLI entry point (trileaf command)
├── run.py                             # Server launcher (uvicorn bootstrap, credential resolution)
├── pyproject.toml                     # Package metadata — registers the trileaf command
├── install.sh                         # macOS/Linux/WSL one-liner installer (thin bootstrap)
├── install.ps1                        # Windows PowerShell one-liner installer (thin bootstrap)
├── install.cmd                        # Windows CMD bootstrap → install.ps1
├── setup.sh                           # Unix canonical setup: venv + deps + LeafHub + models
├── setup.ps1                          # Windows canonical setup: mirrors setup.sh
├── requirements.txt                   # Python runtime dependencies
├── requirements-dev.txt               # CI / test dependencies
│
├── api/
│   ├── optimizer_api.py               # FastAPI app + WebSocket broadcast
│   └── static/                        # Dashboard assets (HTML / JS / CSS)
│
├── scripts/
│   ├── check_env.py                   # Environment / health check (trileaf doctor)
│   ├── rewrite_config.py              # Credential resolution (LeafHub → env vars)
│   ├── app_config.py                  # Application config (~/.trileaf/config.json)
│   ├── orchestrator.py                # Pareto-selection pipeline
│   ├── chunker.py                     # Text cleaning + splitting
│   ├── models_runtime.py              # Model loading, caching, inference, API calls
│   ├── diag_pipeline.py               # Diagnostic / debug script
│   ├── _version.py                    # Single version source of truth
│   └── download_scripts/              # Per-model HuggingFace downloaders
│       ├── desklib_detector_download.py
│       └── mpnet_download.py
│
├── tests/                             # pytest test suite (157+ tests)
├── models/                            # Downloaded model weights (git-ignored)
└── leafhub/                           # leafhub_probe.py (copied here at setup time)
```

---

## 8. Acknowledgements

- [`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01): public AI-generated-text detection model used as Trileaf's local AI-probability scorer.
- [`sentence-transformers/paraphrase-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2): public sentence-embedding model used for semantic similarity scoring and sentence-alignment checks.
- [**LeafHub**](https://github.com/Rebas9512/Leafhub): local encrypted API-key vault that Trileaf uses for secure credential management. Required dependency — installed and configured automatically during setup.
