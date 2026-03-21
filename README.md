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
| CUDA GPU (optional) | Required only for the optional local rewrite model; detection models run on CPU, Apple Silicon MPS, or CUDA |

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
- CLI registration tries to append `~/.local/bin` to `~/.bash_profile`. If that write fails, installation still completes and the script prints the exact `export PATH=...` command to run manually.

**Options** (environment variables, set before the pipe):
```bash
TRILEAF_DIR=~/tools/trileaf  curl -fsSL … | bash   # custom install path
TRILEAF_NO_ONBOARD=1         curl -fsSL … | bash   # skip the wizard (CI / headless)
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
- If the selected directory already exists and is not empty, the installer falls back to a `trileaf\` subdirectory inside it
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

The manual setup script creates an isolated `.venv/` inside the cloned directory. After it completes, activate the venv before using the `trileaf` command:

```bash
source .venv/bin/activate   # macOS / Linux / WSL — once per terminal session
.venv\Scripts\Activate.ps1  # Windows
trileaf run
```

### Onboarding

The first-time wizard (`trileaf onboard`) walks through four steps.

#### Step 1 — Python environment check

Verifies torch, sentence-transformers, and huggingface_hub are installed. Automatically satisfied after setup completes.

#### Step 2 — Detection models (required, ~0.9 GB total)

These two models score every rewrite candidate and are the **minimum local requirement** for running Trileaf. They are always required, regardless of which rewrite backend you choose. Both are public Hugging Face repos and download without a HuggingFace account.

| Model | Size | Role |
|-------|------|------|
| [`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01) | ~0.5 GB | AI-content probability scorer |
| [`sentence-transformers/paraphrase-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) | ~0.4 GB | Semantic similarity measurement |

These models are downloaded during onboarding and stored locally. The two scoring models run comfortably on CPU — CUDA is not required. On Apple Silicon they can also run on MPS.

#### Step 3 — Rewrite provider

The rewrite provider generates candidate rewrites for each text chunk. Three options are available:

##### Option A — LeafHub (recommended)

[LeafHub](https://github.com/Rebas9512/Leafhub) is a local encrypted API-key vault. It stores your provider credentials in an AES-256-GCM encrypted file (`~/.leafhub/providers.enc`) and serves them to Trileaf at runtime — without exposing them in any dotfile or shell history.

```
LeafHub vault  →  trileaf runtime  →  external API
```

During `trileaf onboard` or `trileaf config`, selecting **LeafHub** will:

1. Create a LeafHub project named `trileaf` and link it to this directory (a `.leafhub` token file is written)
2. Let you bind a provider (e.g. MiniMax, OpenAI) to an alias (e.g. `minimax`)
3. Automatically read `base_url`, `model`, `api_format`, and `auth_mode` from the bound provider — no further prompts

After setup, the `.env` file contains only the alias reference:

```
LEAFHUB_ALIAS=minimax
REWRITE_BACKEND=external
REWRITE_BASE_URL=https://api.minimax.io/anthropic
REWRITE_MODEL=MiniMax-M2.5
```

The API key itself lives only in the LeafHub vault — never on disk in plain text.

**Install LeafHub** (macOS / Linux / WSL):
```bash
curl -fsSL https://raw.githubusercontent.com/Rebas9512/Leafhub/main/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/Rebas9512/Leafhub/main/install.ps1 | iex
```

After the installer completes, open a new terminal so the updated PATH takes effect, then run `trileaf onboard` again.

**Automatic fallback if LeafHub setup fails**

If LeafHub is installed but cannot complete the project-link step (e.g. the manage server is not running, a network error, or a corrupted dotfile), the onboarding wizard automatically falls back to the `.env` flow. A clear error message is shown explaining what failed, and the interactive `.env` wizard opens immediately — no manual intervention needed. You can re-link LeafHub later with `trileaf config`.

##### Option B — External API key in `.env` (simple fallback)

If you prefer not to use LeafHub, the wizard stores the API key directly in `PROJECT_ROOT/.env` (chmod 600, git-ignored). This works for quick local use or CI scenarios where LeafHub is not available.

> **Security note:** The `.env` approach stores the API key in plain text on disk. Anyone with read access to the project directory can read it. Use LeafHub for better key hygiene, especially on shared machines.

The `.env` file written by the wizard looks like:

```
REWRITE_BACKEND=external
REWRITE_BASE_URL=https://api.openai.com/v1
REWRITE_MODEL=gpt-4o
REWRITE_API_KEY=sk-...
```

Supported providers include OpenAI, Anthropic, Groq, Mistral, OpenRouter, xAI, Ollama, vLLM, and any OpenAI-compatible gateway.

##### Option C — Local Qwen3-VL-8B (fully offline)

Downloads and runs `Qwen/Qwen3-VL-8B-Instruct` locally. No API key or internet connection needed at inference time.

| Config | VRAM required |
|--------|--------------|
| Scoring models only (external rewrite API) | ~2 GB or CPU |
| Scoring + local Qwen3-VL-8B (bf16) | ~18 GB minimum, **24 GB recommended** |

> If your GPU has less than 16 GB VRAM, use Option A or B for the rewrite step.

Download:
```bash
python -m scripts.download_scripts.qwen3_vl_download   # ~16 GB
```

#### Step 4 — Final validation

`check_env.py` verifies the two required detection models and the active rewrite configuration. Re-run at any time:
```bash
trileaf doctor
```

### Start the dashboard

```bash
trileaf run
```

Open **http://127.0.0.1:8001** in your browser.

All Trileaf operations are available as subcommands:

| Command | What it does |
|---------|-------------|
| `trileaf run` | Start the dashboard server |
| `trileaf onboard` | First-time setup wizard (models + provider) |
| `trileaf config` | Reconfigure the rewrite provider |
| `trileaf weight` | Show or update Pareto utility weights |
| `trileaf update` | Pull the latest version from git and refresh packages |
| `trileaf doctor` | Environment and model health check |
| `trileaf stop` | Stop a running server and release GPU memory |
| `trileaf remove` | Remove Trileaf, generated files, and installer PATH side effects |

Run `trileaf <command> --help` for per-command options.

### Manual setup script flags

| Flag | Effect |
|------|--------|
| `--reinstall` | Delete and recreate `.venv` from scratch |
| `--skip-onboarding` | Skip model download / provider wizard |
| `--headless` | Non-interactive CI mode (implies `--skip-onboarding`) |
| `--doctor` | Run environment check only, then exit |

### Uninstall / clean removal

```bash
trileaf remove
```

One-liner installs are removed completely: the chosen install directory, `~/.trileaf/`, the `trileaf` symlink / PATH entry, generated models, and config are cleaned up.

For a manual source checkout, `trileaf remove` deletes generated files (`.venv`, downloaded models, build artefacts, caches, user config). If you also want to delete the checkout itself:

```bash
trileaf remove --purge-source
```

---

## 2. Configuring the Rewrite Provider

### First-time setup

```bash
trileaf onboard
```

The wizard guides you through all four steps end-to-end.

### Reconfiguring the provider

```bash
trileaf config
```

This re-runs only the provider wizard (Step 3). It is the recommended way to switch between providers, update model names, or re-link a LeafHub project.

### LeafHub flow (linked project)

When a LeafHub project is already linked, `trileaf config` detects the bound aliases automatically:

```
  Bound aliases in this LeafHub project:
    1. minimax
    2. Enter a different alias
  Select alias: 1

  Base URL:  https://api.minimax.io/anthropic
  Model:     MiniMax-M2.5

  [OK] Wrote .env
       LeafHub alias: minimax
       Base URL:     https://api.minimax.io/anthropic
       Model:        MiniMax-M2.5
       API key fetched from LeafHub vault at runtime.
```

If the selected alias has a complete provider profile in LeafHub (`base_url` + `model`), no further prompts are shown. If the profile is partial, only the missing fields are asked.

### Credential resolution order

At runtime, Trileaf resolves credentials in this priority order:

```
1. LeafHub vault      → REWRITE_API_KEY (+ base_url / model / auth_mode from provider config)
2. PROJECT_ROOT/.env  → all REWRITE_* keys loaded as base layer
3. os.environ         → provider-specific fallbacks (e.g. OPENAI_API_KEY)
```

LeafHub's API key always wins. For other fields (model name, auth mode), the `.env` value takes priority over the LeafHub-provided default — this lets you override individual settings locally without changing the vault.

### Health check

```bash
trileaf doctor
```

Prints device info, model paths, and rewrite backend status. Credential source is shown as `leafhub`, `dotenv`, or `env` depending on what resolved.

---

## 3. Project Features

### Core idea

Most AI-detection tools exploit statistical patterns that are characteristic of LLM output: overly uniform sentence length, predictable phrasing, lack of idiomatic variation, and low perplexity relative to a reference distribution. Trileaf attacks those patterns directly, but the key idea is broader: the optimizer is the pipeline, not the rewrite model.

If your preferred model can already write, it can also refine its own writing more effectively when it is wrapped in a disciplined system: diverse rewrite prompts, standardized scoring, hard semantic gates, and deterministic candidate selection. Rather than trusting one rewrite attempt, Trileaf turns each chunk into a controlled competition and picks the version that best trades off detectability reduction against meaning preservation.

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

   Default weights: `W_AI = 0.60`, `W_SEM = 0.35`, `W_RISK = 0.05`. Adjustable via `trileaf weight` or directly in `~/.trileaf/config.json`.

If no candidate passes the gate, the original chunk is kept unchanged — the optimizer never silently degrades quality.

### Short text vs long text mode

The dashboard toggle selects between two chunking strategies:

| Mode | Chunk size | Paragraph strategy | Best for |
|------|------------|-------------------|----------|
| **Short text** | ~200 chars | Each paragraph is its own chunk; large paragraphs split at sentence boundaries | Texts up to ~3 000 chars; fine-grained control; tends to produce the largest AI-score reduction |
| **Long text** | ~400 chars | Consecutive short paragraphs are merged until the target size is reached; large paragraphs are still sentence-split | Texts of ~2 000–8 000 chars; preserves rhetorical flow and style consistency across paragraphs |

Both modes pass through the same Pareto-selection scoring pipeline. The 50 000-character API limit applies in both modes.

### Two-pass optimization

The **Run Mode** toggle on the dashboard selects between two execution strategies:

| Mode | Description |
|------|-------------|
| **Single Run** | One standard optimization pass — default for most tasks |
| **Double Run** | The text passes through the full pipeline twice; the first-pass output becomes the input for the second pass |

In Double Run mode the original textarea text is never modified. The second pass uses an internal buffer so the source copy is always preserved. Final AI-score deltas are reported relative to the **original** input from Pass 1.

### Bring your own model

The rewrite backend is fully pluggable. Any OpenAI-compatible API endpoint works, including:

- Cloud providers (OpenAI, Anthropic, Google Gemini, Groq, Mistral, xAI)
- Self-hosted servers (Ollama, vLLM, LiteLLM)
- Regional providers (MiniMax, Moonshot/Kimi, OpenRouter)

Configure or reconfigure at any time with `trileaf config`.

---

## 4. Pipeline Architecture

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

Thresholds and weights are stored in `~/.trileaf/config.json`. Utility weights can be updated at any time via the CLI:

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

## 5. Project Structure

```
├── trileaf_cli.py                     # CLI entry point
├── run.py                             # Server launcher (called by trileaf_cli)
├── pyproject.toml                     # Package metadata — registers the trileaf command
├── install.sh / install.ps1 / install.cmd  # One-liner installers
├── setup.sh / setup.ps1               # Manual clone-and-run setup scripts
├── requirements.txt                   # Python runtime dependencies
├── requirements-dev.txt               # CI / test dependencies
├── api/
│   ├── optimizer_api.py               # FastAPI app + WebSocket broadcast
│   └── static/                        # Dashboard assets (HTML / JS / CSS)
├── scripts/
│   ├── onboarding.py                  # First-time setup wizard (trileaf onboard)
│   ├── check_env.py                   # Environment / health check (trileaf doctor)
│   ├── rewrite_config.py              # Credential resolution (LeafHub → .env → env vars)
│   ├── rewrite_provider_cli.py        # Interactive provider configuration wizard
│   ├── app_config.py                  # Application config (~/.trileaf/config.json)
│   ├── orchestrator.py                # Pareto-selection pipeline
│   ├── chunker.py                     # Text cleaning + splitting
│   ├── models_runtime.py              # Model loading, caching, inference, API calls
│   ├── diag_pipeline.py               # Diagnostic / debug utilities
│   ├── _version.py                    # Single version source of truth
│   └── download_scripts/              # Per-model HuggingFace downloaders
│       ├── desklib_detector_download.py
│       ├── mpnet_download.py
│       └── qwen3_vl_download.py
├── tests/                             # pytest test suite (229+ tests)
└── models/                            # Downloaded model weights (git-ignored)
```

---

## 6. Acknowledgements

- [`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01): public AI-generated-text detection model used as Trileaf's local AI-probability scorer.
- [`sentence-transformers/paraphrase-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2): public sentence-embedding model used for semantic similarity scoring and sentence-alignment checks.
- [**LeafHub**](https://github.com/Rebas9512/Leafhub): local encrypted API-key vault that Trileaf integrates with for secure credential management.
