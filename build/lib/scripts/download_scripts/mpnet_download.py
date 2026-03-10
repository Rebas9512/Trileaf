#!/usr/bin/env python3
"""Download sentence-transformers/paraphrase-mpnet-base-v2 to the local models directory."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install with: pip install -r requirements.txt"
    ) from exc


DEFAULT_MODEL_ID = "sentence-transformers/paraphrase-mpnet-base-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "sentence-transformers-paraphrase-mpnet-base-v2"
REQUIRED_FILES = [
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download paraphrase-mpnet-base-v2 files.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Local directory for downloaded model files",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (optional, needed for private/gated repos)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume partially downloaded files",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download all files even if they already exist locally",
    )
    return parser.parse_args()


def download_if_needed(
    model_id: str,
    output_dir: Path,
    filename: str,
    token: str | None,
    resume_download: bool,
    force_download: bool,
) -> None:
    local_path = output_dir / filename
    if local_path.exists() and not force_download:
        print(f"[skip] {filename}")
        return

    action = "redownload" if local_path.exists() and force_download else "download"
    print(f"[{action}] {filename}")
    hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        token=token,
        resume_download=resume_download,
        force_download=force_download,
    )


def validate_download(output_dir: Path) -> list[str]:
    return [filename for filename in REQUIRED_FILES if not (output_dir / filename).exists()]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] model: {args.model_id}")
    print(f"[download] target: {output_dir}")

    for filename in REQUIRED_FILES:
        download_if_needed(
            model_id=args.model_id,
            output_dir=output_dir,
            filename=filename,
            token=args.token,
            resume_download=args.resume,
            force_download=args.force_download,
        )

    missing = validate_download(output_dir)
    if missing:
        missing_text = "\n".join(f"- {name}" for name in missing)
        raise SystemExit(
            "Download finished but required files are still missing:\n"
            f"{missing_text}"
        )

    print(f"[done] model downloaded to: {output_dir}")


if __name__ == "__main__":
    main()
