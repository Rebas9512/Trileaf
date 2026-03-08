#!/usr/bin/env python3
"""Download Qwen/Qwen3-VL-8B-Instruct to the local models directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install with: pip install -r requirements.txt"
    ) from exc


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct"
INDEX_FILENAME = "model.safetensors.index.json"
METADATA_FILES = [
    "chat_template.json",
    "config.json",
    "generation_config.json",
    INDEX_FILENAME,
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen3-VL-8B-Instruct files.")
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


def read_required_shards(output_dir: Path) -> list[str]:
    index_path = output_dir / INDEX_FILENAME
    if not index_path.exists():
        return []

    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map", {})
    return sorted(set(weight_map.values()))


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


def ensure_required_files(
    model_id: str,
    output_dir: Path,
    token: str | None,
    resume_download: bool,
    force_download: bool,
) -> None:
    for filename in METADATA_FILES:
        download_if_needed(
            model_id=model_id,
            output_dir=output_dir,
            filename=filename,
            token=token,
            resume_download=resume_download,
            force_download=force_download,
        )

    for filename in read_required_shards(output_dir):
        download_if_needed(
            model_id=model_id,
            output_dir=output_dir,
            filename=filename,
            token=token,
            resume_download=resume_download,
            force_download=force_download,
        )


def validate_download(output_dir: Path) -> list[str]:
    missing = [name for name in METADATA_FILES if not (output_dir / name).exists()]
    missing.extend(
        name for name in read_required_shards(output_dir) if not (output_dir / name).exists()
    )
    return missing


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] model: {args.model_id}")
    print(f"[download] target: {output_dir}")

    ensure_required_files(
        model_id=args.model_id,
        output_dir=output_dir,
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
