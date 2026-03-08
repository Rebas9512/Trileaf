#!/usr/bin/env python3
"""Run local inference with Qwen/Qwen3-VL-8B-Instruct."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependencies. Install with: pip install -r requirements.txt"
    ) from exc
except ImportError as exc:
    raise SystemExit(
        "Your transformers version is too old for this script. "
        "Upgrade with: pip install -U 'transformers>=4.57.1'"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = str(PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct")
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
DEFAULT_PROMPT = "Describe the input in detail."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL-8B-Instruct inference.")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Local model directory (downloaded by scripts/qwen3_vl_download.py)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User prompt text",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Optional local image path for multimodal inference",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device selection",
    )
    return parser.parse_args()


def pick_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def read_required_shards(model_dir: Path) -> list[str]:
    index_path = model_dir / INDEX_FILENAME
    if not index_path.exists():
        return []

    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map", {})
    return sorted(set(weight_map.values()))


def ensure_model_files_complete(model_dir: Path) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(
            "Model directory not found: "
            f"{model_dir}. Run: python3 scripts/qwen3_vl_download.py --resume"
        )

    missing = [name for name in METADATA_FILES if not (model_dir / name).exists()]
    missing.extend(
        name for name in read_required_shards(model_dir) if not (model_dir / name).exists()
    )
    if missing:
        missing_text = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(
            "Model files are incomplete. Missing files:\n"
            f"{missing_text}\n"
            "Run: python3 scripts/qwen3_vl_download.py --resume"
        )


def ensure_transformers_supports_model(model_dir: Path) -> None:
    try:
        AutoConfig.from_pretrained(str(model_dir), trust_remote_code=False)
    except ValueError as exc:
        message = str(exc)
        if "qwen3_vl" in message:
            raise SystemExit(
                "Your installed transformers does not support model_type 'qwen3_vl'.\n"
                "Current fix:\n"
                "1. source /mnt/data/projects/5090_ubuntu/venv/bin/activate\n"
                "2. pip install -U 'transformers>=4.57.1' 'huggingface_hub>=0.23.0' accelerate safetensors\n"
                "3. rerun: python3 scripts/qwen3_vl_test.py\n\n"
                "If PyPI still installs an older incompatible build, use:\n"
                "pip install -U git+https://github.com/huggingface/transformers.git"
            ) from exc
        raise


def build_messages(prompt: str, image_path: str | None) -> list[dict]:
    content: list[dict] = []
    if image_path is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def load_image(image_path: str | None) -> Image.Image | None:
    if image_path is None:
        return None
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    ensure_model_files_complete(model_dir)
    ensure_transformers_supports_model(model_dir)

    device = pick_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[load] model from: {model_dir}")
    print(f"[load] device: {device}, dtype: {dtype}")

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir),
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=False)

    image = load_image(args.image_path)
    messages = build_messages(args.prompt, args.image_path)
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    processor_kwargs = {"text": [prompt_text], "return_tensors": "pt"}
    if image is not None:
        processor_kwargs["images"] = [image]

    inputs = processor(**processor_kwargs)
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    print("\n=== PROMPT ===")
    print(args.prompt)
    if args.image_path:
        print("\n=== IMAGE ===")
        print(str(Path(args.image_path).resolve()))
    print("\n=== OUTPUT ===")
    print(response)


if __name__ == "__main__":
    main()
