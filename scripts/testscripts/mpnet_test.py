#!/usr/bin/env python3
"""Run local embedding inference with sentence-transformers/paraphrase-mpnet-base-v2."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependencies. Install with: pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = str(PROJECT_ROOT / "models" / "sentence-transformers-paraphrase-mpnet-base-v2")
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
DEFAULT_TEXT_A = "Large language models can help users draft and refine text."
DEFAULT_TEXT_B = "Language models are useful for writing assistance and text improvement."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paraphrase-mpnet-base-v2 embedding inference.")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Local model directory (downloaded by scripts/mpnet_download.py)",
    )
    parser.add_argument(
        "--text-a",
        default=DEFAULT_TEXT_A,
        help="First input text",
    )
    parser.add_argument(
        "--text-b",
        default=DEFAULT_TEXT_B,
        help="Second input text",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=384,
        help="Tokenizer max length",
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


def ensure_model_files_complete(model_dir: Path) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(
            "Model directory not found: "
            f"{model_dir}. Run: python3 scripts/mpnet_download.py"
        )

    missing = [name for name in REQUIRED_FILES if not (model_dir / name).exists()]
    if missing:
        missing_text = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(
            "Model files are incomplete. Missing files:\n"
            f"{missing_text}\n"
            "Run: python3 scripts/mpnet_download.py --resume"
        )


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed_embeddings = torch.sum(last_hidden_state * expanded_mask, dim=1)
    summed_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
    return summed_embeddings / summed_mask


def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    max_length: int,
) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    embeddings = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    ensure_model_files_complete(model_dir)

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=False).to(device)
    model.eval()

    embeddings = encode_texts(
        texts=[args.text_a, args.text_b],
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
    )
    similarity = torch.matmul(embeddings[0], embeddings[1]).item()

    print(f"[load] model from: {model_dir}")
    print(f"[load] device: {device}")
    print(f"[embedding] dimension: {embeddings.shape[1]}")

    print("\n=== TEXT A ===")
    print(args.text_a)
    print("\n=== TEXT B ===")
    print(args.text_b)
    print("\n=== OUTPUT ===")
    print(f"cosine_similarity: {similarity:.6f}")


if __name__ == "__main__":
    main()
