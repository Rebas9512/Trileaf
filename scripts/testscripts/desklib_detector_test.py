#!/usr/bin/env python3
"""Run local inference with desklib/ai-text-detector-v1.01."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependencies. Install with: pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = str(PROJECT_ROOT / "models" / "desklib-ai-text-detector-v1.01")
REQUIRED_FILES = [
    "added_tokens.json",
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "spm.model",
    "tokenizer.json",
    "tokenizer_config.json",
]
DEFAULT_TEXT = (
    "This essay presents a balanced overview of the topic, explains both sides clearly, "
    "and concludes with a practical recommendation based on the evidence."
)


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed_embeddings = torch.sum(last_hidden_state * expanded_mask, dim=1)
        summed_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        pooled_output = summed_embeddings / summed_mask

        logits = self.classifier(pooled_output)
        result = {"logits": logits}

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            result["loss"] = loss_fct(logits.view(-1), labels.float())

        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Desklib AI detector inference.")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Local model directory (downloaded by scripts/desklib_detector_download.py)",
    )
    parser.add_argument(
        "--input-text",
        default=DEFAULT_TEXT,
        help="Input text to classify",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for AI-generated classification",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
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
            f"{model_dir}. Run: python3 scripts/desklib_detector_download.py"
        )

    missing = [name for name in REQUIRED_FILES if not (model_dir / name).exists()]
    if missing:
        missing_text = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(
            "Model files are incomplete. Missing files:\n"
            f"{missing_text}\n"
            "Run: python3 scripts/desklib_detector_download.py --resume"
        )


def predict_single_text(
    text: str,
    model: DesklibAIDetectionModel,
    tokenizer,
    device: str,
    max_length: int,
    threshold: float,
) -> tuple[float, int]:
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()

    label = 1 if probability >= threshold else 0
    return probability, label


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    ensure_model_files_complete(model_dir)

    device = pick_device(args.device)
    config = AutoConfig.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    model = DesklibAIDetectionModel.from_pretrained(
        str(model_dir),
        config=config,
        trust_remote_code=False,
    ).to(device)

    max_length = min(args.max_length, getattr(config, "max_position_embeddings", args.max_length))

    print(f"[load] model from: {model_dir}")
    print(f"[load] device: {device}")
    print(f"[inference] max_length: {max_length}")

    probability, predicted_label = predict_single_text(
        text=args.input_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        threshold=args.threshold,
    )

    print("\n=== INPUT ===")
    print(args.input_text)
    print("\n=== OUTPUT ===")
    print(f"probability_ai_generated: {probability:.6f}")
    print(
        "predicted_label: "
        + ("AI Generated" if predicted_label == 1 else "Not AI Generated")
    )


if __name__ == "__main__":
    main()
