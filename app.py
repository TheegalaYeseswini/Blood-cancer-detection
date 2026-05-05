from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src.load_models import ModelLoadError, get_default_model_configs, load_all_models
from src.predict import predict_routed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with leukemia, lymphoma, and tetra blood-cell classifiers."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image to classify.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Inference device. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full prediction output as JSON.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no GPU is available for PyTorch.")
    return requested_device


def format_prediction_output(results: dict) -> str:
    lines = []

    tetra_prediction = results["tetraclassifier"]
    lines.append(f"{tetra_prediction['model_name']}:")
    lines.append(
        f"  Predicted label: {tetra_prediction['predicted_label']} "
        f"(confidence: {tetra_prediction['confidence']:.4f})"
    )

    subtype_prediction = results["selected_subtype_model"]
    if subtype_prediction is not None:
        lines.append(f"{subtype_prediction['model_name']}:")
        lines.append(
            f"  Predicted label: {subtype_prediction['predicted_label']} "
            f"(confidence: {subtype_prediction['confidence']:.4f})"
        )
    else:
        lines.append("Subtype model:")
        lines.append("  Not run for this tetra prediction.")

    combined = results["combined"]
    lines.append("Combined summary:")
    lines.append(f"  Primary label: {combined['primary_label']}")
    lines.append(f"  Secondary label: {combined['secondary_label']}")
    lines.append(f"  Note: {combined['summary']}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Input image not found: {image_path}", file=sys.stderr)
        return 1

    try:
        device = resolve_device(args.device)
        model_configs = get_default_model_configs()
        loaded_models = load_all_models(model_configs, device=device)
        results = predict_routed(image_source=image_path, loaded_models=loaded_models)
    except ModelLoadError as exc:
        print(f"Model loading error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_prediction_output(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
