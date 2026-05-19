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
        description="Run routed blood cancer inference from the command line."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Execution device.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no compatible GPU is available.")
    return requested_device


def format_output(results: dict) -> str:
    lines = []
    broad_prediction = results["tetraclassifier"]
    lines.append(
        f"Broad classifier: {broad_prediction['predicted_label']} "
        f"({broad_prediction['confidence']:.4f})"
    )
    subtype_prediction = results["selected_subtype_model"]
    if subtype_prediction is not None:
        lines.append(
            f"Subtype classifier: {subtype_prediction['predicted_label']} "
            f"({subtype_prediction['confidence']:.4f})"
        )
    lines.append(
        f"Final result: {results['combined']['primary_label']} / "
        f"{results['combined']['secondary_label']}"
    )
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
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_output(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
