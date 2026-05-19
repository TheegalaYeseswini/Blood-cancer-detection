from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.load_models import get_default_model_configs, load_all_models
from src.predict import predict_single_model


ProgressCallback = Callable[[float, str], None]


def resolve_device(device_preference: str) -> str:
    if device_preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_preference == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_preference


@st.cache_resource(show_spinner=False)
def load_cached_models(device: str):
    configs = get_default_model_configs()
    return load_all_models(configs, device=device)


def read_uploaded_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def load_image_from_path(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def run_routed_inference(
    image: Image.Image,
    source_name: str,
    device_preference: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    def notify(progress: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(progress, message)

    device = resolve_device(device_preference)
    notify(0.12, "Loading cached model bundle")
    loaded_models = load_cached_models(device)

    start_time = time.perf_counter()
    notify(0.38, "Running broad classifier")
    tetra_prediction = predict_single_model(image, loaded_models["tetraclassifier"])

    results: dict[str, object] = {
        "tetraclassifier": tetra_prediction,
        "selected_subtype_model": None,
        "combined": {},
        "meta": {
            "source_name": source_name,
            "device": device,
            "inference_ms": 0.0,
        },
    }

    broad_label = tetra_prediction["predicted_label"]

    if broad_label == "LEUKEMIA":
        notify(0.72, "Routing to leukemia subtype classifier")
        subtype_prediction = predict_single_model(image, loaded_models["leukemia"])
        results["selected_subtype_model"] = subtype_prediction
        results["combined"] = {
            "primary_label": broad_label,
            "secondary_label": subtype_prediction["predicted_label"],
            "used_subtype_model": "leukemia",
            "summary": (
                "The broad classifier predicted leukemia, so the leukemia subtype model "
                "was executed to refine the diagnosis."
            ),
        }
    elif broad_label == "LYMPHOMA":
        notify(0.72, "Routing to lymphoma subtype classifier")
        subtype_prediction = predict_single_model(image, loaded_models["lymphoma"])
        results["selected_subtype_model"] = subtype_prediction
        results["combined"] = {
            "primary_label": broad_label,
            "secondary_label": subtype_prediction["predicted_label"],
            "used_subtype_model": "lymphoma",
            "summary": (
                "The broad classifier predicted lymphoma, so the lymphoma subtype model "
                "was executed to refine the diagnosis."
            ),
        }
    else:
        notify(0.72, "No subtype routing needed")
        results["combined"] = {
            "primary_label": broad_label,
            "secondary_label": "N/A",
            "used_subtype_model": None,
            "summary": (
                "The broad classifier predicted a terminal category, so no subtype model "
                "was required."
            ),
        }

    results["meta"]["inference_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
    notify(1.0, "Inference complete")
    return results


def probabilities_to_frame(result: dict[str, object]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    broad_prediction = result["tetraclassifier"]
    for label, score in broad_prediction["probabilities"].items():
        records.append({"model": "broad", "label": label, "probability": score})

    subtype_prediction = result["selected_subtype_model"]
    if subtype_prediction is not None:
        for label, score in subtype_prediction["probabilities"].items():
            records.append(
                {
                    "model": result["combined"]["used_subtype_model"],
                    "label": label,
                    "probability": score,
                }
            )

    return pd.DataFrame(records)
