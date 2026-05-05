from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import joblib
import torch
import torch.nn as nn
from torchvision import models

from src.preprocess import PreprocessConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    path: Path
    framework: str
    class_names: list[str]
    preprocess: PreprocessConfig
    builder: Optional[Callable[[], Any]] = None


@dataclass
class LoadedModel:
    config: ModelConfig
    model: Any
    device: str


def build_leukemia_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Sequential(
        nn.BatchNorm1d(model.classifier[1].in_features),
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 4),
    )
    return model


def build_lymphoma_model() -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 3),
    )
    return model


def build_tetraclassifier_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    return model


def get_default_model_configs() -> Dict[str, ModelConfig]:
    return {
        "leukemia": ModelConfig(
            name="Leukemia Subtype Classifier",
            path=MODELS_DIR / "lukemia_sub.pth",
            framework="torch",
            class_names=["ALL", "AML", "CLL", "CML"],
            preprocess=PreprocessConfig(
                image_size=(224, 224),
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
            builder=build_leukemia_model,
        ),
        "lymphoma": ModelConfig(
            name="Lymphoma Subtype Classifier",
            path=MODELS_DIR / "lymphoma_sub.pth",
            framework="torch",
            class_names=["CLL", "FL", "MCL"],
            preprocess=PreprocessConfig(image_size=(224, 224)),
            builder=build_lymphoma_model,
        ),
        "tetraclassifier": ModelConfig(
            name="Tetra Disease Classifier",
            path=MODELS_DIR / "blood_cancer.pth",
            framework="torch",
            class_names=["LEUKEMIA", "LYMPHOMA", "MYELOMA", "HEALTHY"],
            preprocess=PreprocessConfig(
                image_size=(224, 224),
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
            builder=build_tetraclassifier_model,
        ),
    }


def load_all_models(model_configs: Dict[str, ModelConfig], device: str = "cpu") -> Dict[str, LoadedModel]:
    loaded_models: Dict[str, LoadedModel] = {}
    for key, config in model_configs.items():
        loaded_models[key] = load_model(config, device=device)
    return loaded_models


def load_model(config: ModelConfig, device: str = "cpu") -> LoadedModel:
    model_path = config.path.expanduser().resolve()
    if not model_path.exists():
        raise ModelLoadError(
            f"Model file for '{config.name}' was not found: {model_path}"
        )

    framework = config.framework.lower()

    if framework in {"joblib", "pickle", "pkl"}:
        try:
            model = joblib.load(model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load joblib model '{config.name}' from {model_path}: {exc}"
            ) from exc
        return LoadedModel(config=config, model=model, device=device)

    if framework in {"torch", "pytorch", "pth", "pt"}:
        return _load_torch_model(config=config, model_path=model_path, device=device)

    raise ModelLoadError(
        f"Unsupported framework '{config.framework}' for model '{config.name}'."
    )


def _load_torch_model(config: ModelConfig, model_path: Path, device: str) -> LoadedModel:
    if config.builder is None:
        raise ModelLoadError(
            f"PyTorch model '{config.name}' needs a builder function to recreate the architecture."
        )

    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to read PyTorch weights for '{config.name}' from {model_path}: {exc}"
        ) from exc

    try:
        model = config.builder()

        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        else:
            state_dict = _extract_state_dict(checkpoint)
            model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to build or initialize '{config.name}' from {model_path}: {exc}"
        ) from exc

    return LoadedModel(config=config, model=model, device=device)


def _extract_state_dict(checkpoint: Any) -> OrderedDict:
    if isinstance(checkpoint, OrderedDict):
        return checkpoint

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], (dict, OrderedDict)):
                return checkpoint[key]

    raise ModelLoadError(
        "Checkpoint format is not supported. Expected a state dict or a dictionary containing one."
    )
