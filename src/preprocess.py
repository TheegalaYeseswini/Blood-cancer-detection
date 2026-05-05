from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from PIL import Image
from torchvision import transforms


ImageInput = Union[str, Path, Image.Image]


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: tuple[int, int] = (224, 224)
    mean: Optional[Sequence[float]] = None
    std: Optional[Sequence[float]] = None


def load_image(image_source: ImageInput) -> Image.Image:
    """Load an image from disk or reuse an existing PIL image."""
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    image_path = Path(image_source)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    return Image.open(image_path).convert("RGB")


def build_transform(config: PreprocessConfig) -> transforms.Compose:
    transform_steps = [
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
    ]

    if config.mean is not None and config.std is not None:
        transform_steps.append(transforms.Normalize(config.mean, config.std))

    return transforms.Compose(transform_steps)


def preprocess_image(image_source: ImageInput, config: PreprocessConfig, device: str):
    """Convert an image into a batched tensor ready for inference."""
    image = load_image(image_source)
    transform = build_transform(config)
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device)
