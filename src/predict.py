from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from src.load_models import LoadedModel
from src.preprocess import ImageInput, preprocess_image


def predict_all(image_source: ImageInput, loaded_models: Dict[str, LoadedModel]) -> Dict[str, Any]:
    predictions: Dict[str, Any] = {}

    for key, loaded_model in loaded_models.items():
        predictions[key] = predict_single_model(image_source, loaded_model)

    predictions["combined"] = build_combined_summary(predictions)
    return predictions


def predict_routed(
    image_source: ImageInput, loaded_models: Dict[str, LoadedModel]
) -> Dict[str, Any]:
    tetra_prediction = predict_single_model(
        image_source=image_source,
        loaded_model=loaded_models["tetraclassifier"],
    )

    results: Dict[str, Any] = {
        "tetraclassifier": tetra_prediction,
        "selected_subtype_model": None,
        "combined": {},
    }

    broad_label = tetra_prediction["predicted_label"]

    if broad_label == "LEUKEMIA":
        subtype_prediction = predict_single_model(
            image_source=image_source,
            loaded_model=loaded_models["leukemia"],
        )
        results["selected_subtype_model"] = subtype_prediction
        results["combined"] = {
            "primary_label": broad_label,
            "secondary_label": subtype_prediction["predicted_label"],
            "used_subtype_model": "leukemia",
            "summary": (
                "The tetra classifier predicts leukemia, so only the leukemia "
                f"subtype model was run. Final subtype: '{subtype_prediction['predicted_label']}'."
            ),
        }
        return results

    if broad_label == "LYMPHOMA":
        subtype_prediction = predict_single_model(
            image_source=image_source,
            loaded_model=loaded_models["lymphoma"],
        )
        results["selected_subtype_model"] = subtype_prediction
        results["combined"] = {
            "primary_label": broad_label,
            "secondary_label": subtype_prediction["predicted_label"],
            "used_subtype_model": "lymphoma",
            "summary": (
                "The tetra classifier predicts lymphoma, so only the lymphoma "
                f"subtype model was run. Final subtype: '{subtype_prediction['predicted_label']}'."
            ),
        }
        return results

    results["combined"] = {
        "primary_label": broad_label,
        "secondary_label": "N/A",
        "used_subtype_model": None,
        "summary": (
            "The tetra classifier predicts a non-subtyped category, so no subtype "
            "model was run."
        ),
    }
    return results


def predict_single_model(image_source: ImageInput, loaded_model: LoadedModel) -> Dict[str, Any]:
    if not isinstance(loaded_model.model, torch.nn.Module):
        raise TypeError(
            f"Model '{loaded_model.config.name}' is not a torch.nn.Module. "
            "Update predict_single_model if you want to support a different model type."
        )

    batch = preprocess_image(
        image_source=image_source,
        config=loaded_model.config.preprocess,
        device=loaded_model.device,
    )

    with torch.no_grad():
        outputs = loaded_model.model(batch)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_index = int(np.argmax(probabilities))
    class_names = loaded_model.config.class_names

    return {
        "model_name": loaded_model.config.name,
        "predicted_index": predicted_index,
        "predicted_label": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            label: float(probability)
            for label, probability in zip(class_names, probabilities)
        },
    }
