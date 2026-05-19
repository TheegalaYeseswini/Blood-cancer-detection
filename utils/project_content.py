from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "test_samples"

APP_COPY = {
    "medical_disclaimer": (
        "For educational and portfolio use only. This demo is not a clinical decision-support "
        "system and should not be used for diagnosis or treatment."
    )
}

MODEL_BENCHMARKS = [
    {
        "model": "Broad Classifier",
        "score": 99.79,
        "score_label": "99.79%",
        "metric_type": "Test Accuracy",
    },
    {
        "model": "Leukemia Subtype",
        "score": 87.62,
        "score_label": "87.62%",
        "metric_type": "Test Accuracy",
    },
    {
        "model": "Lymphoma Subtype",
        "score": 84.00,
        "score_label": "84.00%",
        "metric_type": "Validation Accuracy",
    },
]

DATASET_DISTRIBUTION = [
    {"class_group": "Leukemia", "images": 12000},
    {"class_group": "Lymphoma", "images": 374},
    {"class_group": "Myeloma", "images": 498},
    {"class_group": "Healthy", "images": 3000},
]

SAMPLE_EXPECTATIONS = {
    "ALL.jpg": "Expected route: Leukemia -> ALL",
    "MCL.png": "Expected route: Lymphoma -> MCL-like image",
    "1703.bmp": "Myeloma-style BMP sample for file-format coverage",
    "healty_test.jpg": "Healthy smear sample",
}

MODEL_CATALOG = [
    {
        "name": "Tetra Disease Classifier",
        "backbone": "EfficientNet-B0",
        "checkpoint": "models/blood_cancer.pth",
        "labels": ["LEUKEMIA", "LYMPHOMA", "MYELOMA", "HEALTHY"],
        "usage": "Runs first for every request",
        "metric": "99.79% test accuracy",
    },
    {
        "name": "Leukemia Subtype Classifier",
        "backbone": "EfficientNet-B0",
        "checkpoint": "models/lukemia_sub.pth",
        "labels": ["ALL", "AML", "CLL", "CML"],
        "usage": "Runs only when the broad model predicts leukemia",
        "metric": "87.62% test accuracy",
    },
    {
        "name": "Lymphoma Subtype Classifier",
        "backbone": "DenseNet121",
        "checkpoint": "models/lymphoma_sub.pth",
        "labels": ["CLL", "FL", "MCL"],
        "usage": "Runs only when the broad model predicts lymphoma",
        "metric": "84.00% validation accuracy",
    },
]

LEUKEMIA_REPORT = [
    {"label": "ALL", "precision": 0.91, "recall": 0.65, "f1": 0.76},
    {"label": "AML", "precision": 0.79, "recall": 0.93, "f1": 0.85},
    {"label": "CLL", "precision": 0.88, "recall": 0.92, "f1": 0.90},
    {"label": "CML", "precision": 0.95, "recall": 1.00, "f1": 0.97},
]

LYMPHOMA_REPORT = [
    {"label": "CLL", "precision": 0.87, "recall": 0.80, "f1": 0.83},
    {"label": "FL", "precision": 0.90, "recall": 0.90, "f1": 0.90},
    {"label": "MCL", "precision": 0.74, "recall": 0.81, "f1": 0.77},
]

BROAD_REPORT = [
    {"label": "Leukemia", "precision": 0.9992, "recall": 0.9990, "f1": 0.9991},
    {"label": "Lymphoma", "precision": 0.9861, "recall": 0.9467, "f1": 0.9660},
    {"label": "Myeloma", "precision": 1.0000, "recall": 1.0000, "f1": 1.0000},
    {"label": "Healthy", "precision": 0.9930, "recall": 0.9970, "f1": 0.9950},
]

NOTEBOOK_SUMMARIES = [
    {
        "notebook": "blood_cancer.ipynb",
        "task": "Broad 4-class blood cancer classification",
        "backbone": "EfficientNet-B0",
        "hyperparameters": [
            "Epochs: 10",
            "Batch size: 32",
            "Optimizer: Adam (1e-4)",
            "Scheduler: StepLR(step_size=3, gamma=0.5)",
            "Loss: weighted cross-entropy + label smoothing 0.1",
        ],
        "augmentations": [
            "Resize 224x224",
            "Horizontal flip",
            "Rotation 40",
            "Color jitter",
            "Random resized crop",
            "Gaussian blur",
            "Random grayscale",
            "ImageNet normalization",
        ],
    },
    {
        "notebook": "lukemia_sub.ipynb",
        "task": "Leukemia subtype classification",
        "backbone": "EfficientNet-B0",
        "hyperparameters": [
            "Epochs: 25 (training loop configuration)",
            "Batch size: 32",
            "Optimizer: Adam with split learning rates",
            "Scheduler: ReduceLROnPlateau",
            "Loss: cross-entropy + label smoothing 0.05",
            "Mixed precision enabled",
        ],
        "augmentations": [
            "Resize 224x224",
            "Horizontal flip",
            "Small rotation",
            "Random affine translation",
            "Color jitter",
            "ImageNet normalization",
        ],
    },
    {
        "notebook": "lymphoma_sub.ipynb",
        "task": "Lymphoma subtype classification",
        "backbone": "DenseNet121",
        "hyperparameters": [
            "Epochs: 20",
            "Batch size: 32",
            "Optimizer: AdamW with staged learning rates",
            "Scheduler: ReduceLROnPlateau",
            "Loss: FocalLoss",
        ],
        "augmentations": [
            "Resize 224x224",
            "Horizontal flip",
            "Vertical flip",
            "Rotation 20",
            "Color jitter",
            "Random resized crop",
        ],
    },
]

DEPLOYMENT_OPTIONS = [
    {
        "name": "Streamlit Cloud",
        "summary": "Best fit for quick portfolio deployment.",
        "detail": "Free-tier friendly if model files remain manageable and CPU inference is acceptable.",
    },
    {
        "name": "Hugging Face Spaces",
        "summary": "Good for AI demos with visible community presence.",
        "detail": "Works well with Streamlit SDK and is recruiter-friendly for public demos.",
    },
    {
        "name": "Render",
        "summary": "Useful when you want predictable web service deployment.",
        "detail": "Suitable for persistent apps, but cold starts may be noticeable on free plans.",
    },
]

REPO_TREE = """project/
|-- app.py
|-- pages/
|-- components/
|-- utils/
|-- assets/
|-- models/
|-- notebooks/
|-- src/
|-- requirements.txt
|-- Dockerfile
`-- README.md
"""


def get_sample_catalog() -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    if not SAMPLES_DIR.exists():
        return catalog

    for sample_path in sorted(SAMPLES_DIR.iterdir()):
        if sample_path.is_file():
            catalog.append(
                {
                    "name": sample_path.name,
                    "path": str(sample_path),
                    "suffix": sample_path.suffix.lower(),
                }
            )
    return catalog
