# Multi-Model Blood Cell Classifier

This project loads and runs three trained PyTorch models:

- Leukemia subtype classifier
- Lymphoma subtype classifier
- Tetra disease classifier

The project is now self-contained: model weights live in `models/` and the training notebooks live in `notebooks/`.

## Project Structure

```text
Main_model/
|-- app.py
|-- requirements.txt
|-- models/
|   |-- blood_cancer.pth
|   |-- lukemia_sub.pth
|   `-- lymphoma_sub.pth
|-- notebooks/
|   |-- blood_cancer.ipynb
|   |-- lukemia_sub.ipynb
|   `-- lymphoma_sub.ipynb
|-- src/
|   |-- __init__.py
|   |-- load_models.py
|   |-- predict.py
|   `-- preprocess.py
`-- .vscode/
    `-- settings.json
```

## Model Mapping

- `models/blood_cancer.pth`
  Routed as the tetra classifier with labels: `LEUKEMIA`, `LYMPHOMA`, `MYELOMA`, `HEALTHY`
- `models/lukemia_sub.pth`
  Routed only when tetra predicts `LEUKEMIA`, with labels: `ALL`, `AML`, `CLL`, `CML`
- `models/lymphoma_sub.pth`
  Routed only when tetra predicts `LYMPHOMA`, with labels: `CLL`, `FL`, `MCL`

## Notebook Context

The code paths in `src/load_models.py` were matched to the notebooks in `notebooks/`:

- `notebooks/blood_cancer.ipynb`
  EfficientNet-B0 broad classifier for blood cancer vs healthy
- `notebooks/lukemia_sub.ipynb`
  EfficientNet-B0 leukemia subtype classifier
- `notebooks/lymphoma_sub.ipynb`
  DenseNet121 lymphoma subtype classifier

## Run in VS Code

1. Open the folder in VS Code.
2. Open a terminal in VS Code.
3. Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

5. Run prediction on any valid image file (`.jpg`, `.jpeg`, `.png`, or `.bmp`):

```powershell
python app.py --image ".\test_samples\ALL.jpg"
```

6. Optional JSON output:

```powershell
python app.py --image ".\test_samples\ALL.jpg" --json
```

## Notes

- The app loads model files from the local `models/` folder using project-relative paths.
- Inference is routed: the tetra classifier runs first, and only the matching subtype model is run afterward.
- Images are opened with Pillow and converted to RGB before preprocessing, so BMP-based myeloma samples are supported.
