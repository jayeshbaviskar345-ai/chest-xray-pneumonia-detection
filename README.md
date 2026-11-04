# Chest X-ray Pneumonia Detection

**Deep Learning · PyTorch · Image Classification · Image Segmentation**

A PyTorch and Streamlit project that demonstrates chest X-ray classification and segmentation. It combines a ResNet18-based classifier with a U-Net segmentation model in one interactive application.

> **Disclaimer:** For learning and research demonstration only. This project is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making.

## Project overview

Upload one original chest X-ray image to view the classifier's prediction (**NORMAL** or **PNEUMONIA**), probabilities, confidence, and risk gauge. When the segmentation checkpoint is available and segmentation is enabled, the app also displays a predicted mask overlay, probability heatmap, and highlighted-area estimate.

| Task | Model | Output |
|---|---|---|
| Classification | ResNet18 classifier | NORMAL / PNEUMONIA prediction and class probabilities |
| Segmentation | U-Net with ResNet18 encoder | Pixel-wise infection map, overlay, and heatmap |

## Screenshots

### Classification dashboard

![Classification dashboard showing the original X-ray, class probability, pneumonia risk score, and prediction cards](https://private-us-east-1.manuscdn.com/sessionFile/MQeJrZn2pfJnanoKIC1tLQ/sandbox/olCpt4JOWlC9wLPDwuFrIa-images_1793801090257_na1fn_L2hvbWUvdWJ1bnR1L3hyYXktcG5ldW1vbmlhLWdpdGh1Yi9hc3NldHMvYXBwLXNjcmVlbnNob3Q.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvTVFlSnJabjJwZkpuYW5vS0lDMXRMUS9zYW5kYm94L29sQ3B0NEpPV2xDOXdMUER3dUZySWEtaW1hZ2VzXzE3OTM4MDEwOTAyNTdfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzaHlZWGt0Y0c1bGRXMXZibWxoTFdkcGRHaDFZaTloYzNObGRITXZZWEJ3TFhOamNtVmxibk5vYjNRLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5NjA4MzIwMH19fV19&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEUCIQCXQnEpZXMJLWBlBlsEoEHH21ch6o5m22xLY37nLQGJpQIgRIBNtud-fD2pUMVLQykxTlbJN6r7l9-odinmCq4WPOs_)

### Segmentation results

![Segmentation dashboard showing the input X-ray, mask overlay, probability heatmap, and affected-area estimate](https://private-us-east-1.manuscdn.com/sessionFile/MQeJrZn2pfJnanoKIC1tLQ/sandbox/olCpt4JOWlC9wLPDwuFrIa-images_1793801090257_na1fn_L2hvbWUvdWJ1bnR1L3hyYXktcG5ldW1vbmlhLWdpdGh1Yi9hc3NldHMvc2VnbWVudGF0aW9uLXNjcmVlbnNob3Q.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvTVFlSnJabjJwZkpuYW5vS0lDMXRMUS9zYW5kYm94L29sQ3B0NEpPV2xDOXdMUER3dUZySWEtaW1hZ2VzXzE3OTM4MDEwOTAyNTdfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzaHlZWGt0Y0c1bGRXMXZibWxoTFdkcGRHaDFZaTloYzNObGRITXZjMlZuYldWdWRHRjBhVzl1TFhOamNtVmxibk5vYjNRLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5NjA4MzIwMH19fV19&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEQCICU7BQ45~xyusu5Hf2OqsKuUM~dvjthDRo4fVLyYux~5AiAox7lX2~LL943Cnq16AncFA4IiUm5QVscsZwTq~wH6qg__)

## Inference workflow

1. Upload one original chest X-ray image (`.png`, `.jpg`, or `.jpeg`).
2. The classifier preprocesses the image and returns NORMAL/PNEUMONIA probabilities.
3. If enabled and its checkpoint is present, the U-Net processes the same X-ray and generates a segmentation visualization.
4. Review classification in the **Overview** tab and the mask visualization in **Segmentation**.

**Upload the original X-ray, not a segmentation mask.** The model generates its own predicted mask.

## Datasets and interpretation

The classification and segmentation models use different datasets: the classification component is described in the app as using Kaggle Chest X-Ray Pneumonia data, while segmentation uses COVID-QU-Ex infection-mask data. Because of this dataset difference, the models may behave differently on images outside their respective training distributions.

The segmentation model can produce an overlay for a classification-test image, but an overlay appearing only confirms inference ran—it does not confirm that the highlighted area is accurate. Evaluate each model on data appropriate to its task. For segmentation, compare predictions with matching ground-truth masks where available.

## Technology stack

- **Python, PyTorch, Torchvision** — model development and inference
- **ResNet18** — classification model and U-Net encoder
- **U-Net** — pixel-wise segmentation
- **Streamlit** — interactive web application
- **OpenCV, NumPy, Pillow** — image processing
- **Plotly** — probability charts and risk gauge
- **Albumentations, scikit-learn, Matplotlib, Pandas, tqdm** — training/evaluation dependencies listed in `requirements.txt`

## Repository structure

```text
.
├── app.py                      # Streamlit app and inference
├── cls_model.py                # Classification model definition
├── config.py                   # Image sizes and project configuration
├── classification_train.py     # Classifier training
├── classification_evaluate.py  # Classifier evaluation
├── segmentation_dataset.py     # Segmentation data loading
├── segmentation_train.py       # U-Net training
├── predict.py                  # Prediction utility
├── check_data.py               # Dataset inspection utility
├── requirements.txt            # Python dependencies
├── assets/
│   ├── app-screenshot.png
│   └── segmentation-screenshot.png
├── data/                       # Local datasets; not included
├── models/                     # Local checkpoints; not included
└── results/                    # Local outputs; not included
```

## Setup

Clone the repository and enter its folder:

```bash
git clone https://github.com/jayeshbaviskar345-ai/chest-xray-pneumonia-detection.git
cd chest-xray-pneumonia-detection
```

Create a virtual environment and install dependencies.

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA, install matching `torch` and `torchvision` builds for your system using the official PyTorch installation selector, then install the remaining project dependencies.

## Model checkpoints

The classifier checkpoint is required at:

```text
models/resnet18_pneumonia.pth
```

The segmentation checkpoint is optional and expected at:

```text
models/unet_seg.pth
```

Without the classifier checkpoint, the app cannot start. Without the segmentation checkpoint, classification remains available, but segmentation is unavailable. Check `config.py` and the training scripts for dataset locations, input sizes, and checkpoint-saving paths. Checkpoint architecture and class order must match the code (`NORMAL`, then `PNEUMONIA`).

Datasets and checkpoints are not included in this repository. Do not commit private or restricted data. For large model files, use Git LFS or document a trusted download method.

## Run the app

From the repository root, with the environment activated and classifier checkpoint in place:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload an X-ray, and review the **Overview**, **Segmentation**, and **Report** tabs.

## Training and evaluation

Review `config.py` and the training scripts to configure dataset paths and confirm the expected directory structure before training. Keep training, validation, and test data separate. Evaluate segmentation against corresponding ground-truth masks where possible.

## Responsible use

- Predictions and visualizations can be incorrect, especially for images unlike the training data.
- Do not use this project for clinical diagnosis or healthcare decisions.
- Do not upload identifiable medical images to a deployment without appropriate authorization, consent, and safeguards.
- Follow dataset owners' terms and licenses.

## License

No license is specified. Add a license file if you intend to allow others to use, modify, or distribute this project.
