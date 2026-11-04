Chest X-ray Pneumonia Analysis

A Streamlit application that demonstrates chest X-ray classification and segmentation using PyTorch. Upload one chest X-ray to view the classifier's NORMAL/PNEUMONIA result, class probabilities, a pneumonia risk gauge, and—when the segmentation checkpoint is available—a highlighted region overlay and heatmap.


Research and education only. This project is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making.

Application screenshots

Overview and classification

<p align="center">
<img src="assets/app-screenshot.png" alt="Chest X-ray Pneumonia Analysis overview showing the X-ray, classification result, and risk score" width="900">
</p>

Segmentation results

<p align="center">
<img src="assets/segmentation-screenshot.png" alt="Segmentation view showing the input X-ray, mask overlay, probability heatmap, and affected-area estimate" width="900">
</p>

Features

•
Image classification: ResNet18-based classifier with NORMAL and PNEUMONIA probabilities.

•
Segmentation visualization: U-Net with a ResNet18 encoder to produce a mask overlay and probability heatmap.

•
Interactive review: Confidence, risk gauge, highlighted-area estimate, and downloadable summary/overlay.

•
Adjustable segmentation display: Mask sensitivity and overlay opacity controls.

How the models and input work

Upload a single original chest X-ray image (.png, .jpg, or .jpeg). The app sends that same image to both models; do not upload a segmentation mask. If segmentation is enabled and its checkpoint exists, the app generates an overlay.

The classifier and segmentation model in this project were trained using different datasets (the app describes Kaggle Chest X-Ray Pneumonia for classification and COVID-QU-Ex infection masks for segmentation). Therefore, the segmentation model can produce an overlay on a classification-test image, but the overlay may not be reliable for that image. An overlay appearing only means the model ran; it does not establish that the marked region is correct. Evaluate each model with data appropriate to its training/evaluation set, and compare segmentation predictions with matching ground-truth masks when available.

Project structure

Plain Text


.
├── app.py
├── cls_model.py
├── config.py
├── classification_train.py
├── classification_evaluate.py
├── segmentation_dataset.py
├── segmentation_train.py
├── predict.py
├── check_data.py
├── requirements.txt
├── data/                         # datasets (not included here)
├── models/
│   ├── resnet18_pneumonia.pth    # classifier checkpoint required by app.py
│   └── unet_seg.pth              # optional segmentation checkpoint
├── results/                      # generated outputs, if used
└── assets/
    ├── app-screenshot.png
    └── segmentation-screenshot.png



The exact contents of data/, models/, and results/ depend on your setup. Do not commit private or restricted datasets. Model checkpoints can be large; consider Git LFS or provide a documented download method if they exceed GitHub's file-size limits.

Requirements and installation

Use a Python version supported by the PyTorch and torchvision builds available for your operating system. From the repository root, create and activate a virtual environment, then install the dependencies:

Bash


python -m venv .venv



Windows PowerShell:

Plain Text


.venv\Scripts\Activate.ps1
pip install -r requirements.txt



macOS / Linux:

Bash


source .venv/bin/activate
pip install -r requirements.txt



If you need a specific CUDA-enabled PyTorch build, install the matching torch and torchvision versions using the official PyTorch installation selector before installing the remaining requirements.

Model checkpoints

Before launching the application, make sure the classifier checkpoint exists at:

Plain Text


models/resnet18_pneumonia.pth



The segmentation checkpoint is optional and should be placed at:

Plain Text


models/unet_seg.pth



The app reports an error if the classifier checkpoint is missing. If the segmentation checkpoint is absent, classification remains available and the segmentation view is disabled. Check config.py and the training scripts for the expected image sizes, dataset paths, and checkpoint-saving locations. The checkpoint architecture and label order must match the code (NORMAL, then PNEUMONIA).

Run the application

From the repository root, with the virtual environment activated and the required classifier checkpoint in place:

Bash


streamlit run app.py



Open the local URL printed by Streamlit in your browser, upload an original chest X-ray, and review the Overview, Segmentation, and Report tabs. The segmentation tab requires the segmentation checkpoint and segmentation to be enabled in the sidebar.

Training and evaluation

The repository includes scripts for classification and segmentation training/evaluation. Review config.py and each script before running them to set dataset paths and confirm the expected directory structure. Keep training, validation, and test data separated; do not evaluate on training images.

Responsible use and privacy

•
This is an educational machine-learning demonstration, not a clinically validated tool.

•
Model outputs may be wrong, biased, or inconsistent—especially when images differ from the training data.

•
Do not use predictions to make healthcare decisions.

•
Avoid uploading identifiable or sensitive medical images to any deployment unless you have appropriate authorization, safeguards, and consent.

•
Follow the dataset providers' terms and licenses when downloading, using, or redistributing data.

License

No license is specified in this repository. Add a license file if you intend to allow others to use, modify, or distribute the code.

