import torch, numpy as np, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score)
from cls_model import build_model
from config import *

def main():
    tf = transforms.Compose([
        transforms.Resize((CLS_SIZE, CLS_SIZE)), transforms.Grayscale(3),
        transforms.ToTensor(), transforms.Normalize([0.485]*3, [0.229]*3)])
    test = datasets.ImageFolder(f"{CLS_ROOT}/test", tf)
    dl = DataLoader(test, batch_size=32, num_workers=0)
    print("Classes:", test.classes, "| test images:", len(test))

    model = build_model()
    model.load_state_dict(torch.load("models/resnet18_pneumonia.pth", map_location=DEVICE))
    model.to(DEVICE).eval()

    P, T, S = [], [], []
    with torch.no_grad():
        for x, y in dl:
            prob = torch.softmax(model(x.to(DEVICE)), 1).cpu()
            P += prob.argmax(1).tolist()
            S += prob[:, 1].tolist()      # probability of PNEUMONIA
            T += y.tolist()

    report = classification_report(T, P, target_names=test.classes, digits=3)
    auc = roc_auc_score(T, S)
    print(report)
    print(f"ROC-AUC: {auc:.4f}")

    with open("results/classification_report.txt", "w") as f:
        f.write(report + f"\nROC-AUC: {auc:.4f}\n")

    cm = confusion_matrix(T, P)
    ConfusionMatrixDisplay(cm, display_labels=test.classes).plot(cmap="Blues")
    plt.title("Test set confusion matrix")
    plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved results/confusion_matrix.png and results/classification_report.txt")

if __name__ == "__main__":
    main()