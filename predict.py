import sys, cv2, torch, numpy as np, torch.nn as nn
from torchvision import models, transforms
import segmentation_models_pytorch as smp
from PIL import Image
from config import *

img_path = sys.argv[1]

# Classification
clf = models.resnet50(); clf.fc = nn.Linear(clf.fc.in_features, 2)
clf.load_state_dict(torch.load("models/resnet50_pneumonia.pth", map_location=DEVICE))
clf.to(DEVICE).eval()
tf = transforms.Compose([transforms.Resize((CLS_SIZE, CLS_SIZE)), transforms.Grayscale(3),
                         transforms.ToTensor(), transforms.Normalize([0.485]*3, [0.229]*3)])
x = tf(Image.open(img_path)).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    prob = torch.softmax(clf(x), 1)[0].cpu()
label = ["NORMAL", "PNEUMONIA"][prob.argmax().item()]
print(f"Prediction: {label} ({prob.max():.2%})")

# Segmentation
seg = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
seg.load_state_dict(torch.load("models/unet_seg.pth", map_location=DEVICE))
seg.to(DEVICE).eval()
g = cv2.resize(cv2.imread(img_path, 0), (SEG_SIZE, SEG_SIZE))
t = torch.tensor(g/255., dtype=torch.float32).unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    mask = (torch.sigmoid(seg(t))[0, 0] > .5).cpu().numpy().astype(np.uint8)

overlay = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
overlay[mask == 1] = (0.5*overlay[mask == 1] + 0.5*np.array([0, 0, 255])).astype(np.uint8)
cv2.imwrite("results/prediction_overlay.png", overlay)
print("Saved results/prediction_overlay.png")