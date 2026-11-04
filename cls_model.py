import torch.nn as nn
from torchvision import models

def build_model(pretrained=False):
    m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m