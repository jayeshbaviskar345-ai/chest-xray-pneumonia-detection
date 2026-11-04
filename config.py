import os, torch

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def descend(path, must_contain):
    """If the folder doesn't directly contain Train/train, go one level deeper (same-name nesting)."""
    if any(os.path.isdir(os.path.join(path, d)) for d in must_contain):
        return path
    inner = os.path.join(path, os.path.basename(path))
    if os.path.isdir(inner):
        return inner
    return path

# ---------- Classification (fast) ----------
CLS_ROOT = descend(os.path.join(BASE, "data", "archive (2)", "chest_xray"), ["train"])
CLS_SIZE = 160      # image size
CLS_EPOCHS = 3
CLS_BATCH = 32

# ---------- Segmentation (fast) ----------
SEG_ROOT = descend(os.path.join(BASE, "data", "archive (2)", "Infection Segmentation Data"), ["Train"])
MASK_FOLDER = "infection masks"      # exact folder name; check capitalisation on your disk
SEG_SIZE = 192      # must be divisible by 32
SEG_EPOCHS = 2
SEG_BATCH = 8

os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE, "results"), exist_ok=True)