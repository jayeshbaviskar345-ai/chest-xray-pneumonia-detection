import os, glob
from config import *

print("Classification:")
for split in ["train", "val", "test"]:
    for c in ["NORMAL", "PNEUMONIA"]:
        n = len(glob.glob(os.path.join(CLS_ROOT, split, c, "*")))
        print(f"  {split}/{c}: {n}")

print("\nSegmentation:")
for split in ["Train", "Val", "Test"]:
    imgs = glob.glob(os.path.join(SEG_ROOT, split, "*", "images", "*"))
    masks = glob.glob(os.path.join(SEG_ROOT, split, "*", MASK_FOLDER, "*"))
    print(f"  {split}: {len(imgs)} images, {len(masks)} masks")