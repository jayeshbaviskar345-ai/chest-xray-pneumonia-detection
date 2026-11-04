import os, glob, cv2, torch
from torch.utils.data import Dataset
from config import *

def list_pairs(split):
    pairs = []
    for img in glob.glob(os.path.join(SEG_ROOT, split, "*", "images", "*")):
        mask = img.replace(os.sep + "images" + os.sep, os.sep + MASK_FOLDER + os.sep)
        if os.path.exists(mask):
            pairs.append((img, mask))
    return sorted(pairs)

class XrayDataset(Dataset):
    def __init__(self, split, aug=None):
        self.pairs, self.aug = list_pairs(split), aug
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img, mask = cv2.imread(ip, 0), cv2.imread(mp, 0)
        img  = cv2.resize(img,  (SEG_SIZE, SEG_SIZE))
        mask = cv2.resize(mask, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_NEAREST)
        if self.aug:
            r = self.aug(image=img, mask=mask); img, mask = r["image"], r["mask"]
        img = torch.tensor(img / 255., dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        mask = torch.tensor((mask > 127).astype("float32")).unsqueeze(0)
        return img, mask