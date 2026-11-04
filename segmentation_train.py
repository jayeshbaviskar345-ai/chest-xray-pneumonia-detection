import time, random, torch, numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from torch.utils.data import DataLoader, Subset
from segmentation_dataset import XrayDataset
from config import *

N_TRAIN, N_VAL, N_TEST = 800, 150, 150   # subset sizes (raise if you have time)

def subset(ds, n):
    idx = list(range(len(ds)))
    random.Random(42).shuffle(idx)
    return Subset(ds, idx[:min(n, len(ds))])

def main():
    print("Device:", DEVICE)
    aug = A.Compose([A.HorizontalFlip(), A.Rotate(limit=10)])
    train_ds = XrayDataset("Train", aug)
    print("Full train pairs:", len(train_ds))
    train_dl = DataLoader(subset(train_ds, N_TRAIN), batch_size=SEG_BATCH, shuffle=True, num_workers=0)
    val_dl   = DataLoader(subset(XrayDataset("Val"), N_VAL), batch_size=SEG_BATCH, num_workers=0)
    test_dl  = DataLoader(subset(XrayDataset("Test"), N_TEST), batch_size=SEG_BATCH, num_workers=0)

    model = smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
    dice_loss = smp.losses.DiceLoss("binary")
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def dice_score(logits, t):
        p = (torch.sigmoid(logits) > .5).float()
        return ((2*(p*t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)).item()

    def evaluate(dl):
        model.eval(); s = []
        with torch.no_grad():
            for x, y in dl:
                s.append(dice_score(model(x.to(DEVICE)), y.to(DEVICE)))
        return np.mean(s)

    best = 0
    for ep in range(SEG_EPOCHS):
        model.train(); t0 = time.time()
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            (dice_loss(out, y) + bce(out, y)).backward(); opt.step()
            if i % 10 == 0:
                print(f"  epoch {ep+1} batch {i}/{len(train_dl)}  ({time.time()-t0:.0f}s)", flush=True)
        v = evaluate(val_dl)
        print(f"epoch {ep+1}: val dice {v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if v > best or ep == 0:
            best = v
            torch.save(model.state_dict(), "models/unet_seg.pth")   # saved every improvement

    model.load_state_dict(torch.load("models/unet_seg.pth", map_location=DEVICE))
    print("test dice:", evaluate(test_dl))

if __name__ == "__main__":
    main()