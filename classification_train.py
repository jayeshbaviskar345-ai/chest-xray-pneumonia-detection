import time, torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from cls_model import build_model
from config import *

def main():
    print("Device:", DEVICE)
    norm = transforms.Normalize([0.485]*3, [0.229]*3)
    tf_train = transforms.Compose([
        transforms.Resize((CLS_SIZE, CLS_SIZE)), transforms.Grayscale(3),
        transforms.RandomRotation(10), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])
    tf_eval = transforms.Compose([
        transforms.Resize((CLS_SIZE, CLS_SIZE)), transforms.Grayscale(3),
        transforms.ToTensor(), norm])

    full_train = datasets.ImageFolder(f"{CLS_ROOT}/train", tf_train)
    full_val   = datasets.ImageFolder(f"{CLS_ROOT}/train", tf_eval)
    idx_tr, idx_va = train_test_split(range(len(full_train)), test_size=0.1,
                                      stratify=full_train.targets, random_state=42)
    train_dl = DataLoader(Subset(full_train, idx_tr), batch_size=CLS_BATCH,
                          shuffle=True, num_workers=4, persistent_workers=True)
    val_dl   = DataLoader(Subset(full_val, idx_va), batch_size=CLS_BATCH,
                          num_workers=2, persistent_workers=True)

    model = build_model(pretrained=True).to(DEVICE)

    # Freeze everything except the last block and the classifier
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("layer4") or name.startswith("fc")

    targets = torch.tensor([full_train.targets[i] for i in idx_tr])
    counts = torch.bincount(targets)
    weights = (counts.sum() / counts).float().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)

    best = 0
    for ep in range(CLS_EPOCHS):
        model.train(); total = 0; t0 = time.time()
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            total += loss.item()
            if i % 20 == 0:
                print(f"  epoch {ep+1} batch {i}/{len(train_dl)}  loss {loss.item():.4f}", flush=True)
        model.eval(); correct = n = 0
        with torch.no_grad():
            for x, y in val_dl:
                p = model(x.to(DEVICE)).argmax(1).cpu()
                correct += (p == y).sum().item(); n += len(y)
        acc = correct / n
        print(f"epoch {ep+1}: loss {total/len(train_dl):.4f}  val acc {acc:.4f}  ({time.time()-t0:.0f}s)")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "models/resnet18_pneumonia.pth")
    print("best val acc:", best)

if __name__ == "__main__":
    main()