import os, yaml, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import UNet
from data.dataset import SegDataset
from utils.metrics import miou

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_one_epoch(model, dl, opt, loss_fn, device):
    model.train()
    total = 0.0
    for x, y, _ in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(dl.dataset)

@torch.no_grad()
def validate(model, dl, device, n_classes):
    model.eval()
    tot_miou, n = 0.0, 0
    for x, y, _ in dl:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tot_miou += miou(out, y, n_classes)
        n += 1
    return tot_miou / max(n, 1)

def main(cfg_path):
    cfg = load_cfg(cfg_path)
    os.makedirs("checkpoints", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SegDataset(cfg["data_root"], cfg["train_split"], augment=True,  in_ch=cfg["in_ch"])
    val_ds   = SegDataset(cfg["data_root"], cfg["val_split"],   augment=False, in_ch=cfg["in_ch"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model = UNet(in_ch=cfg["in_ch"], n_classes=cfg["n_classes"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    best = -1
    for epoch in range(cfg["epochs"]):
        tr_loss = train_one_epoch(model, train_dl, opt, loss_fn, device)
        val_miou = validate(model, val_dl, device, cfg["n_classes"])
        print(f"Epoch {epoch+1:03d} | train loss {tr_loss:.4f} | val mIoU {val_miou:.4f}")
        if val_miou > best:
            best = val_miou
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"  ✓ Saved best checkpoint (mIoU={best:.4f})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
