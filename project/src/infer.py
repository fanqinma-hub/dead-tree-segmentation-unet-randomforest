import os, yaml, torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models.unet import UNet
from data.dataset import SegDataset

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def main(cfg_path, ckpt):
    cfg = load_cfg(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)

    ds = SegDataset(cfg["data_root"], cfg["val_split"], augment=False, in_ch=cfg["in_ch"])
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = UNet(in_ch=cfg["in_ch"], n_classes=cfg["n_classes"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    for i, (x, _, id_) in enumerate(dl):
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, 1, keepdim=True).float()  # [B,1,H,W]
        save_image(pred, f"results/{id_[0]}_pred.png")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best_model.pth")
    args = ap.parse_args()
    main(args.config, args.ckpt)
