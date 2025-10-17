from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils import transforms as T

class SegDataset(Dataset):
    """
    假定数据目录结构：
    data_root/
      images/  id.png
      masks/   id.png  (整数标签，0/1)
    split 列表文件每行一个 id（不带扩展名）
    """
    def __init__(self, data_root, split_file, augment=True, in_ch=3):
        self.root = Path(data_root)
        self.ids = [l.strip() for l in open(split_file, 'r', encoding='utf-8') if l.strip()]
        self.augment = augment
        self.in_ch = in_ch

    def __len__(self): return len(self.ids)

    def _load_pair(self, id_):
        img_path = self.root / "images" / f"{id_}.png"
        msk_path = self.root / "masks" / f"{id_}.png"
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.array(Image.open(msk_path))
        return img, msk

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img, msk = self._load_pair(id_)

        if self.augment:
            img, msk = T.augment(img, msk)

        if self.in_ch == 4:  # 可选把 NDVI 拼成第4通道
            ndvi = T.compute_ndvi(img)  # 0~1
            img = np.concatenate([img, ndvi[..., None]], axis=2)

        img = T.to_tensor(img)          # HWC->[C,H,W], 0-1 float
        msk = torch.from_numpy(msk).long()
        return img, msk, id_
