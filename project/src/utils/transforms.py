import numpy as np
import torch

def augment(img, msk):
    # 简单增强：水平/垂直翻转 + 90° 旋转
    if np.random.rand() < .5:
        img = np.flip(img, 1).copy()
        msk = np.flip(msk, 1).copy()
    if np.random.rand() < .5:
        img = np.flip(img, 0).copy()
        msk = np.flip(msk, 0).copy()
    if np.random.rand() < .25:
        k = np.random.randint(1, 4)
        img = np.rot90(img, k).copy()
        msk = np.rot90(msk, k).copy()
    return img, msk

def to_tensor(img):
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.5:  # 0-255 -> 0-1
        img /= 255.0
    return torch.from_numpy(img.transpose(2, 0, 1))

def compute_ndvi(img_rgb):
    # 若只有 RGB，做个近似（演示用）；真实应使用 NIR 通道
    r = img_rgb[..., 0].astype(np.float32)
    ir = img_rgb[..., 1].astype(np.float32)  # 假近红外占位
    ndvi = (ir - r) / (ir + r + 1e-6)
    ndvi = (ndvi + 1) / 2.0  # [-1,1] -> [0,1]
    return np.clip(ndvi, 0, 1)
