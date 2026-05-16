import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2
from glob import glob
from tqdm import tqdm
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(directory: str):
    """Load all .npz patches from a directory.
    Returns a list of (img_tensor, mask) tuples:
        img_tensor : (C, H, W) float32  — raw reflectance, invalid pixels kept as-is
        mask       : (C, H, W) bool     — True = valid pixel
    """
    data = []
    all_files = np.array(
        sorted(
            glob(os.path.join(directory, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", "")),
        )
    )
    for file_name in all_files:
        with np.load(file_name) as npz:
            raw_arr    = np.ma.MaskedArray(data=npz["data"], mask=npz["mask"])
            img_tensor = torch.as_tensor(raw_arr.data, dtype=torch.float32)
            mask       = ~torch.as_tensor(raw_arr.mask)   # 1=valid, 0=invalid
        data.append((img_tensor, mask))
    return data


def load_gt(file_path: str) -> np.ndarray:
    """Load ground-truth labels (P, K, Mg, pH) from CSV.
    Returns a 2-D numpy array of shape (N, 4).
    """
    gt_file = pd.read_csv(file_path)
    return gt_file[["P", "K", "Mg", "pH"]].values


# ---------------------------------------------------------------------------
# PCA preprocessing helpers
# ---------------------------------------------------------------------------

def collect_valid_pixels(data_list, max_pixels: int = 3_000_000) -> np.ndarray:
    """Collect valid (non-masked) pixels from all patches for PCA fitting.
    Sampling is applied when the total would exceed max_pixels.
    Returns array of shape (N_valid_pixels, C).
    """
    all_pixels = []
    total = sum(x.shape[1] * x.shape[2] for x, _ in data_list)
    ratio = min(1.0, max_pixels / total)
    if ratio < 1.0:
        print(f"Sampling {ratio*100:.1f}% of pixels to stay within {max_pixels:,} limit.")

    for x, mask in data_list:
        data_np  = x.numpy()                                          # (C, H, W)
        mask_np  = mask[0].numpy().astype(bool)                       # (H, W)
        pixels   = np.transpose(data_np, (1, 2, 0)).reshape(-1, data_np.shape[0])
        valid    = pixels[mask_np.flatten()]
        if valid.shape[0] == 0:
            continue
        if ratio < 1.0:
            n = max(1, int(valid.shape[0] * ratio))
            valid = valid[np.random.choice(valid.shape[0], n, replace=False)]
        all_pixels.append(valid)

    if not all_pixels:
        return np.empty((0, data_list[0][0].shape[0]))
    return np.vstack(all_pixels)


def transform_patches_with_mask(data_list, scaler, pca):
    """Apply a fitted StandardScaler + PCA to every patch.
    Returns a list of (pca_tensor, pca_mask) tuples:
        pca_tensor : (n_components, H, W) float32
        pca_mask   : (n_components, H, W) bool, 1 = valid
    """
    out = []
    for x, mask in data_list:
        C, H, W  = x.numpy().shape
        pixels   = np.transpose(x.numpy(), (1, 2, 0)).reshape(-1, C)
        pca_pix  = pca.transform(scaler.transform(pixels))
        pca_img  = torch.from_numpy(
            np.transpose(pca_pix.reshape(H, W, -1), (2, 0, 1)).astype(np.float32)
        )
        n_comp    = pca_img.shape[0]
        pca_mask  = mask[0].unsqueeze(0).expand(n_comp, -1, -1).clone()
        pca_img   = pca_img * pca_mask.float()   # zero out invalid pixels
        out.append((pca_img, pca_mask))
    return out


def _pad_to_size_static(x, mask, size=(224, 224)):
    _, h, w = x.shape
    th, tw  = size
    if h > th:
        sh = (h - th) // 2
        x, mask = x[:, sh:sh + th, :], mask[:, sh:sh + th, :]
        h = th
    if w > tw:
        sw = (w - tw) // 2
        x, mask = x[:, :, sw:sw + tw], mask[:, :, sw:sw + tw]
        w = tw
    pad_h, pad_w = th - h, tw - w
    pt, pl = pad_h // 2, pad_w // 2
    x    = F.pad(x,    (pl, pad_w - pl, pt, pad_h - pt), value=0)
    mask = F.pad(mask, (pl, pad_w - pl, pt, pad_h - pt), value=0)
    return x, mask


def calculate_global_stats(X_data, pad_size=(224, 224)):
    """Compute per-channel mean and std over valid pixels of PCA-transformed patches."""
    print("Calculating global per-channel stats …")
    n_ch = X_data[0][0].shape[0]
    channel_data = [[] for _ in range(n_ch)]
    for x_item, mask_item in tqdm(X_data, desc="Global stats"):
        xp, mp = _pad_to_size_static(x_item, mask_item, size=tuple(pad_size))
        for c in range(n_ch):
            channel_data[c].extend(xp[c][mp[c]].cpu().numpy())
    means = torch.tensor([np.mean(cd) for cd in channel_data], dtype=torch.float32)
    stds  = torch.tensor([np.std(cd)  for cd in channel_data], dtype=torch.float32)
    print(f"  Means: {means}\n  Stds:  {stds}")
    return means, stds


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

class RandomRotate90:
    """Rotate by a random multiple of 90 degrees — valid for field patches."""
    def __call__(self, x):
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(x, k, dims=(-2, -1))


class RandomSpectralDrop:
    """Zero out random spectral channels to prevent over-reliance on specific bands."""
    def __init__(self, drop_prob: float = 0.05):
        self.drop_prob = drop_prob

    def __call__(self, x):
        keep = torch.bernoulli(torch.ones(x.shape[0]) * (1 - self.drop_prob))
        return x * keep.view(-1, 1, 1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NPZDataset(Dataset):
    """Dataset for PCA-transformed hyperspectral patches.

    Each item is (x, num_valid_pixels, y) for labelled data,
    or (x, num_valid_pixels) for submission data.

    Args:
        tensor_list  : list of (pca_tensor, pca_mask) tuples
        labels       : np.ndarray of shape (N, 4) or None
        augment      : apply spatial + spectral augmentations
        size         : target spatial size after pad/crop
        global_means : per-channel mean for normalization
        global_stds  : per-channel std for normalization
        aug_cfg      : augmentation config (spectral_drop_prob, resized_crop_scale)
    """

    def __init__(self, tensor_list, labels=None, augment=True,
                 size=(224, 224), global_means=None, global_stds=None, aug_cfg=None):
        self.tensor_list  = tensor_list
        self.labels       = labels
        self.augment      = augment
        self.size         = size
        self.global_means = global_means
        self.global_stds  = global_stds

        drop_prob    = aug_cfg.spectral_drop_prob   if aug_cfg else 0.05
        crop_scale   = aug_cfg.resized_crop_scale   if aug_cfg else [0.8, 1.0]

        self.transform_aug = v2.Compose([
            v2.RandomResizedCrop(size=self.size, scale=tuple(crop_scale)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            RandomRotate90(),
            RandomSpectralDrop(drop_prob=drop_prob),
        ])

    def __len__(self):
        return len(self.tensor_list)

    def pad_to_size(self, x, mask):
        return _pad_to_size_static(x, mask, size=self.size)

    def normalize_with_mask(self, x, mask):
        if self.global_means is not None and self.global_stds is not None:
            x = (x - self.global_means.to(x.device).view(-1, 1, 1)) / \
                (self.global_stds.to(x.device).view(-1, 1, 1) + 1e-6)
        return x * mask.float()

    def __getitem__(self, idx):
        x, mask = self.tensor_list[idx]
        x, mask = self.pad_to_size(x, mask)

        num_valid_pixels = mask[0].sum().float()

        x = self.normalize_with_mask(x, mask)

        if self.augment and self.labels is not None:
            x = self.transform_aug(x)
            x = x * mask.float()   # re-zero invalid pixels after spatial augmentation

        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, num_valid_pixels, y
        return x, num_valid_pixels
