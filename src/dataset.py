"""
PyTorch Dataset for CARLA semantic segmentation.

Accepts one OR several dataset subfolders, so you can train across multiple
maps/conditions by listing them in config. Raw CARLA class IDs are remapped
to 3 project classes: road(1)->0, roadline(24)->1, everything else->2.
"""
import numpy as np
from PIL import Image
import glob
import os
import torch
from torch.utils.data import Dataset

from config import repo_path


class CarlaSegDataset(Dataset):
    def __init__(self, data_dirs, img_size=(512, 256)):
        # data_dirs: a string (single folder) or a list of strings.
        # Each is a subfolder name under dataset/ (e.g. 'town03_clear').
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.rgb_paths = []
        self.label_paths = []
        for d in data_dirs:
            base = repo_path('dataset', d)
            rgbs = sorted(glob.glob(os.path.join(base, 'rgb', '*.png')))
            labels = sorted(glob.glob(os.path.join(base, 'label_raw', '*.png')))
            assert len(rgbs) == len(labels), f"rgb/label count mismatch in {d}"
            self.rgb_paths.extend(rgbs)
            self.label_paths.extend(labels)

        assert len(self.rgb_paths) > 0, f"no data found in {data_dirs}"
        self.img_size = img_size

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # RGB -> model input
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        rgb = rgb.resize(self.img_size, Image.BILINEAR)
        rgb = np.array(rgb, dtype=np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)

        # label -> remap. NEAREST so class IDs are never blended.
        label_img = Image.open(self.label_paths[idx])
        label_img = label_img.resize(self.img_size, Image.NEAREST)
        label_raw = np.array(label_img)[:, :, 0]   # R channel = class id

        label = np.full(label_raw.shape, 2, dtype=np.int64)  # background
        label[label_raw == 1] = 0    # road
        label[label_raw == 24] = 1   # lane
        label = torch.from_numpy(label)

        return rgb, label


if __name__ == '__main__':
    ds = CarlaSegDataset(['town03_clear'])
    print("samples:", len(ds))
    rgb, label = ds[0]
    print("rgb:", rgb.shape, rgb.dtype)
    print("label:", label.shape, label.dtype)
    print("classes:", torch.unique(label))
