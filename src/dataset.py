import numpy as np
from PIL import Image
import glob
import os
import torch
from torch.utils.data import Dataset


class CarlaSegDataset(Dataset):
    def __init__(self, data_dir, img_size=(512, 256)):
        # data_dir 예: 'dataset/town03_clear'
        # rgb와 label_raw 파일 목록을 정렬해서 짝을 맞춘다
        self.rgb_paths = sorted(glob.glob(os.path.join(data_dir, 'rgb', '*.png')))
        self.label_paths = sorted(glob.glob(os.path.join(data_dir, 'label_raw', '*.png')))
        assert len(self.rgb_paths) == len(self.label_paths), "rgb와 label 개수 불일치"
        self.img_size = img_size   # (너비, 높이) — 학습 속도 위해 원본보다 줄임

    def __len__(self):
        # 전체 샘플 개수
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # --- 1) RGB 읽기 → 모델 입력 형태로 ---
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        rgb = rgb.resize(self.img_size, Image.BILINEAR)
        rgb = np.array(rgb, dtype=np.float32) / 255.0        # 0~1로 정규화
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)         # (H,W,C) → (C,H,W)

        # --- 2) 라벨 읽기 → 재매핑 ---
        label_img = Image.open(self.label_paths[idx])
        # 라벨은 최근접 보간으로 줄여야 함 (번호가 섞이면 안 되므로)
        label_img = label_img.resize(self.img_size, Image.NEAREST)
        label_raw = np.array(label_img)[:, :, 0]             # R 채널 = 클래스 번호

        # 재매핑: 1(Road)→0, 24(RoadLine)→1, 나머지→2(배경)
        label = np.full(label_raw.shape, 2, dtype=np.int64)  # 기본값 배경
        label[label_raw == 1] = 0    # 주행가능영역
        label[label_raw == 24] = 1   # 차선
        label = torch.from_numpy(label)                      # (H,W)

        return rgb, label

if __name__ == '__main__':
    ds = CarlaSegDataset('dataset/town03_clear')
    print("전체 샘플 수:", len(ds))
    rgb, label = ds[0]
    print("rgb shape:", rgb.shape, "dtype:", rgb.dtype)
    print("label shape:", label.shape, "dtype:", label.dtype)
    print("label에 존재하는 클래스:", torch.unique(label))
    print("rgb 값 범위:", rgb.min().item(), "~", rgb.max().item())