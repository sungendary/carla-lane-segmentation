import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import CarlaSegDataset
from model import build_model

# ===== 설정 =====
EVAL_DIR = 'dataset/town03_sunset'   # 지금은 학습과 같은 곳, 나중에 안 본 맵으로 바꿈
NUM_CLASSES = 3
CLASS_NAMES = ['도로', '차선', '배경']
BATCH_SIZE = 4
# ================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 불러오기
model = build_model(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load('seg_model.pth', map_location=device, weights_only=True))
model.eval()

# 데이터로더
ds = CarlaSegDataset(EVAL_DIR)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 혼동 행렬 누적용 (NUM_CLASSES x NUM_CLASSES)
confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

with torch.no_grad():
    for rgb, label in loader:
        rgb = rgb.to(device)
        output = model(rgb)['out']
        pred = output.argmax(dim=1).cpu().numpy()   # (B,H,W)
        label = label.numpy()                        # (B,H,W)

        # 픽셀을 평탄화해서 혼동 행렬에 누적
        for t, p in zip(label.flatten(), pred.flatten()):
            confusion[t, p] += 1

# 클래스별 IoU 계산
print("=" * 40)
ious = []
for c in range(NUM_CLASSES):
    tp = confusion[c, c]                     # 교집합: 실제c & 예측c
    fp = confusion[:, c].sum() - tp          # 예측은 c인데 실제는 다른 것
    fn = confusion[c, :].sum() - tp          # 실제는 c인데 예측은 다른 것
    union = tp + fp + fn                      # 합집합
    iou = tp / union if union > 0 else 0.0
    ious.append(iou)
    print(f"{CLASS_NAMES[c]:4s} IoU: {iou:.4f}")

print("-" * 40)
print(f"mIoU (평균): {np.mean(ious):.4f}")
print("=" * 40)