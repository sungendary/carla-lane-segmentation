import torch
import numpy as np
from PIL import Image
from dataset import CarlaSegDataset
from model import build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 학습된 모델 불러오기
model = build_model(num_classes=3).to(device)
model.load_state_dict(torch.load('seg_model.pth', map_location=device))
model.eval()

# 데이터셋에서 샘플 하나 꺼내기
ds = CarlaSegDataset('dataset/town03_clear')
idx = 0
rgb, label = ds[idx]

# 예측
with torch.no_grad():
    inp = rgb.unsqueeze(0).to(device)      # 배치 차원 추가: (1,3,H,W)
    output = model(inp)['out']             # (1,3,H,W)
    pred = output.argmax(dim=1).squeeze(0) # 픽셀마다 가장 점수 높은 클래스 선택 → (H,W)
    pred = pred.cpu().numpy()

# 클래스별 색칠 (0=도로:회색, 1=차선:노랑, 2=배경:검정)
def colorize(mask):
    colors = np.array([[128,128,128],[255,255,0],[0,0,0]], dtype=np.uint8)
    return colors[mask]

# 원본 RGB 복원 (0~1 → 0~255)
rgb_vis = (rgb.permute(1,2,0).numpy() * 255).astype(np.uint8)
label_vis = colorize(label.numpy())
pred_vis = colorize(pred)

# 세 장을 가로로 이어붙여 저장
combined = np.concatenate([rgb_vis, label_vis, pred_vis], axis=1)
Image.fromarray(combined).save('prediction_result.png')
print("저장 완료: prediction_result.png (왼쪽부터 원본 / 정답 / 예측)")

# 예측에 각 클래스가 얼마나 나왔는지
unique, counts = np.unique(pred, return_counts=True)
print("예측 클래스별 픽셀 수:", dict(zip(unique.tolist(), counts.tolist())))