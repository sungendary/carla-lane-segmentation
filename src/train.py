import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import CarlaSegDataset
from model import build_model

# ===== 설정 =====
DATA_DIR = 'dataset/town03_clear'
NUM_CLASSES = 3
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
# ================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("사용 장치:", device)

# --- 부품1: 데이터셋 + 학습/검증 분리 ---
full_dataset = CarlaSegDataset(DATA_DIR)
n_val = int(len(full_dataset) * 0.2)
n_train = len(full_dataset) - n_val
train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
print(f"학습 {n_train}장, 검증 {n_val}장")

# --- 부품2: 데이터로더 (배치로 묶고 섞기) ---
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- 부품3: 모델 ---
model = build_model(num_classes=NUM_CLASSES).to(device)

# --- 부품4: 손실 함수 (차선 클래스에 높은 가중치) ---
# 클래스 순서: 0=도로, 1=차선, 2=배경
class_weights = torch.tensor([1.0, 10.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# --- 부품5: 옵티마이저 ---
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- 학습 루프 ---
for epoch in range(EPOCHS):
    # (1) 학습 모드
    model.train()
    train_loss = 0.0
    for rgb, label in train_loader:
        rgb, label = rgb.to(device), label.to(device)

        optimizer.zero_grad()              # 이전 배치 기울기 초기화 (예고한 함정!)
        output = model(rgb)['out']         # 순전파: 예측 얻기
        loss = criterion(output, label)    # 손실 계산: 얼마나 틀렸나
        loss.backward()                    # 역전파: 어디를 고쳐야 하나
        optimizer.step()                   # 가중치 갱신: 실제로 고침

        train_loss += loss.item()
    train_loss /= len(train_loader)

    # (2) 검증 모드 (학습 안 함, 성능만 측정)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():                  # 기울기 계산 끔 (검증엔 불필요)
        for rgb, label in val_loader:
            rgb, label = rgb.to(device), label.to(device)
            output = model(rgb)['out']
            loss = criterion(output, label)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"[{epoch+1}/{EPOCHS}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

# --- 학습된 모델 저장 ---
torch.save(model.state_dict(), 'seg_model.pth')
print("모델 저장 완료: seg_model.pth")