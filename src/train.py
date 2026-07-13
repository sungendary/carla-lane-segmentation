"""
Train the segmentation model. Hyperparameters come from configs/config.yaml
under `train:`. Supports multi-folder (multi-map) training.

Adds:
  - best-model saving: only the epoch with the lowest val_loss is kept
  - early stopping: training stops if val_loss doesn't improve for
    `patience` consecutive epochs (avoids wasting time past overfitting)
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import load_config, repo_path
from dataset import CarlaSegDataset
from model import build_model

cfg = load_config('train')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# --- data ---
img_size = (cfg['img_width'], cfg['img_height'])
full_dataset = CarlaSegDataset(cfg['data_dirs'], img_size=img_size)
print(f"training folders: {cfg['data_dirs']}  (total {len(full_dataset)} frames)")

n_val = int(len(full_dataset) * cfg['val_split'])
n_train = len(full_dataset) - n_val
train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
print(f"train {n_train}, val {n_val}")

train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                          shuffle=True, num_workers=cfg['num_workers'])
val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                        shuffle=False, num_workers=cfg['num_workers'])

# --- model / loss / optimizer ---
model = build_model(num_classes=cfg['num_classes']).to(device)
class_weights = torch.tensor(cfg['class_weights']).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

# --- training loop with best-model saving + early stopping ---
out_path = repo_path(cfg['model_out'])
patience = cfg.get('patience', 5)
best_val = float('inf')
epochs_no_improve = 0

for epoch in range(cfg['epochs']):
    t0 = time.time()

    model.train()
    train_loss = 0.0
    for rgb, label in train_loader:
        rgb, label = rgb.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(rgb)['out']
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for rgb, label in val_loader:
            rgb, label = rgb.to(device), label.to(device)
            output = model(rgb)['out']
            val_loss += criterion(output, label).item()
    val_loss /= len(val_loader)

    dt = time.time() - t0

    # best-model check
    if val_loss < best_val:
        best_val = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), out_path)
        flag = "  <- best, saved"
    else:
        epochs_no_improve += 1
        flag = f"  (no improve {epochs_no_improve}/{patience})"

    print(f"[{epoch+1}/{cfg['epochs']}] train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  {dt:.0f}s{flag}")

    # early stopping
    if epochs_no_improve >= patience:
        print(f"Early stopping: val_loss did not improve for {patience} epochs.")
        break

print(f"Best val_loss: {best_val:.4f}")
print(f"Best model saved at: {out_path}")