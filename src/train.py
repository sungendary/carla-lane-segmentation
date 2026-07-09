"""
Train the segmentation model. All hyperparameters come from
configs/config.yaml under `train:`. Supports training across multiple
dataset folders (map diversity) via the data_dirs list.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import load_config, repo_path
from dataset import CarlaSegDataset
from model import build_model

cfg = load_config('train')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# --- data (one or several folders) ---
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

# --- training loop ---
for epoch in range(cfg['epochs']):
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

    print(f"[{epoch+1}/{cfg['epochs']}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

# --- save ---
out_path = repo_path(cfg['model_out'])
torch.save(model.state_dict(), out_path)
print("saved:", out_path)
