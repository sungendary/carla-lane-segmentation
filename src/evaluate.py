"""
Evaluate a trained model on one dataset folder with per-class IoU / mIoU.
Settings come from configs/config.yaml under `eval:`.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import load_config, repo_path
from dataset import CarlaSegDataset
from model import build_model

cfg = load_config('eval')
CLASS_NAMES = ['road', 'lane', 'background']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = build_model(num_classes=cfg['num_classes']).to(device)
model.load_state_dict(torch.load(repo_path(cfg['model_path']),
                                 map_location=device, weights_only=True))
model.eval()

img_size = (cfg['img_width'], cfg['img_height'])
ds = CarlaSegDataset(cfg['eval_dir'], img_size=img_size)
loader = DataLoader(ds, batch_size=cfg['batch_size'],
                    shuffle=False, num_workers=cfg['num_workers'])

n = cfg['num_classes']
confusion = np.zeros((n, n), dtype=np.int64)

with torch.no_grad():
    for rgb, label in loader:
        rgb = rgb.to(device)
        pred = model(rgb)['out'].argmax(dim=1).cpu().numpy()
        label = label.numpy()
        for t, p in zip(label.flatten(), pred.flatten()):
            confusion[t, p] += 1

print("=" * 40)
print(f"eval on: {cfg['eval_dir']}")
print("-" * 40)
ious = []
for c in range(n):
    tp = confusion[c, c]
    fp = confusion[:, c].sum() - tp
    fn = confusion[c, :].sum() - tp
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    ious.append(iou)
    print(f"{CLASS_NAMES[c]:10s} IoU: {iou:.4f}")
print("-" * 40)
print(f"mIoU: {np.mean(ious):.4f}")
print("=" * 40)
