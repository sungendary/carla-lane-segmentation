"""
Visualize a prediction: runs the trained model on one sample and saves an
RGB / ground-truth / prediction comparison image. Settings come from
configs/config.yaml under `predict:`.
"""
import numpy as np
from PIL import Image
import torch

from config import load_config, repo_path
from dataset import CarlaSegDataset
from model import build_model

cfg = load_config('predict')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
model = build_model(num_classes=cfg['num_classes']).to(device)
model.load_state_dict(torch.load(repo_path(cfg['model_path']),
                                 map_location=device, weights_only=True))
model.eval()

# sample
img_size = (cfg['img_width'], cfg['img_height'])
ds = CarlaSegDataset(cfg['data_dir'], img_size=img_size)
rgb, label = ds[cfg['index']]

# predict
with torch.no_grad():
    inp = rgb.unsqueeze(0).to(device)
    pred = model(inp)['out'].argmax(dim=1).squeeze(0).cpu().numpy()


def colorize(mask):
    colors = np.array([[128, 128, 128], [255, 255, 0], [0, 0, 0]], dtype=np.uint8)
    return colors[mask]


rgb_vis = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
label_vis = colorize(label.numpy())
pred_vis = colorize(pred)

combined = np.concatenate([rgb_vis, label_vis, pred_vis], axis=1)
out_path = repo_path(cfg['out_image'])
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
Image.fromarray(combined).save(out_path)
print(f"saved: {out_path}  (left: RGB / middle: ground truth / right: prediction)")

unique, counts = np.unique(pred, return_counts=True)
print("predicted class pixel counts:", dict(zip(unique.tolist(), counts.tolist())))