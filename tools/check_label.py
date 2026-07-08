import numpy as np
from PIL import Image
import glob

# label_raw에서 아무 파일 하나 열기
path = sorted(glob.glob('dataset/town03_clear/label_raw/*.png'))[0]
img = np.array(Image.open(path))

print("이미지 shape:", img.shape)      # (높이, 너비, 채널) 예상
print("R 채널에 담긴 클래스 번호들:", np.unique(img[:, :, 0]))