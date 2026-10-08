from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

extract_dir = Path("extracted_one_zip")

image_files = [
    f for f in extract_dir.rglob("*")
    if f.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
]

required_channels = ["1", "2", "4"]

groups = {}

for f in image_files:
    name = f.name
    m = re.match(r"^(.*)_w([124]).*\.tif$", name, flags=re.IGNORECASE)

    if m is None:
        continue

    base = m.group(1)
    ch = m.group(2)

    key = str(f.parent / base)
    groups.setdefault(key, {})[ch] = f

triplets = []

for key, chdict in groups.items():
    if all(ch in chdict for ch in required_channels):
        triplets.append([chdict["1"], chdict["2"], chdict["4"]])

print("Found triplets:", len(triplets))

if len(triplets) == 0:
    raise RuntimeError("还是没有找到 w1/w2/w4 三通道配对图像。")

tri = triplets[0]

imgs = []

for f in tri:
    img = np.array(Image.open(f)).astype(np.float32)

    # 背景扣除 + 鲁棒归一化
    low = np.percentile(img, 1)
    high = np.percentile(img, 99.5)

    img = img - low
    img = img / (high - low + 1e-8)
    img = np.clip(img, 0, 1)

    imgs.append(img)

A = np.stack(imgs, axis=-1)

plt.figure(figsize=(12, 4))

titles = [
    "w1 / DNA",
    "w2 / beta-tubulin",
    "w4 / F-actin"
]

for i in range(3):
    plt.subplot(1, 4, i + 1)
    plt.imshow(A[:, :, i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(A)
plt.title("Merged")
plt.axis("off")

plt.tight_layout()
plt.show()

print("A shape:", A.shape)
print("Triplet files:")
for f in tri:
    print(f)