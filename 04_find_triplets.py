from pathlib import Path
import re

extract_dir = Path("extracted_all_zips")

image_files = [
    f for f in extract_dir.rglob("*")
    if f.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
]

print("Found image files:", len(image_files))

# BBBC021 这批文件通常是 w1, w2, w4 三个通道
required_channels = ["1", "2", "4"]

groups = {}

for f in image_files:
    name = f.name

    # 匹配：
    # Week10_200907_B04_s1_w1B31DB1BD-....
    # 捕获 base = Week10_200907_B04_s1
    # 捕获 ch = 1 / 2 / 4
    m = re.match(r"^(.*)_w([124]).*\.tif$", name, flags=re.IGNORECASE)

    if m is None:
        continue

    base = m.group(1)
    ch = m.group(2)

    # key 需要包含文件夹路径，避免不同文件夹里同名 base 冲突
    key = str(f.parent / base)

    groups.setdefault(key, {})[ch] = f

triplets = []

for key, chdict in groups.items():
    if all(ch in chdict for ch in required_channels):
        triplets.append([chdict["1"], chdict["2"], chdict["4"]])

print("Found triplets:", len(triplets))

for i, tri in enumerate(triplets[:10]):
    print(f"\nTriplet {i+1}:")
    print("w1:", tri[0])
    print("w2:", tri[1])
    print("w4:", tri[2])