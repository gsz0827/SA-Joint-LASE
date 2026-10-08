from pathlib import Path
import re
import json

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy import ndimage as ndi

from skimage import filters, measure, morphology, segmentation, exposure
from skimage.feature import peak_local_max


# ============================================================
# Paths
# ============================================================

# 当前脚本所在目录：
# E:\gsz\LASE_second_paper\BBBC021_download
bbbc_root = Path(__file__).resolve().parent

# 你的解压目录：
# E:\gsz\LASE_second_paper\BBBC021_download\extracted_all_zips
extract_dir = bbbc_root / "extracted_all_zips"

# 单细胞 patch 输出目录
out_dir = bbbc_root / "data" / "processed" / "bbbc021_single_cell_patches"
out_dir.mkdir(parents=True, exist_ok=True)

# 预览图输出目录
result_dir = bbbc_root / "results" / "bbbc021_single_cell_preview"
result_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Parameters
# ============================================================

PATCH_SIZE = 128
HALF = PATCH_SIZE // 2

REQUIRED_CHANNELS = ["1", "2", "4"]

# nucleus filtering
MIN_NUC_AREA = 40
MAX_NUC_AREA = 3000

# approximate cell expansion radius in pixels
CELL_EXPAND_DISTANCE = 35

# patch QC
MIN_CELL_AREA = 300
MAX_CELL_AREA = 10000
MAX_PATCHES = 10000

RANDOM_SEED = 20260517


# ============================================================
# Utility functions
# ============================================================

def robust_normalize(img, p_low=1, p_high=99.5):
    img = img.astype(np.float32)

    low = np.percentile(img, p_low)
    high = np.percentile(img, p_high)

    img = img - low
    img = img / (high - low + 1e-8)
    img = np.clip(img, 0, 1)

    return img.astype(np.float32)


def find_triplets(extract_dir):
    image_files = [
        f for f in extract_dir.rglob("*")
        if f.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    ]

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
        if all(ch in chdict for ch in REQUIRED_CHANNELS):
            triplets.append([chdict["1"], chdict["2"], chdict["4"]])

    triplets = sorted(triplets, key=lambda x: str(x[0]))

    return triplets


def load_triplet(tri):
    imgs = []

    for f in tri:
        img = np.array(Image.open(f)).astype(np.float32)
        img = robust_normalize(img)
        imgs.append(img)

    A_fov = np.stack(imgs, axis=-1).astype(np.float32)

    return A_fov


def segment_nuclei(dna):
    """
    Segment nuclei from DNA channel.

    Input:
        dna: [H, W], normalized to 0-1

    Output:
        labels: [H, W], int32 nucleus instance labels
    """

    dna_smooth = filters.gaussian(dna, sigma=1.2)

    threshold = filters.threshold_otsu(dna_smooth)
    mask = dna_smooth > threshold

    mask = morphology.remove_small_objects(mask, min_size=MIN_NUC_AREA)
    mask = morphology.remove_small_holes(mask, area_threshold=64)

    distance = ndi.distance_transform_edt(mask)

    coords = peak_local_max(
        distance,
        labels=mask,
        min_distance=6,
        exclude_border=False,
    )

    markers = np.zeros_like(dna, dtype=np.int32)

    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    labels = segmentation.watershed(
        -distance,
        markers=markers,
        mask=mask,
    )

    # filter nuclei by area
    labels_filtered = np.zeros_like(labels, dtype=np.int32)
    new_id = 1

    props = measure.regionprops(labels)

    for prop in props:
        area = prop.area

        if area < MIN_NUC_AREA:
            continue

        if area > MAX_NUC_AREA:
            continue

        labels_filtered[labels == prop.label] = new_id
        new_id += 1

    return labels_filtered


def build_cell_labels(nuc_labels, cyto):
    """
    Approximate whole-cell labels by expanding nuclear labels.

    BBBC021 does not provide a clean membrane marker.
    For first version, expand_labels is more stable than aggressive cytoplasm watershed.
    """

    expanded = segmentation.expand_labels(
        nuc_labels,
        distance=CELL_EXPAND_DISTANCE,
    )

    # optional cytoplasm foreground restriction
    cyto_smooth = filters.gaussian(cyto, sigma=1.0)

    try:
        th = filters.threshold_otsu(cyto_smooth)
        fg = cyto_smooth > (0.5 * th)
    except Exception:
        fg = cyto_smooth > np.percentile(cyto_smooth, 40)

    fg = morphology.remove_small_holes(fg, area_threshold=128)

    # Do not fully zero by fg, because weak cytoplasm cells may disappear.
    # Use fg only to suppress far background.
    cell_labels = expanded.copy()
    cell_labels[~fg & (nuc_labels == 0)] = 0

    return cell_labels.astype(np.int32)


def crop_patch(arr, cy, cx, patch_size=128):
    half = patch_size // 2

    y0 = int(round(cy)) - half
    y1 = y0 + patch_size
    x0 = int(round(cx)) - half
    x1 = x0 + patch_size

    if y0 < 0 or x0 < 0 or y1 > arr.shape[0] or x1 > arr.shape[1]:
        return None, None

    return arr[y0:y1, x0:x1], (y0, y1, x0, x1)


def touches_border(mask):
    return (
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def make_single_cell_patches_from_fov(A_fov, fov_id):
    """
    Input:
        A_fov: [H, W, 3]
            channel 0: DNA
            channel 1: beta-tubulin
            channel 2: F-actin

    Output:
        list of dicts
    """

    dna = A_fov[:, :, 0]
    tub = A_fov[:, :, 1]
    actin = A_fov[:, :, 2]

    cyto = np.maximum(tub, actin)

    nuc_labels = segment_nuclei(dna)
    cell_labels = build_cell_labels(nuc_labels, cyto)

    props = measure.regionprops(nuc_labels)

    patches = []

    for prop in props:
        label = prop.label
        cy, cx = prop.centroid

        A_crop, box = crop_patch(A_fov, cy, cx, PATCH_SIZE)
        if A_crop is None:
            continue

        cell_crop, _ = crop_patch(cell_labels, cy, cx, PATCH_SIZE)
        nuc_crop, _ = crop_patch(nuc_labels, cy, cx, PATCH_SIZE)

        current_cell_mask = cell_crop == label
        current_nuc_mask = nuc_crop == label

        cell_area = int(current_cell_mask.sum())
        nuc_area = int(current_nuc_mask.sum())

        if cell_area < MIN_CELL_AREA:
            continue

        if cell_area > MAX_CELL_AREA:
            continue

        if nuc_area < MIN_NUC_AREA:
            continue

        # reject likely truncated cells
        if touches_border(current_cell_mask):
            continue

        # reject if another nucleus is too close to the center region
        other_nuc = (nuc_crop > 0) & (nuc_crop != label)
        other_area = int(other_nuc.sum())

        # allow tiny fragments, reject obvious neighboring nuclei
        if other_area > 20:
            continue

        # Apply single-cell mask to remove neighboring cells and background.
        A_patch = A_crop * current_cell_mask[:, :, None].astype(np.float32)

        # Ensure DNA nucleus remains visible.
        if A_patch[:, :, 0].max() < 0.1:
            continue

        meta = {
            "fov_id": fov_id,
            "label": int(label),
            "centroid_y": float(cy),
            "centroid_x": float(cx),
            "cell_area": int(cell_area),
            "nuc_area": int(nuc_area),
            "box": [int(v) for v in box],
        }

        patches.append({
            "A": A_patch.astype(np.float32),
            "mask": current_cell_mask.astype(np.uint8),
            "nuc_mask": current_nuc_mask.astype(np.uint8),
            "meta": meta,
        })

    return patches


def save_preview(A_patches, out_path, max_show=8):
    n = min(len(A_patches), max_show)

    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5 * n), constrained_layout=True)

    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        A = A_patches[i]

        titles = ["DNA", "beta-tubulin", "F-actin", "Merged"]

        for ch in range(3):
            axes[i, ch].imshow(A[:, :, ch], cmap="gray", vmin=0, vmax=1)
            axes[i, ch].set_title(titles[ch])
            axes[i, ch].axis("off")

        axes[i, 3].imshow(A, vmin=0, vmax=1)
        axes[i, 3].set_title(titles[3])
        axes[i, 3].axis("off")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved preview:", out_path)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    triplets = find_triplets(extract_dir)

    print("Found triplets:", len(triplets))

    if len(triplets) == 0:
        raise RuntimeError("No triplets found. Please check extracted_one_zip path.")

    all_A = []
    all_masks = []
    all_nuc_masks = []
    all_meta = []

    for fov_idx, tri in enumerate(triplets):
        A_fov = load_triplet(tri)

        patches = make_single_cell_patches_from_fov(
            A_fov=A_fov,
            fov_id=fov_idx,
        )

        print(
            f"FOV {fov_idx + 1}/{len(triplets)}: "
            f"{len(patches)} accepted single-cell patches"
        )

        for p in patches:
            all_A.append(p["A"])
            all_masks.append(p["mask"])
            all_nuc_masks.append(p["nuc_mask"])
            all_meta.append(p["meta"])

            if len(all_A) >= MAX_PATCHES:
                break

        if len(all_A) >= MAX_PATCHES:
            break

    if len(all_A) == 0:
        raise RuntimeError("No single-cell patches accepted. QC may be too strict.")

    A = np.stack(all_A, axis=0).astype(np.float32)
    masks = np.stack(all_masks, axis=0).astype(np.uint8)
    nuc_masks = np.stack(all_nuc_masks, axis=0).astype(np.uint8)

    # shuffle
    idx = np.arange(A.shape[0])
    rng.shuffle(idx)

    A = A[idx]
    masks = masks[idx]
    nuc_masks = nuc_masks[idx]
    all_meta = [all_meta[i] for i in idx]

    out_path = out_dir / f"A_bbbc021_single_cell_{A.shape[0]}cells_128x128x3.npz"

    np.savez_compressed(
        out_path,
        A=A,
        masks=masks,
        nuc_masks=nuc_masks,
        metadata=json.dumps(all_meta),
        channel_names=np.array(["DNA", "beta_tubulin", "F_actin"]),
        patch_size=np.array(PATCH_SIZE, dtype=np.int32),
    )

    print("\nSaved:")
    print(out_path)
    print("A shape:", A.shape)
    print("masks shape:", masks.shape)
    print("nuc_masks shape:", nuc_masks.shape)
    print("A min/max:", float(A.min()), float(A.max()))

    preview_path = result_dir / "bbbc021_single_cell_preview.png"
    save_preview(A[:8], preview_path)


if __name__ == "__main__":
    main()