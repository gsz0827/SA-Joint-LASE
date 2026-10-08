from pathlib import Path
import json
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from main_exp_config_practical_44spots import (
    SEED as MAIN_SEED,
    EVAL_SPLIT as MAIN_EVAL_SPLIT,
    MAX_SAMPLES as MAIN_MAX_SAMPLES,
    OUT_DIR as MAIN_OUT_DIR,
    get_or_create_eval_ids,
    EPS as MAIN_EPS,
)


# ============================================================
# Paths
# ============================================================

bbbc_root = Path(__file__).resolve().parent

data_dir = bbbc_root / "data" / "processed" / "bbbc021_lase_train_ready_44spots"

result_dir = bbbc_root / "results" / "bbbc021_two_step_decode_then_unmix_44spots_practical"
result_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Config
# ============================================================

RANDOM_SEED = MAIN_SEED

IMAGE_H = 128
IMAGE_W = 128
NUM_SPOTS = 44
NUM_BANDS = 16
NUM_DYES = 3

# Use the same split and sample order as 09 for fair comparison.
EVAL_SPLIT = MAIN_EVAL_SPLIT

# For debugging, use 50 first. After checking, set to None for the full test split.
MAX_SAMPLES = MAIN_MAX_SAMPLES

BATCH_SIZE = 1

# Two-step decoding options.
# If True, use the known adjacent-spot sidelobe coefficient from metadata to
# invert the deterministic tri-diagonal coupling before image interpolation.
# This gives the two-step baseline a fair correction for the known LASE optics.
DECOUPLE_SIDELOBE = True

# After background subtraction and de-sidelobe, low-count regions may be negative.
# For image-domain unmixing, negative fluorescence intensity is nonphysical, so clip.
CLIP_XHAT_NONNEG = True

# Image-domain unmixing method.
# First baseline: exact active-set NNLS for 3 dyes.
UNMIX_METHOD = "nnls_active_set"

# Poisson/KL refinement settings for the stronger two-step baseline.
POISSON_REFINE_ITERS = 1000
POISSON_REFINE_LR = 5e-2
POISSON_REFINE_VERBOSE = False

SAVE_FIGURES_FOR_FIRST_N = 5
EPS = MAIN_EPS


def current_main_method_name():
    if UNMIX_METHOD == "nnls_active_set":
        return "two_step_nnls"
    if UNMIX_METHOD == "nnls_init_poisson_kl":
        return "two_step_nnls_init_poisson_kl"
    raise ValueError(f"Unknown UNMIX_METHOD={UNMIX_METHOD}")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# File loading utilities
# ============================================================

def find_latest_metadata_file(data_dir):
    files = sorted(
        data_dir.glob("bbbc021_lase_train_ready_metadata_*cells_44spots.npz"),
        key=lambda p: p.stat().st_mtime,
    )
    if len(files) == 0:
        raise FileNotFoundError(
            f"No bbbc021_lase_train_ready_metadata_*cells_44spots.npz found in {data_dir}"
        )
    return files[-1]


def _as_python_string(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.bytes_):
        return x.tobytes().decode("utf-8")
    return str(x)


def _load_json_from_np_scalar(x):
    if hasattr(x, "item"):
        x = x.item()
    return json.loads(_as_python_string(x))


def waveform_to_column_tensor(Y_cell, gap, num_spots, axial_len):
    """
    Convert one serialized LASE waveform back to column tensor.

    Input:
        Y_cell: [T, L]
    Output:
        cols:   [L, H, N] = [16, 128, 44]
    """
    if Y_cell.ndim != 2:
        raise ValueError(f"Expected Y_cell shape [T, L], got {Y_cell.shape}")

    T, L = Y_cell.shape
    expected_T = num_spots * axial_len + (num_spots - 1) * gap
    if T != expected_T:
        raise ValueError(
            f"Waveform length mismatch: got T={T}, expected {expected_T}. "
            f"num_spots={num_spots}, axial_len={axial_len}, gap={gap}"
        )

    cols = np.zeros((L, axial_len, num_spots), dtype=np.float32)
    offset = 0
    for i in range(num_spots):
        segment = Y_cell[offset:offset + axial_len, :]  # [H, L]
        cols[:, :, i] = segment.T                       # [L, H]
        offset += axial_len
        if i < num_spots - 1:
            offset += gap

    if offset != T:
        raise RuntimeError(f"Internal serialization error: final offset={offset}, T={T}")

    return cols


# ============================================================
# Dataset
# ============================================================

class BBBC021LASETwoStepDataset(Dataset):
    """
    Data for two-step baseline.

    Disk data:
        Y_obs_norm: [N, T, 16]
        A:          [N, 128, 128, 3]

    Returned sample:
        Y_cols:     [16, 128, 44]
        A_eval:     [3, 128, 128]

    A_eval is used only for evaluation and figures.
    """

    def __init__(self, split_name, metadata_path=None):
        super().__init__()

        if split_name not in ["train", "val", "test"]:
            raise ValueError(f"split_name must be train/val/test, got {split_name}")

        if metadata_path is None:
            metadata_path = find_latest_metadata_file(data_dir)

        self.metadata_path = Path(metadata_path)
        print(f"Loading train-ready metadata for split={split_name}:")
        print(self.metadata_path)

        meta = np.load(self.metadata_path, allow_pickle=True)

        self.A_path = Path(str(meta["A_npy_path"].item()))
        self.Y_path = Path(str(meta["Y_obs_norm_npy_path"].item()))

        self.A = np.load(self.A_path, mmap_mode="r")
        self.Y = np.load(self.Y_path, mmap_mode="r")

        self.S = meta["S"].astype(np.float32)                  # [3, 16]
        self.wavelengths = meta["wavelengths"].astype(np.float32)
        self.split = np.array([_as_python_string(x) for x in meta["split"]], dtype=object)
        self.cell_ids = np.array([_as_python_string(x) for x in meta["cell_ids"]], dtype=object)
        self.gap = int(meta["gap"])

        if "channel_names" in meta.files:
            self.channel_names = [_as_python_string(x) for x in meta["channel_names"]]
        else:
            self.channel_names = ["DNA", "beta-tubulin", "F-actin"]

        self.A_shape_meta = tuple(int(x) for x in meta["A_shape"])
        self.Y_shape_meta = tuple(int(x) for x in meta["Y_obs_norm_shape"])
        self.params = _load_json_from_np_scalar(meta["params"])

        self.num_spots = int(self.params["num_spots"])
        self.axial_len = int(self.A_shape_meta[1])
        self.num_bands = int(self.Y_shape_meta[2])

        expected_T = self.num_spots * self.axial_len + (self.num_spots - 1) * self.gap

        if self.A.shape != self.A_shape_meta:
            raise ValueError(f"A shape mismatch: file {self.A.shape}, metadata {self.A_shape_meta}")
        if self.Y.shape != self.Y_shape_meta:
            raise ValueError(f"Y shape mismatch: file {self.Y.shape}, metadata {self.Y_shape_meta}")
        if self.A.shape[0] != self.Y.shape[0]:
            raise ValueError(f"N mismatch: A has {self.A.shape[0]}, Y has {self.Y.shape[0]}")
        if self.A.shape[1:] != (IMAGE_H, IMAGE_W, NUM_DYES):
            raise ValueError(f"A shape should end with {(IMAGE_H, IMAGE_W, NUM_DYES)}, got {self.A.shape[1:]}")
        if self.Y.shape[1] != expected_T:
            raise ValueError(f"Y length mismatch: got T={self.Y.shape[1]}, expected {expected_T}")
        if self.num_spots != NUM_SPOTS:
            raise ValueError(f"NUM_SPOTS mismatch: config {NUM_SPOTS}, metadata {self.num_spots}")
        if self.num_bands != NUM_BANDS:
            raise ValueError(f"NUM_BANDS mismatch: config {NUM_BANDS}, metadata {self.num_bands}")

        mask = self.split == split_name
        self.indices = np.where(mask)[0]
        if len(self.indices) == 0:
            raise ValueError(f"No samples found for split={split_name}")

        print(f"{split_name} samples:", len(self.indices))
        print("A mmap shape:", self.A.shape)
        print("Y mmap shape:", self.Y.shape)
        print("S shape:", self.S.shape)
        print("S row sums:", np.sum(self.S, axis=1))
        print("num_spots:", self.num_spots)
        print("axial_len:", self.axial_len)
        print("gap:", self.gap)
        print("expected T:", expected_T)
        print("")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])

        Y_cell = np.asarray(self.Y[real_idx], dtype=np.float32)  # [T, 16]
        A_cell = np.asarray(self.A[real_idx], dtype=np.float32)  # [128, 128, 3]

        Y_cols = waveform_to_column_tensor(
            Y_cell=Y_cell,
            gap=self.gap,
            num_spots=self.num_spots,
            axial_len=self.axial_len,
        )                                                        # [16, 128, 44]

        A_eval = A_cell.transpose(2, 0, 1).astype(np.float32)     # [3, 128, 128]
        A_eval = np.clip(A_eval, 0.0, 1.0)

        return (
            torch.from_numpy(Y_cols).contiguous(),
            torch.from_numpy(A_eval).contiguous(),
            real_idx,
        )


# ============================================================
# Step 1: LASE waveform/column decoding into X_hat
# ============================================================

def build_sidelobe_matrix(num_spots, alpha):
    """
    Build the deterministic adjacent-spot coupling matrix C.

    08/09 use:
        observed_i = (1 - alpha * neighbor_count_i) * clean_i
                     + alpha * clean_{i-1} + alpha * clean_{i+1}

    Boundary spots have one neighbor; interior spots have two neighbors.
    """
    C = np.zeros((num_spots, num_spots), dtype=np.float32)
    for i in range(num_spots):
        if i > 0:
            C[i, i - 1] = alpha
        if i < num_spots - 1:
            C[i, i + 1] = alpha
        neighbor_count = (1 if i > 0 else 0) + (1 if i < num_spots - 1 else 0)
        C[i, i] = 1.0 - alpha * neighbor_count
    return C


def decouple_sidelobe_columns(Y_cols, params):
    """
    Invert known adjacent-spot coupling along the spot dimension.

    Y_cols: [L, H, N]
    """
    alpha = float(params.get("sidelobe_alpha", 0.0))
    if alpha <= 0:
        return Y_cols.astype(np.float32)

    L, H, N = Y_cols.shape
    C = build_sidelobe_matrix(N, alpha=alpha)
    C_inv = np.linalg.inv(C).astype(np.float32)

    # For each band and axial row, clean_spots = C_inv @ observed_spots.
    clean_cols = np.einsum("ij,lhj->lhi", C_inv, Y_cols, optimize=True)
    return clean_cols.astype(np.float32)


def interpolate_columns_to_image(Y_cols, image_w=IMAGE_W):
    """
    Convert sampled LASE columns [L, H, N] to image-domain X_hat [H, W, L]
    by linear interpolation along the spot/x dimension.
    """
    L, H, N = Y_cols.shape

    old_x = np.linspace(0.0, image_w - 1.0, N, dtype=np.float32)
    new_x = np.arange(image_w, dtype=np.float32)

    X_lhw = np.empty((L, H, image_w), dtype=np.float32)
    for l in range(L):
        for h in range(H):
            X_lhw[l, h, :] = np.interp(new_x, old_x, Y_cols[l, h, :]).astype(np.float32)

    X_hat = np.transpose(X_lhw, (1, 2, 0))  # [H, W, L]
    return X_hat.astype(np.float32)


def decode_xhat_from_ycols(Y_cols, params):
    """
    Traditional two-step Step 1:
        raw waveform columns -> decoded hyperspectral image X_hat.

    This does not use A_true.
    """
    Y_cols_np = np.asarray(Y_cols, dtype=np.float32)

    if DECOUPLE_SIDELOBE:
        Y_cols_np = decouple_sidelobe_columns(Y_cols_np, params=params)

    X_hat = interpolate_columns_to_image(Y_cols_np, image_w=IMAGE_W)

    if CLIP_XHAT_NONNEG:
        X_hat = np.clip(X_hat, 0.0, None)

    return X_hat.astype(np.float32)


# ============================================================
# Step 2: Image-domain spectral unmixing
# ============================================================

def unmix_nnls_active_set_3d(X_hat, S):
    """
    Exact active-set NNLS for 3 dyes and many pixels.

    For each pixel x in R^16, solve:
        min_{a >= 0} || x - S.T @ a ||_2^2

    X_hat: [H, W, L]
    S:     [D, L] = [3, 16]

    Returns:
        A_hat: [D, H, W]
    """
    X_hat = np.asarray(X_hat, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)

    H, W, L = X_hat.shape
    D, Ls = S.shape
    if D != 3:
        raise ValueError("This active-set implementation assumes exactly 3 dyes.")
    if L != Ls:
        raise ValueError(f"Band mismatch: X has L={L}, S has L={Ls}")

    X = X_hat.reshape(-1, L).astype(np.float32)        # [P, L]
    M = S.T.astype(np.float32)                         # [L, D]
    P = X.shape[0]

    best_A = np.zeros((P, D), dtype=np.float32)
    best_res = np.sum(X * X, axis=1).astype(np.float32) # zero solution residual

    # Enumerate all non-empty active sets: 001, 010, ..., 111.
    for mask in range(1, 1 << D):
        active = [d for d in range(D) if (mask >> d) & 1]
        Mp = M[:, active]                              # [L, k]
        pinv = np.linalg.pinv(Mp).astype(np.float32)   # [k, L]
        A_sub = X @ pinv.T                             # [P, k]

        valid = np.all(A_sub >= -1e-7, axis=1)
        A_sub_clip = np.maximum(A_sub, 0.0)

        X_pred = A_sub_clip @ Mp.T                     # [P, L]
        res = np.sum((X - X_pred) ** 2, axis=1).astype(np.float32)
        res[~valid] = np.inf

        improve = res < best_res
        if np.any(improve):
            best_res[improve] = res[improve]
            best_A[improve, :] = 0.0
            best_A[np.ix_(improve, active)] = A_sub_clip[improve]

    A_hat = best_A.reshape(H, W, D).transpose(2, 0, 1) # [D, H, W]
    return A_hat.astype(np.float32)


def softplus_inverse_tensor(y, eps=1e-6):
    """Stable inverse of softplus for positive initialization."""
    y = torch.clamp(y, min=eps)
    return torch.where(y > 20.0, y, torch.log(torch.expm1(y)))


def poisson_refine_from_nnls(
    X_hat,
    S,
    A_init,
    num_iters=POISSON_REFINE_ITERS,
    lr=POISSON_REFINE_LR,
    device=None,
    verbose=POISSON_REFINE_VERBOSE,
):
    """
    Stronger two-step baseline:
        decoded image X_hat -> NNLS initialization -> image-domain Poisson/KL refinement.

    X_hat:  [H, W, L]
    S:      [D, L]
    A_init: [D, H, W] from NNLS, not from ground truth
    return: [D, H, W]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_hat = np.asarray(X_hat, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)
    A_init = np.asarray(A_init, dtype=np.float32)

    H, W, L = X_hat.shape
    D, Ls = S.shape
    if L != Ls:
        raise ValueError(f"Band mismatch: X has L={L}, S has L={Ls}")
    if A_init.shape != (D, H, W):
        raise ValueError(f"A_init shape {A_init.shape}, expected {(D, H, W)}")

    X = torch.tensor(X_hat, dtype=torch.float32, device=device)
    X = torch.clamp(X, min=0.0)
    S_t = torch.tensor(S, dtype=torch.float32, device=device)

    A0 = torch.tensor(A_init, dtype=torch.float32, device=device)
    A0 = torch.clamp(A0, min=1e-6)

    # Key point: initialize by NNLS, not random initialization.
    A_raw = softplus_inverse_tensor(A0).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([A_raw], lr=lr)

    for it in range(1, num_iters + 1):
        optimizer.zero_grad(set_to_none=True)

        A = F.softplus(A_raw)
        X_pred = torch.einsum("dhw,dl->hwl", A, S_t)
        X_pred = torch.clamp(X_pred, min=EPS)

        # Image-domain Poisson negative log-likelihood / generalized KL,
        # up to constants independent of A.
        loss = torch.mean(X_pred - X * torch.log(X_pred))

        loss.backward()
        optimizer.step()

        if verbose and (it == 1 or it == num_iters or it % 100 == 0):
            print(f"[Poisson refine] iter={it:04d}/{num_iters}, loss={loss.item():.6e}")

    A_hat = F.softplus(A_raw).detach().cpu().numpy().astype(np.float32)
    return A_hat


def two_step_decode_then_unmix(Y_cols, S, params):
    """
    Full traditional two-step baseline:
        Y -> X_hat -> A_hat
    """
    X_hat = decode_xhat_from_ycols(Y_cols, params=params)

    A_nnls = unmix_nnls_active_set_3d(X_hat=X_hat, S=S)

    if UNMIX_METHOD == "nnls_active_set":
        A_hat = A_nnls
    elif UNMIX_METHOD == "nnls_init_poisson_kl":
        A_hat = poisson_refine_from_nnls(
            X_hat=X_hat,
            S=S,
            A_init=A_nnls,
        )
    else:
        raise ValueError(f"Unknown UNMIX_METHOD={UNMIX_METHOD}")

    return A_hat, X_hat


# ============================================================
# Metrics and visualization
# ============================================================

def compute_eval_metrics_np(A_hat, A_true):
    """
    A_hat, A_true: [D, H, W]
    """
    A_hat = np.asarray(A_hat, dtype=np.float32)
    A_true = np.asarray(A_true, dtype=np.float32)

    diff = A_hat - A_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse + EPS))
    nrmse = float(rmse / (np.sqrt(np.mean(A_true ** 2)) + EPS))

    # Pearson is computed per dye and then averaged, matching main_exp_metrics.py.
    pearsons = []
    for d in range(A_true.shape[0]):
        h = A_hat[d].reshape(-1)
        t = A_true[d].reshape(-1)
        h0 = h - np.mean(h)
        t0 = t - np.mean(t)
        denom = np.sqrt(np.sum(h0 ** 2) * np.sum(t0 ** 2)) + EPS
        if denom > EPS:
            pearsons.append(float(np.sum(h0 * t0) / denom))
    pearson = float(np.nanmean(pearsons)) if len(pearsons) else float("nan")

    per_dye_mse = np.mean(diff ** 2, axis=(1, 2))
    per_dye_mae = np.mean(np.abs(diff), axis=(1, 2))
    per_dye_rmse = np.sqrt(per_dye_mse + EPS)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "pearson": pearson,
        "per_dye_mse": per_dye_mse.astype(float).tolist(),
        "per_dye_mae": per_dye_mae.astype(float).tolist(),
        "per_dye_rmse": per_dye_rmse.astype(float).tolist(),
    }


def _safe_vmax(img, percentile=99.5):
    vmax = np.percentile(img, percentile)
    if vmax <= EPS:
        vmax = float(np.max(img)) + EPS
    return vmax


def save_inversion_figure(
    A_hat,
    A_true,
    out_path,
    channel_names,
    title=None,
):
    if title is None:
        title = "Two-step abundance inversion"

    A_hat = np.asarray(A_hat, dtype=np.float32)
    A_true = np.asarray(A_true, dtype=np.float32)
    err = np.abs(A_hat - A_true)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)

    for ch in range(NUM_DYES):
        name = channel_names[ch] if ch < len(channel_names) else f"ch{ch}"
        vmax = max(_safe_vmax(A_true[ch]), _safe_vmax(A_hat[ch]), 1e-3)

        axes[0, ch].imshow(A_true[ch], cmap="gray", vmin=0, vmax=vmax)
        axes[0, ch].set_title(f"Eval true {name}")
        axes[0, ch].axis("off")

        axes[1, ch].imshow(A_hat[ch], cmap="gray", vmin=0, vmax=vmax)
        axes[1, ch].set_title(f"Estimated {name}")
        axes[1, ch].axis("off")

        axes[2, ch].imshow(err[ch], cmap="magma")
        axes[2, ch].set_title(f"Abs error {name}")
        axes[2, ch].axis("off")

    fig.suptitle(title, fontsize=15)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved inversion figure:")
    print(out_path)


def save_x_decode_figure(
    X_hat,
    A_true,
    S,
    wavelengths,
    out_path,
    bands=(0, 5, 10, 15),
    title="Two-step Step 1 X decode diagnostic",
):
    """
    Diagnostic figure for two-step Step 1.

    X_ref is computed from A_true and S for evaluation/visual reference only.
    X_hat is decoded from LASE waveform columns and does not use A_true.
    """
    X_hat = np.asarray(X_hat, dtype=np.float32)       # [H, W, L]
    A_true = np.asarray(A_true, dtype=np.float32)     # [D, H, W]
    S = np.asarray(S, dtype=np.float32)               # [D, L]

    # Reference spectral image from ground-truth abundance.
    # This is only for visualization/evaluation, not used by the method.
    X_ref = np.einsum("dhw,dl->hwl", A_true, S).astype(np.float32)

    fig, axes = plt.subplots(
        3,
        len(bands),
        figsize=(4 * len(bands), 9),
        constrained_layout=True,
    )

    for j, b in enumerate(bands):
        ref = X_ref[:, :, b]
        pred = X_hat[:, :, b]
        err = np.abs(pred - ref)

        vmax = max(_safe_vmax(ref), _safe_vmax(pred), 1e-6)

        band_title = (
            f"B{b + 1} {wavelengths[b]:.0f} nm"
            if b < len(wavelengths)
            else f"B{b + 1}"
        )

        axes[0, j].imshow(ref, cmap="gray", vmin=0, vmax=vmax)
        axes[0, j].set_title(f"Reference A_true @ S {band_title}")
        axes[0, j].axis("off")

        axes[1, j].imshow(pred, cmap="gray", vmin=0, vmax=vmax)
        axes[1, j].set_title(f"Decoded X_hat {band_title}")
        axes[1, j].axis("off")

        axes[2, j].imshow(err, cmap="magma")
        axes[2, j].set_title(f"Abs error {band_title}")
        axes[2, j].axis("off")

    fig.suptitle(title, fontsize=15)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved X decode figure:")
    print(out_path)


# ============================================================
# Main
# ============================================================

def main():
    global UNMIX_METHOD

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unmix",
        type=str,
        default="nnls",
        choices=["nnls", "poisson"],
        help="nnls = Two-step + NNLS; poisson = Two-step + NNLS-init Poisson/KL",
    )
    args = parser.parse_args()

    if args.unmix == "nnls":
        UNMIX_METHOD = "nnls_active_set"
    elif args.unmix == "poisson":
        UNMIX_METHOD = "nnls_init_poisson_kl"
    else:
        raise ValueError(f"Unknown --unmix={args.unmix}")

    method_file_name = current_main_method_name()
    method_display_name = {
        "two_step_nnls": "Two-step + NNLS",
        "two_step_nnls_init_poisson_kl": "Two-step + NNLS-init Poisson/KL",
    }[method_file_name]

    set_seed(RANDOM_SEED)

    metadata_path = find_latest_metadata_file(data_dir)
    print("Using train-ready metadata:")
    print(metadata_path)
    print("")

    dataset = BBBC021LASETwoStepDataset(EVAL_SPLIT, metadata_path=metadata_path)

    n_total = len(dataset)

    # Use exactly the same local sample indices as 09.
    # The dataset then returns the corresponding original real_idx for logging.
    eval_local_ids = get_or_create_eval_ids(list(range(n_total)))
    n_to_process = len(eval_local_ids)
    if n_to_process <= 0:
        raise ValueError("n_to_process must be positive.")

    print("Two-step split:", EVAL_SPLIT)
    print("Samples in split:", n_total)
    print("Samples to process:", n_to_process)
    print("Eval local ids first 10:", eval_local_ids[:10])
    print("Method file name:", method_file_name)
    print("Method display name:", method_display_name)
    print("Batch size:", BATCH_SIZE)
    print("UNMIX_METHOD:", UNMIX_METHOD)
    print("DECOUPLE_SIDELOBE:", DECOUPLE_SIDELOBE)
    print("CLIP_XHAT_NONNEG:", CLIP_XHAT_NONNEG)
    print("")

    eval_dataset = Subset(dataset, eval_local_ids)

    loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    out_dir = result_dir / f"{EVAL_SPLIT}_two_step_{UNMIX_METHOD}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    A_hat_path = out_dir / f"A_hat_two_step_{UNMIX_METHOD}_{EVAL_SPLIT}_{n_to_process}samples.npy"
    A_hat_map = np.lib.format.open_memmap(
        A_hat_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_to_process, NUM_DYES, IMAGE_H, IMAGE_W),
    )

    real_indices = np.zeros((n_to_process,), dtype=np.int64)
    metrics = []

    cursor = 0

    for batch_idx, batch in enumerate(loader):
        if cursor >= n_to_process:
            break

        Y_cols_batch, A_true_batch, real_idx_batch = batch

        # This script is designed for BATCH_SIZE=1 first, but also handles B>1 sequentially.
        batch_size = int(Y_cols_batch.shape[0])
        remaining = n_to_process - cursor
        if batch_size > remaining:
            Y_cols_batch = Y_cols_batch[:remaining]
            A_true_batch = A_true_batch[:remaining]
            real_idx_batch = real_idx_batch[:remaining]
            batch_size = remaining

        print("\n" + "=" * 80)
        print(
            f"Batch {batch_idx} | output rows {cursor}:{cursor + batch_size} | "
            f"real_idx={real_idx_batch.detach().cpu().numpy().tolist()}"
        )
        print("=" * 80)

        for j in range(batch_size):
            row = cursor + j

            Y_cols = Y_cols_batch[j].detach().cpu().numpy().astype(np.float32)       # [16, 128, 44]
            A_true = A_true_batch[j].detach().cpu().numpy().astype(np.float32)       # [3, 128, 128]
            real_idx = int(real_idx_batch[j].item())

            A_hat, X_hat = two_step_decode_then_unmix(
                Y_cols=Y_cols,
                S=dataset.S,
                params=dataset.params,
            )

            # Since A_true is normalized to [0, 1], clipping A_hat only for evaluation storage
            # is NOT applied. NNLS is nonnegative and may exceed 1; metrics reflect that.
            A_hat_map[row] = A_hat
            A_hat_map.flush()
            real_indices[row] = real_idx

            m = compute_eval_metrics_np(A_hat=A_hat, A_true=A_true)

            item_metrics = {
                "output_row": int(row),
                "dataset_split": EVAL_SPLIT,
                "real_idx": int(real_idx),
                "mse_vs_eval_A": float(m["mse"]),
                "mae_vs_eval_A": float(m["mae"]),
                "rmse_vs_eval_A": float(m["rmse"]),
                "nrmse_vs_eval_A": float(m["nrmse"]),
                "pearson_vs_eval_A": float(m["pearson"]),
                "per_dye_mae_vs_eval_A": [float(x) for x in m["per_dye_mae"]],
                "per_dye_rmse_vs_eval_A": [float(x) for x in m["per_dye_rmse"]],
                "A_min": float(np.min(A_hat)),
                "A_max": float(np.max(A_hat)),
                "A_mean": float(np.mean(A_hat)),
            }
            metrics.append(item_metrics)

            print(
                f"row={row} real_idx={real_idx} | "
                f"MSE={m['mse']:.6e} | "
                f"MAE={m['mae']:.6e} | "
                f"NRMSE={m['nrmse']:.6e} | "
                f"Pearson={m['pearson']:.4f} | "
                f"A_max={np.max(A_hat):.4f}"
            )

            if row < SAVE_FIGURES_FOR_FIRST_N:
                inv_fig_path = fig_dir / f"{method_file_name}_inversion_row_{row:04d}_realidx_{real_idx}.png"
                save_inversion_figure(
                    A_hat=A_hat,
                    A_true=A_true,
                    out_path=inv_fig_path,
                    channel_names=dataset.channel_names,
                    title=f"{method_display_name} abundance inversion",
                )

                x_fig_path = fig_dir / f"{method_file_name}_x_decode_row_{row:04d}_realidx_{real_idx}.png"
                save_x_decode_figure(
                    X_hat=X_hat,
                    A_true=A_true,
                    S=dataset.S,
                    wavelengths=dataset.wavelengths,
                    out_path=x_fig_path,
                    title=f"Shared Step 1 X decode diagnostic ({method_display_name})",
                )

        cursor += batch_size

    A_hat_map.flush()

    real_indices_path = out_dir / f"real_indices_two_step_{UNMIX_METHOD}_{EVAL_SPLIT}_{n_to_process}samples.npy"
    np.save(real_indices_path, real_indices)

    metrics_path = out_dir / f"metrics_two_step_{UNMIX_METHOD}_{EVAL_SPLIT}_{n_to_process}samples.json"

    mse_values = [m["mse_vs_eval_A"] for m in metrics]
    mae_values = [m["mae_vs_eval_A"] for m in metrics]
    rmse_values = [m["rmse_vs_eval_A"] for m in metrics]
    nrmse_values = [m["nrmse_vs_eval_A"] for m in metrics]
    pearson_values = [m["pearson_vs_eval_A"] for m in metrics]
    amax_values = [m["A_max"] for m in metrics]

    per_dye_mae = np.array([m["per_dye_mae_vs_eval_A"] for m in metrics], dtype=np.float32)
    per_dye_rmse = np.array([m["per_dye_rmse_vs_eval_A"] for m in metrics], dtype=np.float32)

    summary = {
        "script": "10_two_step_bbbc021_decode_then_unmix.py",
        "method": method_file_name,
        "metadata_path": str(metadata_path),
        "split": EVAL_SPLIT,
        "n_total_in_split": int(n_total),
        "n_processed": int(n_to_process),
        "A_hat_path": str(A_hat_path),
        "real_indices_path": str(real_indices_path),
        "result_dir": str(out_dir),
        "config": {
            "random_seed": RANDOM_SEED,
            "image_h": IMAGE_H,
            "image_w": IMAGE_W,
            "num_spots": NUM_SPOTS,
            "num_bands": NUM_BANDS,
            "num_dyes": NUM_DYES,
            "batch_size": BATCH_SIZE,
            "max_samples": MAX_SAMPLES,
            "unmix_method": UNMIX_METHOD,
            "decouple_sidelobe": DECOUPLE_SIDELOBE,
            "clip_xhat_nonneg": CLIP_XHAT_NONNEG,
        },
        "params_from_metadata": dataset.params,
        "eval_metrics_note": "A_true is used only for evaluation and figures, not for decoding or unmixing.",
        "mean_mse_vs_eval_A": float(np.mean(mse_values)) if mse_values else None,
        "mean_mae_vs_eval_A": float(np.mean(mae_values)) if mae_values else None,
        "mean_rmse_vs_eval_A": float(np.mean(rmse_values)) if rmse_values else None,
        "mean_nrmse_vs_eval_A": float(np.mean(nrmse_values)) if nrmse_values else None,
        "mean_pearson_vs_eval_A": float(np.mean(pearson_values)) if pearson_values else None,
        "mean_A_max": float(np.mean(amax_values)) if amax_values else None,
        "max_A_max": float(np.max(amax_values)) if amax_values else None,
        "mean_per_dye_mae_vs_eval_A": per_dye_mae.mean(axis=0).astype(float).tolist() if len(metrics) else None,
        "mean_per_dye_rmse_vs_eval_A": per_dye_rmse.mean(axis=0).astype(float).tolist() if len(metrics) else None,
        "items": metrics,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    main_npz_path = Path(MAIN_OUT_DIR) / f"{method_file_name}.npz"
    np.savez(
        main_npz_path,
        sample_ids=real_indices.copy(),
        mae=np.array(mae_values, dtype=np.float32),
        rmse=np.array(rmse_values, dtype=np.float32),
        nrmse=np.array(nrmse_values, dtype=np.float32),
        pearson=np.array(pearson_values, dtype=np.float32),
    )

    print("\n" + "=" * 80)
    print("Finished two-step decode-then-unmix baseline.")
    print("Saved A_hat:")
    print(A_hat_path)
    print("Saved real indices:")
    print(real_indices_path)
    print("Saved metrics:")
    print(metrics_path)
    print("Saved main experiment npz:")
    print(main_npz_path)
    print("Saved figures:")
    print(fig_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()