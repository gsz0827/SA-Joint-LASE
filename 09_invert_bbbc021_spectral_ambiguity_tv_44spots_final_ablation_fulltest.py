from pathlib import Path
import json
import random
import math
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

# Pure direct joint inversion output folder.
# This version starts from a constant abundance field and optimizes A only
# through the raw LASE waveform Poisson data-consistency term and priors.
result_dir = bbbc_root / "results" / "bbbc021_channel_competition_sparse_44spots"
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

# Which split to invert.
# Usually use "test" first. You can change this to "train" or "val".
INVERT_SPLIT = MAIN_EVAL_SPLIT

# None means invert the whole split.
# For the final full-test ablation, keep this as None.
MAX_SAMPLES = None

# This script supports batch-wise independent optimization:
# Z has shape [B, 3, 128, 128].
# If CUDA memory is limited, keep this at 1.
INVERSION_BATCH_SIZE = 1

NUM_ITERS = 1000
LEARNING_RATE = 2e-2

# ============================================================
# Ablation settings
# ============================================================

# Default full model.
# This is NOT the old global L1 sparse term A.mean().
# It is a per-pixel cross-channel competition sparsity:
# the same spatial location should usually be dominated by only a small number of dyes.
LAMBDA_CHANNEL_SPARSE = 0.0
CHANNEL_SPARSE_TAU = 0.02
LAMBDA_TV = 1e-4

# Adaptive spectral-correlation reweighted sparsity.
# This is a safer replacement for a strong fixed channel-competition prior:
# weights are detached, clipped, updated only every few iterations, and use
# spectral correlation thresholding so that only similar dyes compete strongly.
LAMBDA_ADAPTIVE_SPARSE = 1e-4
ADAPTIVE_EPS = 1e-3
ADAPTIVE_BETA = 10.0
ADAPTIVE_RHO_THRESHOLD = 0.5
ADAPTIVE_WEIGHT_UPDATE_EVERY = 20
ADAPTIVE_W_MIN = 0.0
ADAPTIVE_W_MAX = 100.0

# Spectral-ambiguity-aware TV regularization.
# Instead of forcing mutually exclusive dye channels, this stabilizes only
# the dye-combination directions that are poorly identifiable from highly
# correlated spectra.
LAMBDA_AMBIGUITY_TV = 2e-4
AMBIGUITY_ETA = 1e-3
AMBIGUITY_POWER = 1.0
AMBIGUITY_W_MAX = 20.0

# Soft upper-bound penalty.
# The simulated ground-truth abundance maps were normalized to [0, 1].
# This term discourages local overshoots A > 1 without imposing a hard sigmoid bound.
A_UPPER = 1.0
LAMBDA_UPPER = 1e-3

# Ablation variants.
# Keep upper-bound fixed for all variants, because it is mainly a numerical
# and normalization-range constraint rather than the scientific prior to test.
ABLATION_CONFIGS = {
    # Final selected model.
    # This is the model to report as the proposed method:
    # Poisson/KL + standard TV + spectral-ambiguity-direction TV + upper-bound.
    "final": {
        "method_name": "joint_spectral_ambiguity_tv_final",
        "sparse_mode": "none",
        "lambda_channel_sparse": 0.0,
        "channel_sparse_tau": CHANNEL_SPARSE_TAU,
        "lambda_adaptive_sparse": 0.0,
        "lambda_ambiguity_tv": 2e-4,
        "lambda_tv": 1e-4,
        "lambda_upper": 1e-3,
        "description": "Final: Poisson/KL + standard TV + spectral-ambiguity-direction TV + upper-bound",
    },

    # Core ablations for the final method.
    "poisson_only": {
        "method_name": "ablation_poisson_upper_only",
        "sparse_mode": "none",
        "lambda_channel_sparse": 0.0,
        "channel_sparse_tau": CHANNEL_SPARSE_TAU,
        "lambda_adaptive_sparse": 0.0,
        "lambda_ambiguity_tv": 0.0,
        "lambda_tv": 0.0,
        "lambda_upper": 1e-3,
        "description": "Ablation: Poisson/KL + upper-bound only",
    },
    "no_ambiguity_tv": {
        "method_name": "ablation_no_ambiguity_tv",
        "sparse_mode": "none",
        "lambda_channel_sparse": 0.0,
        "channel_sparse_tau": CHANNEL_SPARSE_TAU,
        "lambda_adaptive_sparse": 0.0,
        "lambda_ambiguity_tv": 0.0,
        "lambda_tv": 1e-4,
        "lambda_upper": 1e-3,
        "description": "Ablation: Poisson/KL + standard TV + upper-bound, without spectral-ambiguity TV",
    },
    "ambiguity_tv_only": {
        "method_name": "ablation_ambiguity_tv_only",
        "sparse_mode": "none",
        "lambda_channel_sparse": 0.0,
        "channel_sparse_tau": CHANNEL_SPARSE_TAU,
        "lambda_adaptive_sparse": 0.0,
        "lambda_ambiguity_tv": 2e-4,
        "lambda_tv": 0.0,
        "lambda_upper": 1e-3,
        "description": "Ablation: Poisson/KL + spectral-ambiguity-direction TV + upper-bound, without standard TV",
    },
    "no_upper": {
        "method_name": "ablation_no_upper",
        "sparse_mode": "none",
        "lambda_channel_sparse": 0.0,
        "channel_sparse_tau": CHANNEL_SPARSE_TAU,
        "lambda_adaptive_sparse": 0.0,
        "lambda_ambiguity_tv": 2e-4,
        "lambda_tv": 1e-4,
        "lambda_upper": 0.0,
        "description": "Ablation: Poisson/KL + standard TV + spectral-ambiguity-direction TV, without upper-bound penalty",
    },

    # Legacy sparse-prior comparisons. These are kept for negative/diagnostic ablation.
    "fixed_sparse": {
        "method_name": "ablation_fixed_channel_sparse",
        "sparse_mode": "fixed",
        "lambda_channel_sparse": 1e-3,
        "channel_sparse_tau": CHANNEL_SPARSE_TAU,
        "lambda_adaptive_sparse": 0.0,
        "lambda_ambiguity_tv": 0.0,
        "lambda_tv": 1e-4,
        "lambda_upper": 1e-3,
        "description": "Legacy ablation: Poisson/KL + fixed channel-competition sparse + standard TV + upper-bound",
    },
    "adaptive_sparse": {
        "method_name": "ablation_adaptive_spectral_sparse",
        "sparse_mode": "adaptive",
        "lambda_channel_sparse": 0.0,
        "channel_sparse_tau": CHANNEL_SPARSE_TAU,
        "lambda_adaptive_sparse": LAMBDA_ADAPTIVE_SPARSE,
        "lambda_ambiguity_tv": 0.0,
        "lambda_tv": 1e-4,
        "lambda_upper": 1e-3,
        "description": "Legacy ablation: Poisson/KL + adaptive spectral-correlation reweighted sparse + standard TV + upper-bound",
    },

    # Backward-compatible aliases for older runners/commands.
    "ambiguity_tv": {
        "method_name": "joint_spectral_ambiguity_tv",
        "sparse_mode": "none",
        "lambda_channel_sparse": 0.0,
        "channel_sparse_tau": CHANNEL_SPARSE_TAU,
        "lambda_adaptive_sparse": 0.0,
        "lambda_ambiguity_tv": 2e-4,
        "lambda_tv": 1e-4,
        "lambda_upper": 1e-3,
        "description": "Alias of final-style model: Poisson/KL + standard TV + spectral-ambiguity-direction TV + upper-bound",
    },
    "no_sparse": {
        "method_name": "joint_no_sparse",
        "sparse_mode": "none",
        "lambda_channel_sparse": 0.0,
        "channel_sparse_tau": CHANNEL_SPARSE_TAU,
        "lambda_adaptive_sparse": 0.0,
        "lambda_ambiguity_tv": 0.0,
        "lambda_tv": 1e-4,
        "lambda_upper": 1e-3,
        "description": "Alias: Poisson/KL + standard TV + upper-bound, without spectral-ambiguity TV",
    },
    "full": {
        "method_name": "joint_channel_sparse_full",
        "sparse_mode": "fixed",
        "lambda_channel_sparse": 1e-3,
        "channel_sparse_tau": CHANNEL_SPARSE_TAU,
        "lambda_adaptive_sparse": 0.0,
        "lambda_ambiguity_tv": 0.0,
        "lambda_tv": 1e-4,
        "lambda_upper": 1e-3,
        "description": "Alias: legacy fixed channel-competition sparse model",
    },
    "adaptive": {
        "method_name": "joint_adaptive_spectral_sparse",
        "sparse_mode": "adaptive",
        "lambda_channel_sparse": 0.0,
        "channel_sparse_tau": CHANNEL_SPARSE_TAU,
        "lambda_adaptive_sparse": LAMBDA_ADAPTIVE_SPARSE,
        "lambda_ambiguity_tv": 0.0,
        "lambda_tv": 1e-4,
        "lambda_upper": 1e-3,
        "description": "Alias: legacy adaptive spectral-correlation sparse model",
    },
    "no_tv": {
        "method_name": "joint_no_tv",
        "sparse_mode": "fixed",
        "lambda_channel_sparse": 1e-3,
        "channel_sparse_tau": CHANNEL_SPARSE_TAU,
        "lambda_adaptive_sparse": 0.0,
        "lambda_ambiguity_tv": 0.0,
        "lambda_tv": 0.0,
        "lambda_upper": 1e-3,
        "description": "Legacy alias: fixed channel-competition sparse + upper-bound, without standard TV",
    },
}
# Gradient clipping helps avoid unstable early updates.
GRAD_CLIP_NORM = 10.0

# Initialization: constant abundance field only.
# No image-domain decoding or external abundance initializer is used
# anywhere in this script.
INIT_A_CONSTANT = 1e-3
INIT_A_FLOOR = 1e-5

PRINT_EVERY = 100
SAVE_FIGURES_FOR_FIRST_N = 5

EPS = MAIN_EPS

# Match scipy.ndimage.gaussian_filter default truncate=4.0 used in 08.
GAUSSIAN_TRUNCATE = 4.0


# ============================================================
# Reproducibility
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure joint Poisson LASE abundance inversion with ablation variants."
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="final",
        choices=list(ABLATION_CONFIGS.keys()),
        help="Ablation variant to run.",
    )

    parser.add_argument(
        "--lambda-channel-sparse",
        type=float,
        default=None,
        help="Override lambda_channel_sparse in the selected ablation config.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override MAX_SAMPLES. Omit this argument to run the whole split when MAX_SAMPLES=None.",
    )

    parser.add_argument(
        "--num-iters",
        type=int,
        default=None,
        help="Override NUM_ITERS. Use 300 for quick channel-sparse sweep.",
    )

    parser.add_argument(
        "--sparse-mode",
        type=str,
        default="config",
        choices=["config", "fixed", "adaptive", "none"],
        help="Sparse prior mode. 'config' uses the selected variant setting.",
    )

    parser.add_argument(
        "--lambda-adaptive-sparse",
        type=float,
        default=None,
        help="Override lambda for adaptive spectral-correlation reweighted sparsity.",
    )

    parser.add_argument(
        "--adaptive-beta",
        type=float,
        default=ADAPTIVE_BETA,
        help="Strength of spectral-correlation competition in adaptive sparse weights.",
    )

    parser.add_argument(
        "--adaptive-eps",
        type=float,
        default=ADAPTIVE_EPS,
        help="Epsilon in 1 / (A + eps) for adaptive reweighting.",
    )

    parser.add_argument(
        "--adaptive-rho-threshold",
        type=float,
        default=ADAPTIVE_RHO_THRESHOLD,
        help="Only spectral cosine similarities above this threshold contribute to competition.",
    )

    parser.add_argument(
        "--adaptive-weight-update-every",
        type=int,
        default=ADAPTIVE_WEIGHT_UPDATE_EVERY,
        help="Update detached adaptive sparse weights every N iterations.",
    )

    parser.add_argument(
        "--adaptive-w-max",
        type=float,
        default=ADAPTIVE_W_MAX,
        help="Maximum clipped adaptive sparse weight.",
    )

    parser.add_argument(
        "--adaptive-w-min",
        type=float,
        default=ADAPTIVE_W_MIN,
        help="Minimum clipped adaptive sparse weight.",
    )

    parser.add_argument(
        "--lambda-ambiguity-tv",
        type=float,
        default=None,
        help="Weight for spectral-ambiguity-direction TV regularization.",
    )

    parser.add_argument(
        "--ambiguity-eta",
        type=float,
        default=AMBIGUITY_ETA,
        help="Small stabilizer for inverse-eigenvalue ambiguity weights.",
    )

    parser.add_argument(
        "--ambiguity-power",
        type=float,
        default=AMBIGUITY_POWER,
        help="Power applied to inverse-eigenvalue ambiguity weights.",
    )

    parser.add_argument(
        "--ambiguity-w-max",
        type=float,
        default=AMBIGUITY_W_MAX,
        help="Maximum clipped spectral ambiguity weight.",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lambda_to_tag(x):
    """
    Convert a float lambda to a safe filename tag.

    Examples:
        0      -> 0
        1e-4   -> 1em04
        3e-4   -> 3em04
        1e-3   -> 1em03
        3e-3   -> 3em03
    """
    x = float(x)
    if x == 0:
        return "0"
    return f"{x:.0e}".replace("+", "").replace("-", "m")


# ============================================================
# File loading
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
    """
    Robustly convert np.str_, bytes, np.bytes_, and object scalar values to str.
    This avoids split labels becoming strings like "b'train'".
    """
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.bytes_):
        return x.tobytes().decode("utf-8")
    return str(x)


def _load_json_from_np_scalar(x):
    """
    08 stores params as json.dumps(PARAMS) inside np.savez_compressed.
    Depending on numpy dtype, .item() may be needed.
    """
    if hasattr(x, "item"):
        x = x.item()
    return json.loads(_as_python_string(x))


def waveform_to_column_tensor(Y_cell, gap, num_spots, axial_len):
    """
    Convert one serialized LASE waveform back to column tensor.

    Input:
        Y_cell: [T, L],
                T = num_spots * axial_len + (num_spots - 1) * gap

    Output:
        M: [L, H, N] = [16, 128, 44]
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
        if segment.shape != (axial_len, L):
            raise RuntimeError(
                f"Bad segment shape at spot {i}: got {segment.shape}, "
                f"expected {(axial_len, L)}"
            )

        cols[:, :, i] = segment.T                       # [L, H]
        offset += axial_len

        if i < num_spots - 1:
            offset += gap

    if offset != T:
        raise RuntimeError(f"Internal serialization error: final offset={offset}, T={T}")

    return cols.astype(np.float32)


# ============================================================
# Dataset
# ============================================================

class BBBC021LASEColumnDataset(Dataset):
    """
    Dataset for physics-guided inversion.

    Disk data:
        Y_obs_norm: [N, T, 16]
        A:          [N, 128, 128, 3]

    Returned sample:
        Y_cols:     [16, 128, 44]
        A_eval:     [3, 128, 128]

    Important:
        A_eval is returned only for evaluation and figures.
        It is never used in the optimization objective.
    """

    def __init__(self, split_name, metadata_path=None):
        super().__init__()

        if split_name not in ["train", "val", "test"]:
            raise ValueError(f"split_name must be train/val/test, got {split_name}")

        if metadata_path is None:
            metadata_path = find_latest_metadata_file(data_dir)

        self.metadata_path = Path(metadata_path)

        print(f"Loading metadata for {split_name}:")
        print(self.metadata_path)

        meta = np.load(self.metadata_path, allow_pickle=True)

        self.A_path = Path(str(meta["A_npy_path"].item()))
        self.Y_path = Path(str(meta["Y_obs_norm_npy_path"].item()))

        self.A = np.load(self.A_path, mmap_mode="r")
        self.Y = np.load(self.Y_path, mmap_mode="r")

        self.S = meta["S"].astype(np.float32)
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
            raise ValueError(
                f"A shape should end with {(IMAGE_H, IMAGE_W, NUM_DYES)}, got {self.A.shape[1:]}"
            )

        if self.Y.shape[1] != expected_T:
            raise ValueError(
                f"Y length mismatch: got T={self.Y.shape[1]}, expected {expected_T}. "
                f"num_spots={self.num_spots}, axial_len={self.axial_len}, gap={self.gap}"
            )

        if self.num_spots != NUM_SPOTS:
            raise ValueError(f"NUM_SPOTS mismatch: config {NUM_SPOTS}, metadata {self.num_spots}")

        if self.num_bands != NUM_BANDS:
            raise ValueError(f"NUM_BANDS mismatch: config {NUM_BANDS}, metadata {self.num_bands}")

        if len(self.split) != self.A.shape[0]:
            raise ValueError(f"split length mismatch: {len(self.split)} vs N={self.A.shape[0]}")

        mask = self.split == split_name
        self.indices = np.where(mask)[0]

        print(f"{split_name} samples:", len(self.indices))
        print("A mmap shape:", self.A.shape)
        print("Y mmap shape:", self.Y.shape)
        print("num_spots:", self.num_spots)
        print("axial_len:", self.axial_len)
        print("num_bands:", self.num_bands)
        print("gap:", self.gap)
        print("expected T:", expected_T)
        print("")

        if len(self.indices) == 0:
            raise ValueError(f"No samples found for split={split_name}. Check meta['split'].")

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

        Y_cols = torch.from_numpy(Y_cols).contiguous()
        A_eval = torch.from_numpy(A_eval).contiguous()

        return Y_cols, A_eval, real_idx


# ============================================================
# Differentiable LASE forward model
# ============================================================

def fwhm_to_sigma_torch(fwhm):
    return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def gaussian_kernel1d(sigma, device, dtype, truncate=GAUSSIAN_TRUNCATE):
    if sigma <= 0:
        return None

    radius = max(1, int(math.ceil(float(truncate) * float(sigma))))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / torch.sum(k)
    return k


def gaussian_blur2d_depthwise(x, sigma, truncate=GAUSSIAN_TRUNCATE):
    """
    x: [B, C, H, W]

    Uses separable depthwise convolution with replicate padding,
    matching scipy.ndimage.gaussian_filter(..., mode="nearest")
    as closely as practical in PyTorch.
    """
    if sigma <= 0:
        return x

    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    k = gaussian_kernel1d(sigma, device, dtype, truncate=truncate)
    radius = k.numel() // 2

    # Horizontal blur.
    weight_x = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, weight_x, groups=C)

    # Vertical blur.
    weight_y = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    x = F.conv2d(x, weight_y, groups=C)

    return x


def forward_lase_columns_torch(A, S, params):
    """
    Differentiable approximation of 08 sample_lase_columns.

    A:
        [B, 3, H, W]

    S:
        [3, 16]

    Returns:
        Y_pred_cols_norm:
            [B, 16, H, num_spots]

    Notes:
        08 included random spot gain noise via gain_std.
        In inverse mode that random gain is not known, so this deterministic
        forward model does not sample random gains.
    """
    if A.ndim != 4:
        raise ValueError(f"Expected A shape [B, 3, H, W], got {tuple(A.shape)}")

    B, D, H, W = A.shape

    if D != NUM_DYES:
        raise ValueError(f"Expected D={NUM_DYES}, got {D}")

    device = A.device
    dtype = A.dtype

    S = S.to(device=device, dtype=dtype)

    if S.shape != (NUM_DYES, NUM_BANDS):
        raise ValueError(f"Expected S shape {(NUM_DYES, NUM_BANDS)}, got {tuple(S.shape)}")

    # A -> spectral image X.
    # 08 equivalent:
    # X_cell = np.einsum("hwd,dl->hwl", A_cell, S)
    X = torch.einsum("bdhw,dl->blhw", A, S)  # [B, 16, H, W]
    X = torch.clamp(X, min=0.0)

    # Optical blur.
    patch_size_um = float(params["patch_size_um"])
    fwhm_um = float(params["fwhm_um"])

    pixel_size_um = patch_size_um / float(H)
    sigma_um = fwhm_to_sigma_torch(fwhm_um)
    sigma_pix = sigma_um / pixel_size_um

    X_blur = gaussian_blur2d_depthwise(X, sigma_pix, truncate=GAUSSIAN_TRUNCATE)

    # Sample columns along x direction using differentiable bilinear sampling.
    num_spots = int(params["num_spots"])
    x_positions = torch.linspace(0, W - 1, num_spots, device=device, dtype=dtype)

    x0 = torch.floor(x_positions).long()
    x1 = torch.clamp(x0 + 1, max=W - 1)
    wx = (x_positions - x0.to(dtype)).view(1, 1, 1, num_spots)

    X0 = torch.index_select(X_blur, dim=3, index=x0)  # [B, 16, H, N]
    X1 = torch.index_select(X_blur, dim=3, index=x1)  # [B, 16, H, N]

    cols = (1.0 - wx) * X0 + wx * X1                 # [B, 16, H, N]

    # Deterministic adjacent-spot sidelobe coupling.
    alpha = float(params.get("sidelobe_alpha", 0.0))

    if alpha > 0:
        left = torch.zeros_like(cols)
        right = torch.zeros_like(cols)

        left[:, :, :, 1:] = cols[:, :, :, :-1]
        right[:, :, :, :-1] = cols[:, :, :, 1:]

        neighbor_count = torch.ones(
            (1, 1, 1, num_spots),
            device=device,
            dtype=dtype,
        ) * 2.0
        neighbor_count[:, :, :, 0] = 1.0
        neighbor_count[:, :, :, -1] = 1.0

        center_weight = 1.0 - alpha * neighbor_count
        cols = center_weight * cols + alpha * (left + right)

    cols = torch.clamp(cols, min=0.0)

    return cols


# ============================================================
# Objective
# ============================================================

def poisson_kl_from_normalized(mu_norm, y_norm, params):
    """
    Poisson data-consistency term for background-subtracted normalized observations.

    Revised 08 saves:
        y_norm = (observed_counts - background_count) / peak_photon_count

    Therefore the observed photon counts are reconstructed as:
        y_counts = y_norm * peak_photon_count + background_count

    y_norm may be slightly negative in dark regions. This is expected and
    corresponds to observed photon counts smaller than the background level,
    e.g. (0 - 2) / 5000 = -0.0004. The Poisson KL/NLL is evaluated in the
    count domain after converting y_norm back to y_counts.

    The returned loss is a generalized KL form:
        mu - y + y * (log y - log mu)

    This has the same gradient with respect to mu as Poisson NLL up to
    constants independent of mu.
    """
    peak = float(params["peak_photon_count"])
    background = float(params["background_count"])

    mu_counts = peak * mu_norm + background
    y_counts = peak * y_norm + background

    mu_counts = torch.clamp(mu_counts, min=EPS)
    y_counts = torch.clamp(y_counts, min=0.0)

    loss = mu_counts - y_counts + y_counts * (
        torch.log(torch.clamp(y_counts, min=EPS)) - torch.log(mu_counts)
    )

    # Divide by peak to keep the numerical scale convenient for Adam.
    return torch.mean(loss) / peak


def tv_loss(A):
    """
    Anisotropic total variation for A: [B, 3, H, W].
    """
    dx = torch.abs(A[:, :, :, 1:] - A[:, :, :, :-1]).mean()
    dy = torch.abs(A[:, :, 1:, :] - A[:, :, :-1, :]).mean()
    return dx + dy


def channel_competition_sparse_loss(A, tau=CHANNEL_SPARSE_TAU, eps=EPS):
    """
    Per-pixel cross-channel competition sparsity for A: [B, 3, H, W].

    This encodes the assumption:
        at the same spatial location, only a small number of dyes should dominate.

    It is different from the old L1 sparse term A.mean():
        - old L1 sparse penalizes total abundance magnitude;
        - this term penalizes multi-dye co-activation at the same pixel.

    Implementation:
        P_c(h,w) = A_c(h,w) / sum_c A_c(h,w)

        impurity = 1 - sum_c P_c(h,w)^2

    impurity is near 0 when one dye dominates, and larger when channels are evenly mixed.
    The active_weight avoids forcing meaningless competition in near-background pixels.
    """
    if A.ndim != 4:
        raise ValueError(f"Expected A shape [B, 3, H, W], got {tuple(A.shape)}")

    A_sum = torch.sum(A, dim=1, keepdim=True)          # [B, 1, H, W]
    P = A / (A_sum + eps)                             # [B, 3, H, W]

    impurity = 1.0 - torch.sum(P ** 2, dim=1)         # [B, H, W]

    active_weight = A_sum.squeeze(1) / (A_sum.squeeze(1) + float(tau))
    active_weight = active_weight.detach()

    return torch.sum(active_weight * impurity) / (torch.sum(active_weight) + eps)


def make_spectral_competition_matrix(S, rho_threshold=ADAPTIVE_RHO_THRESHOLD):
    """
    Build thresholded spectral-correlation competition matrix.

    S: [D, L] dye spectra. Rows are L2-normalized before cosine correlation.

    rho_comp[k, j] = max(cos(s_k, s_j) - rho_threshold, 0), with zero diagonal.
    This prevents unrelated or weakly similar dyes from competing everywhere.
    """
    if S.ndim != 2:
        raise ValueError(f"Expected S shape [D, L], got {tuple(S.shape)}")

    S_norm = S / (torch.linalg.norm(S, dim=1, keepdim=True) + EPS)
    rho = torch.matmul(S_norm, S_norm.T)
    rho_comp = torch.clamp(rho - float(rho_threshold), min=0.0)
    rho_comp = rho_comp - torch.diag(torch.diag(rho_comp))
    return rho, rho_comp.detach()


def update_adaptive_sparse_weights(
    A,
    rho_comp,
    eps=ADAPTIVE_EPS,
    beta=ADAPTIVE_BETA,
    w_min=ADAPTIVE_W_MIN,
    w_max=ADAPTIVE_W_MAX,
):
    """
    Detached adaptive spectral-correlation reweighted sparse weights.

    A:        [B, D, H, W]
    rho_comp: [D, D], thresholded spectral competition matrix.

    w_{p,k} = 1 / (A_{p,k} + eps)
              + beta * sum_{j != k} rho_comp[k, j] * A_{p,j}

    The returned weights are clipped and detached. They are treated as fixed
    during the next optimizer step, which is the standard reweighted-L1
    majorization style and avoids gradients through the weight update.
    """
    if A.ndim != 4:
        raise ValueError(f"Expected A shape [B, D, H, W], got {tuple(A.shape)}")

    A_det = A.detach()
    inv_term = 1.0 / (A_det + float(eps))
    competition_term = torch.einsum("kj,bdhw->bkhw", rho_comp.to(A_det.device, A_det.dtype), A_det)
    weights = inv_term + float(beta) * competition_term
    weights = torch.clamp(weights, min=float(w_min), max=float(w_max))
    return weights.detach()


def adaptive_reweighted_sparse_loss(A, adaptive_weights):
    """
    R_adaptive(A; S) = mean_{p,k} w_{p,k} * A_{p,k}.

    adaptive_weights must be detached. If it is None, the loss is zero.
    """
    if adaptive_weights is None:
        return torch.zeros((), device=A.device, dtype=A.dtype)

    if adaptive_weights.shape != A.shape:
        raise ValueError(
            f"adaptive_weights shape {tuple(adaptive_weights.shape)} does not match A {tuple(A.shape)}"
        )

    return torch.mean(adaptive_weights.detach() * A)


def build_spectral_ambiguity_regularizer(
    S,
    eta=AMBIGUITY_ETA,
    power=AMBIGUITY_POWER,
    w_max=AMBIGUITY_W_MAX,
):
    """
    Build the eigensystem used for spectral-ambiguity-aware TV.

    S: [D, L] dye spectra. Rows are L2-normalized before computing the
    spectral correlation Gram matrix C = S_norm S_norm^T.

    If spectra are highly correlated, C has small eigenvalues. The
    corresponding eigenvectors define dye-combination directions that are
    weakly identifiable from spectral data. We therefore apply extra TV to
    abundance maps projected onto these directions, instead of forcing
    pixel-wise channel competition.
    """
    if S.ndim != 2:
        raise ValueError(f"Expected S shape [D, L], got {tuple(S.shape)}")

    S_norm = S / (torch.linalg.norm(S, dim=1, keepdim=True) + EPS)
    C = torch.matmul(S_norm, S_norm.T)
    eigvals, eigvecs = torch.linalg.eigh(C)

    # Largest eigenvalue gets weight about 1. Smaller eigenvalues get larger
    # weights, capped for numerical stability.
    lam_max = torch.clamp(torch.amax(eigvals), min=float(eta))
    weights = ((lam_max + float(eta)) / (eigvals + float(eta))) ** float(power)
    weights = torch.clamp(weights, min=1.0, max=float(w_max))

    return eigvals.detach(), eigvecs.detach(), weights.detach(), C.detach()


def spectral_ambiguity_tv_loss(A, eigvecs, eig_weights):
    """
    TV on spectral ambiguity directions.

    A:          [B, D, H, W]
    eigvecs:    [D, D], columns are eigenvectors of the spectral Gram matrix
    eig_weights:[D], larger for spectrally ambiguous directions

    This regularizer stabilizes poorly identifiable dye-combination directions
    without explicitly penalizing true multi-channel co-localization.
    """
    if eigvecs is None or eig_weights is None:
        return torch.zeros((), device=A.device, dtype=A.dtype)

    eigvecs = eigvecs.to(device=A.device, dtype=A.dtype)
    eig_weights = eig_weights.to(device=A.device, dtype=A.dtype)

    A_proj = torch.einsum("dm,bdhw->bmhw", eigvecs, A)

    dx = torch.abs(A_proj[:, :, :, 1:] - A_proj[:, :, :, :-1]).mean(dim=(0, 2, 3))
    dy = torch.abs(A_proj[:, :, 1:, :] - A_proj[:, :, :-1, :]).mean(dim=(0, 2, 3))
    tv_per_direction = dx + dy

    return torch.sum(eig_weights * tv_per_direction) / (torch.sum(eig_weights) + EPS)


def upper_bound_loss(A, upper=A_UPPER):
    """
    Soft upper-bound penalty for normalized simulated abundance maps.

    This is not a hard physical constraint for real fluorescence intensity.
    It is used here because the simulated evaluation abundance maps A_true
    were generated from normalized BBBC021 patches clipped to [0, 1].
    """
    return torch.mean(F.relu(A - float(upper)) ** 2)


def inverse_softplus_tensor(y):
    """
    Stable inverse of softplus for positive tensor y.
    """
    y = torch.clamp(y, min=INIT_A_FLOOR)

    return torch.where(
        y > 20.0,
        y,
        torch.log(torch.expm1(y)),
    )


def make_initial_A(batch_size, device, dtype):
    """
    Constant nonnegative initialization.

    This deliberately avoids any intermediate image reconstruction
    or external abundance estimate.
    """
    return torch.full(
        (batch_size, NUM_DYES, IMAGE_H, IMAGE_W),
        fill_value=INIT_A_CONSTANT,
        device=device,
        dtype=dtype,
    )


def physics_objective(
    Z,
    Y_obs_cols,
    S,
    params,
    sparse_mode,
    lambda_channel_sparse,
    channel_sparse_tau,
    lambda_adaptive_sparse,
    adaptive_sparse_weights,
    lambda_ambiguity_tv,
    ambiguity_eigvecs,
    ambiguity_eig_weights,
    lambda_tv,
    lambda_upper,
    a_upper,
):
    """
    Z:
        unconstrained variable [B, 3, H, W]

    A:
        softplus(Z), so A >= 0

    Loss:
        Poisson/KL data term between predicted and observed raw LASE columns
        + either fixed channel-competition sparsity or adaptive spectral-correlation reweighted sparsity
        + spatial TV penalty
        + soft upper-bound penalty for normalized simulated A.
    """
    A = F.softplus(Z)

    Y_pred_cols = forward_lase_columns_torch(
        A=A,
        S=S,
        params=params,
    )

    loss_pois = poisson_kl_from_normalized(
        mu_norm=Y_pred_cols,
        y_norm=Y_obs_cols,
        params=params,
    )

    loss_channel_sparse = channel_competition_sparse_loss(
        A,
        tau=channel_sparse_tau,
    )
    loss_adaptive_sparse = adaptive_reweighted_sparse_loss(
        A,
        adaptive_sparse_weights,
    )
    loss_tv = tv_loss(A)
    loss_ambiguity_tv = spectral_ambiguity_tv_loss(
        A,
        eigvecs=ambiguity_eigvecs,
        eig_weights=ambiguity_eig_weights,
    )
    loss_upper = upper_bound_loss(A, upper=a_upper)

    if sparse_mode == "fixed":
        sparse_weighted = lambda_channel_sparse * loss_channel_sparse
        adaptive_sparse_weighted = torch.zeros((), device=A.device, dtype=A.dtype)
    elif sparse_mode == "adaptive":
        sparse_weighted = torch.zeros((), device=A.device, dtype=A.dtype)
        adaptive_sparse_weighted = lambda_adaptive_sparse * loss_adaptive_sparse
    elif sparse_mode == "none":
        sparse_weighted = torch.zeros((), device=A.device, dtype=A.dtype)
        adaptive_sparse_weighted = torch.zeros((), device=A.device, dtype=A.dtype)
    else:
        raise ValueError(f"Unknown sparse_mode: {sparse_mode}")

    loss = (
        loss_pois
        + sparse_weighted
        + adaptive_sparse_weighted
        + lambda_tv * loss_tv
        + lambda_ambiguity_tv * loss_ambiguity_tv
        + lambda_upper * loss_upper
    )

    with torch.no_grad():
        over = F.relu(A - float(a_upper))
        over_frac = torch.mean((A > float(a_upper)).to(A.dtype))
        over_max = torch.amax(over)

    logs = {
        "total": float(loss.detach().cpu()),
        "poisson_kl": float(loss_pois.detach().cpu()),
        "sparse_mode": sparse_mode,
        "channel_sparse": float(loss_channel_sparse.detach().cpu()),
        "channel_sparse_weighted": float(sparse_weighted.detach().cpu()),
        "adaptive_sparse": float(loss_adaptive_sparse.detach().cpu()),
        "adaptive_sparse_weighted": float(adaptive_sparse_weighted.detach().cpu()),
        "tv": float(loss_tv.detach().cpu()),
        "tv_weighted": float((lambda_tv * loss_tv).detach().cpu()),
        "ambiguity_tv": float(loss_ambiguity_tv.detach().cpu()),
        "ambiguity_tv_weighted": float((lambda_ambiguity_tv * loss_ambiguity_tv).detach().cpu()),
        "upper": float(loss_upper.detach().cpu()),
        "upper_weighted": float((lambda_upper * loss_upper).detach().cpu()),
        "A_min": float(A.detach().amin().cpu()),
        "A_max": float(A.detach().amax().cpu()),
        "A_mean": float(A.detach().mean().cpu()),
        "A_over_upper_frac": float(over_frac.detach().cpu()),
        "A_over_upper_max": float(over_max.detach().cpu()),
    }

    return loss, A, Y_pred_cols, logs


def invert_batch_A(
    Y_obs_cols,
    S,
    params,
    device,
    num_iters,
    lr,
    sparse_mode,
    lambda_channel_sparse,
    channel_sparse_tau,
    lambda_adaptive_sparse,
    adaptive_beta,
    adaptive_eps,
    adaptive_rho_threshold,
    adaptive_weight_update_every,
    adaptive_w_min,
    adaptive_w_max,
    lambda_ambiguity_tv,
    ambiguity_eigvecs,
    ambiguity_eig_weights,
    lambda_tv,
    lambda_upper,
    a_upper,
    print_every,
):
    """
    Pure direct joint abundance inversion for one independent batch.

    Input:
        Y_obs_cols: [B, 16, 128, 44]

    Returns:
        A_hat:       [B, 3, 128, 128]
        Y_pred_cols: [B, 16, 128, 44]
        logs_history: list[dict]
    """
    Y_obs_cols = Y_obs_cols.to(device, non_blocking=True)

    with torch.no_grad():
        B = int(Y_obs_cols.shape[0])
        A_init = make_initial_A(
            batch_size=B,
            device=device,
            dtype=Y_obs_cols.dtype,
        )
        Z_init = inverse_softplus_tensor(A_init)

    Z = Z_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([Z], lr=lr)

    rho = None
    rho_comp = None
    adaptive_sparse_weights = None

    if sparse_mode == "adaptive":
        with torch.no_grad():
            rho, rho_comp = make_spectral_competition_matrix(
                S.to(device=device, dtype=Y_obs_cols.dtype),
                rho_threshold=adaptive_rho_threshold,
            )
            print("Spectral cosine rho:")
            print(rho.detach().cpu().numpy())
            print("Thresholded competition rho_comp:")
            print(rho_comp.detach().cpu().numpy())

    logs_history = []

    for it in range(1, num_iters + 1):
        optimizer.zero_grad(set_to_none=True)

        if sparse_mode == "adaptive" and (
            adaptive_sparse_weights is None
            or it == 1
            or (adaptive_weight_update_every > 0 and (it - 1) % adaptive_weight_update_every == 0)
        ):
            with torch.no_grad():
                A_for_weights = F.softplus(Z.detach())
                adaptive_sparse_weights = update_adaptive_sparse_weights(
                    A=A_for_weights,
                    rho_comp=rho_comp,
                    eps=adaptive_eps,
                    beta=adaptive_beta,
                    w_min=adaptive_w_min,
                    w_max=adaptive_w_max,
                )

        loss, A, Y_pred_cols, logs = physics_objective(
            Z=Z,
            Y_obs_cols=Y_obs_cols,
            S=S,
            params=params,
            sparse_mode=sparse_mode,
            lambda_channel_sparse=lambda_channel_sparse,
            channel_sparse_tau=channel_sparse_tau,
            lambda_adaptive_sparse=lambda_adaptive_sparse,
            adaptive_sparse_weights=adaptive_sparse_weights,
            lambda_ambiguity_tv=lambda_ambiguity_tv,
            ambiguity_eigvecs=ambiguity_eigvecs,
            ambiguity_eig_weights=ambiguity_eig_weights,
            lambda_tv=lambda_tv,
            lambda_upper=lambda_upper,
            a_upper=a_upper,
        )

        loss.backward()

        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_([Z], max_norm=GRAD_CLIP_NORM)

        optimizer.step()

        if it == 1 or it == num_iters or (print_every is not None and it % print_every == 0):
            if adaptive_sparse_weights is not None:
                logs["adaptive_w_min"] = float(adaptive_sparse_weights.amin().detach().cpu())
                logs["adaptive_w_max"] = float(adaptive_sparse_weights.amax().detach().cpu())
                logs["adaptive_w_mean"] = float(adaptive_sparse_weights.mean().detach().cpu())

            logs_with_iter = {"iter": it, **logs}
            logs_history.append(logs_with_iter)

            print(
                f"iter {it:04d}/{num_iters} | "
                f"loss={logs['total']:.6e} | "
                f"poisKL={logs['poisson_kl']:.6e} | "
                f"chSparse={logs['channel_sparse']:.6e} | "
                f"chSparseW={logs['channel_sparse_weighted']:.6e} | "
                f"adSparse={logs['adaptive_sparse']:.6e} | "
                f"adSparseW={logs['adaptive_sparse_weighted']:.6e} | "
                f"tv={logs['tv']:.6e} | "
                f"tvW={logs['tv_weighted']:.6e} | "
                f"ambTV={logs['ambiguity_tv']:.6e} | "
                f"ambTVW={logs['ambiguity_tv_weighted']:.6e} | "
                f"upper={logs['upper']:.6e} | "
                f"upperW={logs['upper_weighted']:.6e} | "
                f"A_mean={logs['A_mean']:.6e} | "
                f"A_max={logs['A_max']:.6e} | "
                f"over_frac={logs['A_over_upper_frac']:.3e}"
            )

    with torch.no_grad():
        A_hat = F.softplus(Z).detach()
        Y_pred_cols = forward_lase_columns_torch(
            A=A_hat,
            S=S,
            params=params,
        ).detach()

    return A_hat, Y_pred_cols, logs_history


# ============================================================
# Metrics and visualization
# ============================================================

def compute_batch_eval_metrics(A_hat, A_true):
    """
    Evaluation only. A_true is not used for optimization.

    Returns numpy arrays so that the caller can serialize them easily.
    """
    diff = A_hat - A_true

    mse = torch.mean(diff ** 2, dim=(1, 2, 3))
    mae = torch.mean(torch.abs(diff), dim=(1, 2, 3))
    rmse = torch.sqrt(mse + EPS)

    denom = torch.sqrt(torch.mean(A_true ** 2, dim=(1, 2, 3)) + EPS)
    nrmse = rmse / (denom + EPS)

    # Pearson is computed per dye and then averaged, matching main_exp_metrics.py.
    pearson_per_dye = []
    for d in range(A_hat.shape[1]):
        h = A_hat[:, d, :, :].flatten(start_dim=1)
        t = A_true[:, d, :, :].flatten(start_dim=1)
        h0 = h - h.mean(dim=1, keepdim=True)
        t0 = t - t.mean(dim=1, keepdim=True)
        r = torch.sum(h0 * t0, dim=1) / (
            torch.sqrt(torch.sum(h0 ** 2, dim=1) * torch.sum(t0 ** 2, dim=1)) + EPS
        )
        pearson_per_dye.append(r)
    pearson = torch.stack(pearson_per_dye, dim=1).mean(dim=1)

    # Per-dye metrics are useful for diagnosing dye-specific imbalance.
    per_dye_mse = torch.mean(diff ** 2, dim=(2, 3))
    per_dye_mae = torch.mean(torch.abs(diff), dim=(2, 3))
    per_dye_rmse = torch.sqrt(per_dye_mse + EPS)

    return {
        "mse": mse.detach().cpu().numpy(),
        "mae": mae.detach().cpu().numpy(),
        "rmse": rmse.detach().cpu().numpy(),
        "nrmse": nrmse.detach().cpu().numpy(),
        "pearson": pearson.detach().cpu().numpy(),
        "per_dye_mse": per_dye_mse.detach().cpu().numpy(),
        "per_dye_mae": per_dye_mae.detach().cpu().numpy(),
        "per_dye_rmse": per_dye_rmse.detach().cpu().numpy(),
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
    title="Pure joint Poisson channel-competition sparse inversion",
):
    """
    A_hat:  [3, H, W]
    A_true: [3, H, W]
    """
    A_hat = np.asarray(A_hat, dtype=np.float32)
    A_true = np.asarray(A_true, dtype=np.float32)
    err = np.abs(A_hat - A_true)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)

    for ch in range(NUM_DYES):
        name = channel_names[ch] if ch < len(channel_names) else f"ch{ch}"

        vmax = max(
            _safe_vmax(A_true[ch]),
            _safe_vmax(A_hat[ch]),
            1e-3,
        )

        axes[0, ch].imshow(A_true[ch], cmap="gray", vmin=0, vmax=vmax)
        axes[0, ch].set_title(f"Eval true {name}")
        axes[0, ch].axis("off")

        axes[1, ch].imshow(A_hat[ch], cmap="gray", vmin=0, vmax=vmax)
        axes[1, ch].set_title(f"Inverted {name}")
        axes[1, ch].axis("off")

        axes[2, ch].imshow(err[ch], cmap="magma")
        axes[2, ch].set_title(f"Abs error {name}")
        axes[2, ch].axis("off")

    fig.suptitle(title, fontsize=15)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved inversion figure:")
    print(out_path)


def save_column_fit_figure(
    Y_obs_cols,
    Y_pred_cols,
    wavelengths,
    out_path,
    bands=(0, 5, 10, 15),
):
    """
    Show observed vs predicted PMT column images for selected bands.

    Y_obs_cols:  [16, H, N]
    Y_pred_cols: [16, H, N]
    """
    Y_obs_cols = np.asarray(Y_obs_cols, dtype=np.float32)
    Y_pred_cols = np.asarray(Y_pred_cols, dtype=np.float32)

    fig, axes = plt.subplots(3, len(bands), figsize=(4 * len(bands), 9), constrained_layout=True)

    for j, b in enumerate(bands):
        obs = Y_obs_cols[b]
        pred = Y_pred_cols[b]
        err = np.abs(pred - obs)

        vmax = max(_safe_vmax(obs), _safe_vmax(pred), 1e-6)

        if b < len(wavelengths):
            band_title = f"B{b + 1} {wavelengths[b]:.0f} nm"
        else:
            band_title = f"B{b + 1}"

        axes[0, j].imshow(obs, cmap="gray", aspect="auto", vmin=0, vmax=vmax)
        axes[0, j].set_title(f"Observed {band_title}")
        axes[0, j].set_xlabel("Spot index")
        axes[0, j].set_ylabel("Axial index")

        axes[1, j].imshow(pred, cmap="gray", aspect="auto", vmin=0, vmax=vmax)
        axes[1, j].set_title(f"Predicted {band_title}")
        axes[1, j].set_xlabel("Spot index")
        axes[1, j].set_ylabel("Axial index")

        axes[2, j].imshow(err, cmap="magma", aspect="auto")
        axes[2, j].set_title(f"Abs error {band_title}")
        axes[2, j].set_xlabel("Spot index")
        axes[2, j].set_ylabel("Axial index")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved column fit figure:")
    print(out_path)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.variant not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown variant: {args.variant}")

    ablation_cfg = ABLATION_CONFIGS[args.variant]

    method_name = ablation_cfg["method_name"]
    sparse_mode = str(ablation_cfg.get("sparse_mode", "fixed"))
    lambda_channel_sparse = float(ablation_cfg["lambda_channel_sparse"])
    channel_sparse_tau = float(ablation_cfg["channel_sparse_tau"])
    lambda_adaptive_sparse = float(ablation_cfg.get("lambda_adaptive_sparse", 0.0))
    lambda_ambiguity_tv = float(ablation_cfg.get("lambda_ambiguity_tv", 0.0))
    lambda_tv = float(ablation_cfg["lambda_tv"])
    lambda_upper = float(ablation_cfg["lambda_upper"])

    if args.sparse_mode != "config":
        sparse_mode = args.sparse_mode

    # Command-line overrides for sparse weights.
    if args.lambda_channel_sparse is not None:
        lambda_channel_sparse = float(args.lambda_channel_sparse)

    if args.lambda_adaptive_sparse is not None:
        lambda_adaptive_sparse = float(args.lambda_adaptive_sparse)

    if args.lambda_ambiguity_tv is not None:
        lambda_ambiguity_tv = float(args.lambda_ambiguity_tv)

    if sparse_mode == "fixed":
        method_name = f"{method_name}_fixed_lam_{lambda_to_tag(lambda_channel_sparse)}"
    elif sparse_mode == "adaptive":
        method_name = (
            f"{method_name}_adaptive_lam_{lambda_to_tag(lambda_adaptive_sparse)}"
            f"_beta_{lambda_to_tag(args.adaptive_beta)}"
            f"_rho_{lambda_to_tag(args.adaptive_rho_threshold)}"
        )
    elif sparse_mode == "none":
        method_name = f"{method_name}_nosparse"
    else:
        raise ValueError(f"Unknown sparse_mode: {sparse_mode}")

    if lambda_ambiguity_tv > 0:
        method_name = (
            f"{method_name}_ambtv_{lambda_to_tag(lambda_ambiguity_tv)}"
            f"_p_{lambda_to_tag(args.ambiguity_power)}"
        )

    max_samples = MAX_SAMPLES
    if args.max_samples is not None:
        max_samples = int(args.max_samples)

    num_iters = NUM_ITERS
    if args.num_iters is not None:
        num_iters = int(args.num_iters)

    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Ablation variant:", args.variant)
    print("Method name:", method_name)
    print("Description:", ablation_cfg["description"])
    print("sparse_mode:", sparse_mode)
    print("lambda_channel_sparse:", lambda_channel_sparse)
    print("channel_sparse_tau:", channel_sparse_tau)
    print("lambda_adaptive_sparse:", lambda_adaptive_sparse)
    print("adaptive_beta:", args.adaptive_beta)
    print("adaptive_eps:", args.adaptive_eps)
    print("adaptive_rho_threshold:", args.adaptive_rho_threshold)
    print("adaptive_weight_update_every:", args.adaptive_weight_update_every)
    print("adaptive_w_min:", args.adaptive_w_min)
    print("adaptive_w_max:", args.adaptive_w_max)
    print("lambda_ambiguity_tv:", lambda_ambiguity_tv)
    print("ambiguity_eta:", args.ambiguity_eta)
    print("ambiguity_power:", args.ambiguity_power)
    print("ambiguity_w_max:", args.ambiguity_w_max)
    print("lambda_tv:", lambda_tv)
    print("lambda_upper:", lambda_upper)
    print("max_samples:", max_samples)
    print("num_iters:", num_iters)
    print("")

    dataset = BBBC021LASEColumnDataset(INVERT_SPLIT)

    # Save the actual metadata path used by the dataset.
    metadata_path = dataset.metadata_path

    n_total = len(dataset)

    # Use fixed local sample indices so repeated runs evaluate the same cells.
    eval_local_ids = get_or_create_eval_ids(list(range(n_total)))

    if max_samples is not None:
        eval_local_ids = eval_local_ids[:max_samples]

    n_to_process = len(eval_local_ids)

    if n_to_process <= 0:
        raise ValueError("n_to_process must be positive.")

    print("Inversion split:", INVERT_SPLIT)
    print("Samples in split:", n_total)
    print("Samples to process:", n_to_process)
    print("Eval local ids first 10:", eval_local_ids[:10])
    print("Method name:", method_name)
    print("Batch size:", INVERSION_BATCH_SIZE)
    print("Iterations per batch:", num_iters)
    print("")

    params = dataset.params

    # Enforce key shape assumptions.
    if int(params["num_spots"]) != NUM_SPOTS:
        raise ValueError(f"params['num_spots']={params['num_spots']} does not match NUM_SPOTS={NUM_SPOTS}")

    if abs(float(params.get("gain_std", 0.0))) > 0:
        print(
            "Note: metadata params include gain_std="
            f"{params.get('gain_std')}. This inverse model does not sample random gains; "
            "it uses the deterministic optical blur, column sampling, and sidelobe model."
        )

    obs_norm = params.get("observation_normalization", "unknown")
    print("Observation normalization:", obs_norm)
    print("S row sums:", np.sum(dataset.S, axis=1))
    print("S row max:", np.max(dataset.S, axis=1))

    S = torch.from_numpy(dataset.S).float().to(device)

    ambiguity_eigvals, ambiguity_eigvecs, ambiguity_eig_weights, ambiguity_gram = (
        build_spectral_ambiguity_regularizer(
            S,
            eta=args.ambiguity_eta,
            power=args.ambiguity_power,
            w_max=args.ambiguity_w_max,
        )
    )
    print("Spectral cosine Gram matrix C = S_norm @ S_norm.T:")
    print(ambiguity_gram.detach().cpu().numpy())
    print("Spectral Gram eigenvalues ascending:")
    print(ambiguity_eigvals.detach().cpu().numpy())
    print("Spectral ambiguity TV weights:")
    print(ambiguity_eig_weights.detach().cpu().numpy())
    print("")

    eval_dataset = Subset(dataset, eval_local_ids)

    loader = DataLoader(
        eval_dataset,
        batch_size=INVERSION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    out_dir = result_dir / f"{INVERT_SPLIT}_{method_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    A_hat_path = out_dir / f"A_hat_{INVERT_SPLIT}_{n_to_process}samples.npy"
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

        Y_obs_cols, A_true, real_idx = batch

        remaining = n_to_process - cursor
        if Y_obs_cols.shape[0] > remaining:
            Y_obs_cols = Y_obs_cols[:remaining]
            A_true = A_true[:remaining]
            real_idx = real_idx[:remaining]

        batch_size = int(Y_obs_cols.shape[0])

        print("\n" + "=" * 80)
        print(
            f"Batch {batch_idx} | output rows {cursor}:{cursor + batch_size} | "
            f"real_idx={real_idx.detach().cpu().numpy().tolist()}"
        )
        print("=" * 80)

        A_hat, Y_pred_cols, logs_history = invert_batch_A(
            Y_obs_cols=Y_obs_cols,
            S=S,
            params=params,
            device=device,
            num_iters=num_iters,
            lr=LEARNING_RATE,
            sparse_mode=sparse_mode,
            lambda_channel_sparse=lambda_channel_sparse,
            channel_sparse_tau=channel_sparse_tau,
            lambda_adaptive_sparse=lambda_adaptive_sparse,
            adaptive_beta=args.adaptive_beta,
            adaptive_eps=args.adaptive_eps,
            adaptive_rho_threshold=args.adaptive_rho_threshold,
            adaptive_weight_update_every=args.adaptive_weight_update_every,
            adaptive_w_min=args.adaptive_w_min,
            adaptive_w_max=args.adaptive_w_max,
            lambda_ambiguity_tv=lambda_ambiguity_tv,
            ambiguity_eigvecs=ambiguity_eigvecs,
            ambiguity_eig_weights=ambiguity_eig_weights,
            lambda_tv=lambda_tv,
            lambda_upper=lambda_upper,
            a_upper=A_UPPER,
            print_every=PRINT_EVERY,
        )

        A_true_device = A_true.to(device, non_blocking=True)

        eval_metrics = compute_batch_eval_metrics(A_hat, A_true_device)
        mse_arr = eval_metrics["mse"]
        mae_arr = eval_metrics["mae"]
        rmse_arr = eval_metrics["rmse"]
        nrmse_arr = eval_metrics["nrmse"]
        pearson_arr = eval_metrics["pearson"]
        per_dye_mae_arr = eval_metrics["per_dye_mae"]
        per_dye_rmse_arr = eval_metrics["per_dye_rmse"]

        A_hat_np = A_hat.detach().cpu().numpy().astype(np.float32)
        Y_pred_np = Y_pred_cols.detach().cpu().numpy().astype(np.float32)
        Y_obs_np = Y_obs_cols.detach().cpu().numpy().astype(np.float32)
        A_true_np = A_true.detach().cpu().numpy().astype(np.float32)
        real_idx_np = real_idx.detach().cpu().numpy().astype(np.int64)

        A_hat_map[cursor:cursor + batch_size] = A_hat_np
        A_hat_map.flush()

        real_indices[cursor:cursor + batch_size] = real_idx_np

        for j in range(batch_size):
            row = cursor + j

            final_logs = logs_history[-1] if len(logs_history) else {}

            item_metrics = {
                "output_row": int(row),
                "dataset_split": INVERT_SPLIT,
                "real_idx": int(real_idx_np[j]),
                "mse_vs_eval_A": float(mse_arr[j]),
                "mae_vs_eval_A": float(mae_arr[j]),
                "rmse_vs_eval_A": float(rmse_arr[j]),
                "nrmse_vs_eval_A": float(nrmse_arr[j]),
                "pearson_vs_eval_A": float(pearson_arr[j]),
                "per_dye_mae_vs_eval_A": [float(x) for x in per_dye_mae_arr[j]],
                "per_dye_rmse_vs_eval_A": [float(x) for x in per_dye_rmse_arr[j]],
                "final_logs": final_logs,
            }
            metrics.append(item_metrics)

            print(
                f"row={row} real_idx={int(real_idx_np[j])} | "
                f"MSE(eval only)={mse_arr[j]:.6e} | "
                f"MAE(eval only)={mae_arr[j]:.6e} | "
                f"NRMSE={nrmse_arr[j]:.6e} | "
                f"Pearson={pearson_arr[j]:.4f}"
            )

            if row < SAVE_FIGURES_FOR_FIRST_N:
                inv_fig_path = fig_dir / f"inversion_row_{row:04d}_realidx_{int(real_idx_np[j])}.png"
                save_inversion_figure(
                    A_hat=A_hat_np[j],
                    A_true=A_true_np[j],
                    out_path=inv_fig_path,
                    channel_names=dataset.channel_names,
                )

                fit_fig_path = fig_dir / f"column_fit_row_{row:04d}_realidx_{int(real_idx_np[j])}.png"
                save_column_fit_figure(
                    Y_obs_cols=Y_obs_np[j],
                    Y_pred_cols=Y_pred_np[j],
                    wavelengths=dataset.wavelengths,
                    out_path=fit_fig_path,
                )

        cursor += batch_size

    A_hat_map.flush()

    real_indices_path = out_dir / f"real_indices_{INVERT_SPLIT}_{n_to_process}samples.npy"
    np.save(real_indices_path, real_indices)

    metrics_path = out_dir / f"metrics_{INVERT_SPLIT}_{n_to_process}samples.json"

    mse_values = [m["mse_vs_eval_A"] for m in metrics]
    mae_values = [m["mae_vs_eval_A"] for m in metrics]
    rmse_values = [m["rmse_vs_eval_A"] for m in metrics]
    nrmse_values = [m["nrmse_vs_eval_A"] for m in metrics]
    pearson_values = [m["pearson_vs_eval_A"] for m in metrics]

    summary = {
        "script": "09_invert_bbbc021_spectral_ambiguity_tv_44spots_final_ablation.py",
        "method": method_name,
        "ablation_variant": args.variant,
        "ablation_description": ablation_cfg["description"],
        "metadata_path": str(metadata_path),
        "split": INVERT_SPLIT,
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
            "inversion_batch_size": INVERSION_BATCH_SIZE,
            "num_iters": num_iters,
            "max_samples": max_samples,
            "learning_rate": LEARNING_RATE,
            "sparse_mode": sparse_mode,
            "lambda_channel_sparse": lambda_channel_sparse,
            "channel_sparse_tau": channel_sparse_tau,
            "lambda_adaptive_sparse": lambda_adaptive_sparse,
            "lambda_ambiguity_tv": lambda_ambiguity_tv,
            "ambiguity_eta": float(args.ambiguity_eta),
            "ambiguity_power": float(args.ambiguity_power),
            "ambiguity_w_max": float(args.ambiguity_w_max),
            "ambiguity_eigvals": [float(x) for x in ambiguity_eigvals.detach().cpu().numpy()],
            "ambiguity_eig_weights": [float(x) for x in ambiguity_eig_weights.detach().cpu().numpy()],
            "adaptive_beta": float(args.adaptive_beta),
            "adaptive_eps": float(args.adaptive_eps),
            "adaptive_rho_threshold": float(args.adaptive_rho_threshold),
            "adaptive_weight_update_every": int(args.adaptive_weight_update_every),
            "adaptive_w_min": float(args.adaptive_w_min),
            "adaptive_w_max": float(args.adaptive_w_max),
            "lambda_tv": lambda_tv,
            "lambda_upper": lambda_upper,
            "a_upper": A_UPPER,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "init_A_constant": INIT_A_CONSTANT,
            "init_A_floor": INIT_A_FLOOR,
            "gaussian_truncate": GAUSSIAN_TRUNCATE,
        },
        "params_from_metadata": params,
        "eval_metrics_note": "A_true is used only for evaluation and figures, not for optimization. This pure joint variant uses constant initialization only. Sparse mode may be none, fixed channel competition, or adaptive spectral-correlation reweighted sparsity. Spectral-ambiguity TV can be added to stabilize ill-conditioned spectral directions without enforcing pixel-wise channel exclusivity.",
        "mean_mse_vs_eval_A": float(np.mean(mse_values)) if mse_values else None,
        "mean_mae_vs_eval_A": float(np.mean(mae_values)) if mae_values else None,
        "mean_rmse_vs_eval_A": float(np.mean(rmse_values)) if rmse_values else None,
        "mean_nrmse_vs_eval_A": float(np.mean(nrmse_values)) if nrmse_values else None,
        "mean_pearson_vs_eval_A": float(np.mean(pearson_values)) if pearson_values else None,
        "items": metrics,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    main_npz_path = Path(MAIN_OUT_DIR) / f"{method_name}.npz"
    np.savez(
        main_npz_path,
        sample_ids=real_indices.copy(),
        mae=np.array(mae_values, dtype=np.float32),
        rmse=np.array(rmse_values, dtype=np.float32),
        nrmse=np.array(nrmse_values, dtype=np.float32),
        pearson=np.array(pearson_values, dtype=np.float32),
    )

    print("\n" + "=" * 80)
    print("Finished pure joint Poisson spectral-ambiguity-aware inversion with constant initialization.")
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