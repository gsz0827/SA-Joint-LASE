from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# ============================================================
# Paths
# ============================================================

bbbc_root = Path(__file__).resolve().parent

metadata_dir = bbbc_root / "data" / "processed" / "bbbc021_spectral_metadata"

out_dir = bbbc_root / "data" / "processed" / "bbbc021_lase_train_ready_44spots"
out_dir.mkdir(parents=True, exist_ok=True)

result_dir = bbbc_root / "results" / "bbbc021_lase_train_ready_preview_44spots"
result_dir.mkdir(parents=True, exist_ok=True)

# Optional: set this to a specific metadata file if you do not want to use
# the newest bbbc021_3ch_spectral_metadata_*cells.npz automatically.
INPUT_METADATA_FILE = None


# ============================================================
# LASE simulation parameters
# ============================================================

PARAMS = {
    # Physical field of view represented by a 128 x 128 single-cell patch.
    "patch_size_um": 30.0,

    # This is an algorithm-development setting, not an exact reproduction of
    # Han et al. Device 2023. 44 spots give dx = 30/(44-1) = 0.698 um.
    "num_spots": 44,

    # Effective Gaussian optical blur FWHM in the object plane.
    # Use 1.0 um as a practical model-matched setting; it is less idealized than 0.6 um while still not being an extreme stress test.
    "fwhm_um": 1.0,

    # Keep gain_std = 0.0 for the first model-matched Poisson inversion.
    # Unknown random spot gains are not estimated in the current inversion model.
    # Keep gain_std = 0.0 for the main model-matched experiment.
    "gain_std": 0.0,

    # Deterministic nearest-neighbor column/spot sidelobe coupling.
    # The inversion scripts read the same value from metadata, so this
    # nonideality is included consistently in both forward simulation and inversion.
    "sidelobe_alpha": 0.04,

    # Gap between serialized LASE segments, expressed as a fraction of axial H.
    "gap_ratio": 0.01,

    # Photon-count simulation: counts ~ Poisson(peak * clean_signal + background).
    "peak_photon_count": 1000.0,
    "background_count": 10.0,

    # Normalization saved to disk:
    #   Y_obs_norm = (observed_counts - background_count) / peak_photon_count
    # Negative values are intentionally preserved, so 09 can reconstruct the
    # actual Poisson counts by peak * Y_obs_norm + background_count.
    "observation_normalization": "background_subtracted_by_peak_no_clip",

    "random_seed": 20260517,
}

SAVE_DTYPE = np.float32
EPS = 1e-8


# ============================================================
# Utility functions
# ============================================================

def fwhm_to_sigma(fwhm):
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def find_latest_metadata_file(metadata_dir):
    files = sorted(metadata_dir.glob("bbbc021_3ch_spectral_metadata_*cells.npz"))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No bbbc021_3ch_spectral_metadata_*cells.npz found in {metadata_dir}"
        )

    files = sorted(files, key=lambda p: p.stat().st_mtime)
    return files[-1]


def choose_metadata_file():
    if INPUT_METADATA_FILE is not None:
        p = Path(INPUT_METADATA_FILE)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_METADATA_FILE does not exist: {p}")
        return p
    return find_latest_metadata_file(metadata_dir)


def _as_str(x):
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.bytes_):
        return x.tobytes().decode("utf-8")
    return str(x)


def validate_params(params, H, W):
    required = [
        "patch_size_um", "num_spots", "fwhm_um", "gain_std", "sidelobe_alpha",
        "gap_ratio", "peak_photon_count", "background_count", "random_seed",
    ]
    for k in required:
        if k not in params:
            raise KeyError(f"PARAMS missing key: {k}")

    if params["patch_size_um"] <= 0:
        raise ValueError("patch_size_um must be positive.")
    if params["num_spots"] <= 1:
        raise ValueError("num_spots must be > 1.")
    if params["num_spots"] > W:
        raise ValueError(
            f"num_spots={params['num_spots']} exceeds image width W={W}. "
            "This script samples existing columns and does not supersample."
        )
    if params["fwhm_um"] < 0:
        raise ValueError("fwhm_um must be non-negative.")
    if params["gain_std"] < 0:
        raise ValueError("gain_std must be non-negative.")
    if params["sidelobe_alpha"] < 0:
        raise ValueError("sidelobe_alpha must be non-negative.")
    if params["sidelobe_alpha"] >= 0.5:
        raise ValueError("sidelobe_alpha must be < 0.5 for non-negative center weights.")
    if params["gap_ratio"] < 0:
        raise ValueError("gap_ratio must be non-negative.")
    if params["peak_photon_count"] <= 0:
        raise ValueError("peak_photon_count must be positive.")
    if params["background_count"] < 0:
        raise ValueError("background_count must be non-negative.")

    dx_um = params["patch_size_um"] / float(params["num_spots"] - 1)
    pixel_size_um = params["patch_size_um"] / float(H)
    sigma_pix = fwhm_to_sigma(params["fwhm_um"]) / pixel_size_um
    return dx_um, pixel_size_um, sigma_pix


def validate_loaded_data(A, S, wavelengths, split, cell_ids):
    if A.ndim != 4:
        raise ValueError(f"Expected A shape [N, H, W, 3], got {A.shape}")
    if A.shape[0] <= 0:
        raise ValueError("A contains zero cells.")
    if A.shape[-1] != 3:
        raise ValueError(f"Expected A last dimension D=3, got {A.shape[-1]}")
    if not np.all(np.isfinite(A)):
        raise ValueError("A contains NaN or Inf.")
    if A.min() < -1e-6:
        raise ValueError(f"A contains negative values: min={A.min()}")

    if S.ndim != 2:
        raise ValueError(f"Expected S shape [3, L], got {S.shape}")
    if S.shape[0] != 3:
        raise ValueError(f"Expected S.shape[0]=3, got {S.shape[0]}")
    if not np.all(np.isfinite(S)):
        raise ValueError("S contains NaN or Inf.")
    if S.min() < -1e-8:
        raise ValueError(f"S contains negative values: min={S.min()}")

    L = S.shape[1]
    if wavelengths.ndim != 1 or wavelengths.shape[0] != L:
        raise ValueError(f"Expected wavelengths shape ({L},), got {wavelengths.shape}")
    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError("wavelengths must be strictly increasing.")

    n = A.shape[0]
    if len(split) != n:
        raise ValueError(f"split length mismatch: len(split)={len(split)}, n={n}")
    if len(cell_ids) != n:
        raise ValueError(f"cell_ids length mismatch: len(cell_ids)={len(cell_ids)}, n={n}")

    row_sums = S.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        print("Warning: S rows are not unit-sum. row_sums =", row_sums)


def sample_lase_columns(X_cell, params, rng):
    """
    Input:
        X_cell: [H, W, L], non-negative spectral image.

    Output:
        cols: [H, N, L], decoded LASE column matrix before serialization.
    """
    H, W, L = X_cell.shape

    num_spots = int(params["num_spots"])
    patch_size_um = float(params["patch_size_um"])
    fwhm_um = float(params["fwhm_um"])

    X_cell = np.clip(X_cell.astype(np.float32), 0, None)

    pixel_size_um = patch_size_um / H
    sigma_um = fwhm_to_sigma(fwhm_um)
    sigma_pix = sigma_um / pixel_size_um

    if sigma_pix > 0:
        X_blur = gaussian_filter(
            X_cell,
            sigma=(sigma_pix, sigma_pix, 0),
            mode="nearest",
        ).astype(np.float32)
    else:
        X_blur = X_cell.copy()

    # Evenly spaced spot centers across the 30-um FOV.
    x_positions_pix = np.linspace(0, W - 1, num_spots, dtype=np.float32)

    cols = np.zeros((H, num_spots, L), dtype=np.float32)

    # Linear interpolation along x gives a smooth forward operator when the spot
    # positions are not exactly integer pixel columns.
    for i, x in enumerate(x_positions_pix):
        x0 = int(np.floor(float(x)))
        x1 = min(x0 + 1, W - 1)
        wx = float(x - x0)
        cols[:, i, :] = (1.0 - wx) * X_blur[:, x0, :] + wx * X_blur[:, x1, :]

    # Optional random spot gain. Default is 0 for model-matched inversion.
    gain_std = float(params.get("gain_std", 0.0))
    if gain_std > 0:
        gains = rng.normal(loc=1.0, scale=gain_std, size=(1, num_spots, 1)).astype(np.float32)
        gains = np.clip(gains, 0.5, 1.5)
        cols = cols * gains

    # Deterministic nearest-neighbor sidelobe coupling across spot/column index.
    alpha = float(params.get("sidelobe_alpha", 0.0))
    if alpha > 0:
        left = np.zeros_like(cols)
        right = np.zeros_like(cols)
        left[:, 1:, :] = cols[:, :-1, :]
        right[:, :-1, :] = cols[:, 1:, :]

        neighbor_count = np.ones((1, num_spots, 1), dtype=np.float32) * 2.0
        neighbor_count[:, 0, :] = 1.0
        neighbor_count[:, -1, :] = 1.0
        center_weight = 1.0 - alpha * neighbor_count
        cols = center_weight * cols + alpha * (left + right)

    return np.clip(cols, 0, None).astype(np.float32)


def serialize_columns(cols, params):
    """
    cols: [H, N, L]
    waveform: [T, L]
    """
    H, N, L = cols.shape
    gap = int(np.ceil(float(params["gap_ratio"]) * H))
    gap = max(gap, 0)

    T = N * H + (N - 1) * gap
    waveform = np.zeros((T, L), dtype=np.float32)

    offset = 0
    for i in range(N):
        waveform[offset:offset + H, :] = cols[:, i, :]
        offset += H
        if i < N - 1:
            offset += gap

    if offset != T:
        raise RuntimeError(f"Internal serialization error: final offset={offset}, T={T}")

    return waveform.astype(np.float32), gap


def add_poisson_noise(waveform_clean, params, rng):
    peak = float(params["peak_photon_count"])
    background = float(params["background_count"])

    waveform_clean = np.clip(waveform_clean, 0, None)
    waveform_lambda = peak * waveform_clean + background
    waveform_obs = rng.poisson(waveform_lambda).astype(np.float32)
    return waveform_obs


def normalize_observed_counts(waveform_obs, params):
    """
    Return background-subtracted normalized observations.

    Important: do NOT clip negative values after background subtraction.
    If observed_counts is 0 or 1 and background_count is 2, the normalized value
    is negative. Preserving this lets the Poisson loss recover the original counts
    via counts = peak * Y_obs_norm + background_count.
    """
    peak = float(params["peak_photon_count"])
    background = float(params["background_count"])
    y = (waveform_obs.astype(np.float32) - background) / (peak + EPS)
    return y.astype(np.float32)


def waveform_to_column_matrix(waveform, gap, num_spots=44, axial_len=128):
    """Convert serialized waveform [T, L] back to column matrix [H, N, L]."""
    T, L = waveform.shape
    expected_T = num_spots * axial_len + (num_spots - 1) * gap
    if T != expected_T:
        raise ValueError(
            f"Waveform length mismatch: got T={T}, expected {expected_T} "
            f"for num_spots={num_spots}, axial_len={axial_len}, gap={gap}"
        )

    cols = np.zeros((axial_len, num_spots, L), dtype=np.float32)
    offset = 0
    for i in range(num_spots):
        cols[:, i, :] = waveform[offset:offset + axial_len, :]
        offset += axial_len
        if i < num_spots - 1:
            offset += gap

    if offset != T:
        raise RuntimeError(f"Internal serialization error: final offset={offset}, T={T}")
    return cols


def _safe_vmax(img, percentile=99.5):
    img = np.asarray(img, dtype=np.float32)
    vmax = np.percentile(img, percentile)
    if vmax <= EPS:
        vmax = float(np.max(img)) + EPS
    return vmax


def plot_band_image_grid(data, wavelengths, out_path, title_prefix, cmap="gray", aspect=None):
    """Display all spectral bands in a nearly square grid."""
    L = int(data.shape[-1])
    ncols = int(np.ceil(np.sqrt(L)))
    nrows = int(np.ceil(L / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), constrained_layout=True)
    axes = np.asarray(axes).ravel()

    for b in range(L):
        img = data[:, :, b]
        vmax = _safe_vmax(img)

        axes[b].imshow(
            img,
            cmap=cmap,
            aspect=aspect if aspect is not None else None,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        axes[b].set_title(f"{title_prefix} band {b + 1}\n{wavelengths[b]:.0f} nm")
        axes[b].set_xticks([])
        axes[b].set_yticks([])

        if data.shape[1] == PARAMS["num_spots"]:
            axes[b].set_xlabel("Spot index")
            axes[b].set_ylabel("Axial index")

    for b in range(L, len(axes)):
        axes[b].axis("off")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved preview:")
    print(out_path)


def plot_waveform_grid(Y_clean0, Y_obs_norm0, wavelengths, out_path):
    """Plot full clean and observed normalized waveforms for all bands."""
    T, L = Y_clean0.shape
    step = max(1, T // 2000)
    x = np.arange(0, T, step)

    ncols = int(np.ceil(np.sqrt(L)))
    nrows = int(np.ceil(L / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), constrained_layout=True)
    axes = np.asarray(axes).ravel()

    for b in range(L):
        axes[b].plot(x, Y_clean0[x, b], label="clean signal", linewidth=1.2)
        axes[b].plot(x, Y_obs_norm0[x, b], label="obs bg-sub norm", linewidth=0.7, alpha=0.75)
        axes[b].set_title(f"Waveform band {b + 1} ({wavelengths[b]:.0f} nm)")
        axes[b].set_xlabel("Time sample")
        axes[b].set_ylabel("Norm. intensity")
        axes[b].grid(True, alpha=0.3)
        if b == 0:
            axes[b].legend(fontsize=8)

    for b in range(L, len(axes)):
        axes[b].axis("off")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved preview:")
    print(out_path)


def save_preview(A0, X0, Y_clean0, Y_obs_norm0, gap, wavelengths, out_path):
    """Save preview figures for checking the BBBC021 LASE simulation."""
    out_path = Path(out_path)
    out_dir_preview = out_path.parent
    stem = out_path.stem

    axial_len = int(A0.shape[0])

    cols_clean = waveform_to_column_matrix(
        Y_clean0,
        gap=gap,
        num_spots=int(PARAMS["num_spots"]),
        axial_len=axial_len,
    )

    cols_obs = waveform_to_column_matrix(
        Y_obs_norm0,
        gap=gap,
        num_spots=int(PARAMS["num_spots"]),
        axial_len=axial_len,
    )

    titles = ["DNA", "beta-tubulin", "F-actin", "Merged"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)

    for ch in range(3):
        axes[ch].imshow(A0[:, :, ch], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[ch].set_title(f"A {titles[ch]}")
        axes[ch].axis("off")

    axes[3].imshow(np.clip(A0, 0, 1), vmin=0, vmax=1, interpolation="nearest")
    axes[3].set_title("A merged")
    axes[3].axis("off")

    A_path = out_dir_preview / f"{stem}_A_overview.png"
    plt.savefig(A_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved preview:")
    print(A_path)

    X_path = out_dir_preview / f"{stem}_X_allbands.png"
    plot_band_image_grid(X0, wavelengths, X_path, title_prefix="X", cmap="gray", aspect=None)

    clean_cols_path = out_dir_preview / f"{stem}_clean_columns_allbands.png"
    plot_band_image_grid(cols_clean, wavelengths, clean_cols_path, title_prefix="Clean LASE", cmap="gray", aspect="auto")

    obs_cols_path = out_dir_preview / f"{stem}_observed_columns_allbands.png"
    plot_band_image_grid(cols_obs, wavelengths, obs_cols_path, title_prefix="Observed input", cmap="gray", aspect="auto")

    waveform_path = out_dir_preview / f"{stem}_waveforms_allbands.png"
    plot_waveform_grid(Y_clean0, Y_obs_norm0, wavelengths, waveform_path)


def main():
    rng = np.random.default_rng(int(PARAMS["random_seed"]))

    metadata_path = choose_metadata_file()
    print("Loading metadata:")
    print(metadata_path)

    meta = np.load(metadata_path, allow_pickle=True)

    S = meta["S"].astype(np.float32)
    wavelengths = meta["wavelengths"].astype(np.float32)
    split = meta["split"]
    cell_ids = meta["cell_ids"]
    similarity = meta["similarity"].astype(np.float32) if "similarity" in meta.files else np.eye(S.shape[0], dtype=np.float32)
    channel_names = meta["channel_names"] if "channel_names" in meta.files else np.array(["DNA", "beta_tubulin", "F_actin"], dtype=object)

    source_A_file = Path(_as_str(meta["source_A_file"]))

    print("Loading A file:")
    print(source_A_file)

    A_data = np.load(source_A_file, allow_pickle=True)
    A = A_data["A"].astype(np.float32)

    validate_loaded_data(A, S, wavelengths, split, cell_ids)

    n, H, W, D = A.shape
    L = S.shape[1]

    if H != W:
        print(f"Warning: expected square patches, got H={H}, W={W}")

    dx_um, pixel_size_um, sigma_pix = validate_params(PARAMS, H, W)

    gap = int(np.ceil(float(PARAMS["gap_ratio"]) * H))
    T = int(PARAMS["num_spots"]) * H + (int(PARAMS["num_spots"]) - 1) * gap

    # Expected signal statistics before running the full loop.
    x_upper = float(np.max(A)) * float(np.max(S))
    expected_peak_counts_upper = float(PARAMS["peak_photon_count"]) * x_upper + float(PARAMS["background_count"])

    print("A shape:", A.shape)
    print("A min/max:", float(A.min()), float(A.max()))
    print("S shape:", S.shape)
    print("S row sums:", S.sum(axis=1))
    print("S row max:", S.max(axis=1))
    print("wavelengths:", wavelengths)
    print("num_spots:", int(PARAMS["num_spots"]))
    print("dx_um:", dx_um)
    print("pixel_size_um:", pixel_size_um)
    print("fwhm_um:", float(PARAMS["fwhm_um"]))
    print("sigma_pix:", sigma_pix)
    print("gain_std:", float(PARAMS["gain_std"]))
    print("sidelobe_alpha:", float(PARAMS["sidelobe_alpha"]))
    print("Y_obs_norm output shape:", (n, T, L))
    print("gap:", gap)
    print("upper-bound peak expected counts per band:", expected_peak_counts_upper)

    A_npy_path = out_dir / f"A_bbbc021_3ch_{n}cells_128x128x3.npy"
    Y_npy_path = out_dir / f"Y_obs_norm_bbbc021_lase_3ch_{n}cells_44spots_Tx{L}.npy"

    print("\nCreating A memmap:")
    print(A_npy_path)
    A_map = np.lib.format.open_memmap(A_npy_path, mode="w+", dtype=np.float32, shape=A.shape)
    A_map[:] = A[:]
    A_map.flush()

    print("Creating Y_obs_norm memmap:")
    print(Y_npy_path)
    Y_map = np.lib.format.open_memmap(Y_npy_path, mode="w+", dtype=SAVE_DTYPE, shape=(n, T, L))

    preview_A0 = None
    preview_X0 = None
    preview_Y_clean0 = None
    preview_Y_obs_norm0 = None

    y_min = float("inf")
    y_max = float("-inf")
    clean_min = float("inf")
    clean_max = float("-inf")

    for i in range(n):
        A_cell = A[i]

        # X_cell: [H, W, L] = sum_d A_d * S_d(lambda)
        X_cell = np.einsum("hwd,dl->hwl", A_cell, S).astype(np.float32)
        X_cell = np.clip(X_cell, 0, None)

        cols = sample_lase_columns(X_cell, PARAMS, rng)
        waveform_clean, gap_used = serialize_columns(cols, PARAMS)

        if gap_used != gap:
            raise RuntimeError(f"Gap mismatch: main gap={gap}, serialize gap={gap_used}")
        if waveform_clean.shape != (T, L):
            raise RuntimeError(f"Clean waveform shape mismatch: got {waveform_clean.shape}, expected {(T, L)}")

        waveform_obs = add_poisson_noise(waveform_clean, PARAMS, rng)
        waveform_obs_norm = normalize_observed_counts(waveform_obs, PARAMS)

        if waveform_obs_norm.shape != (T, L):
            raise RuntimeError(f"Observed waveform shape mismatch: got {waveform_obs_norm.shape}, expected {(T, L)}")
        if not np.all(np.isfinite(waveform_obs_norm)):
            raise RuntimeError(f"Non-finite Y_obs_norm found at cell index {i}")

        Y_map[i] = waveform_obs_norm.astype(SAVE_DTYPE)

        y_min = min(y_min, float(waveform_obs_norm.min()))
        y_max = max(y_max, float(waveform_obs_norm.max()))
        clean_min = min(clean_min, float(waveform_clean.min()))
        clean_max = max(clean_max, float(waveform_clean.max()))

        if i == 0:
            preview_A0 = A_cell.copy()
            preview_X0 = X_cell.copy()
            preview_Y_clean0 = waveform_clean.copy()
            preview_Y_obs_norm0 = waveform_obs_norm.copy()

        if (i + 1) % 100 == 0 or i + 1 == n:
            Y_map.flush()
            print(
                f"processed {i + 1}/{n} | "
                f"Y_obs_norm min/max: {y_min:.6f}, {y_max:.6f} | "
                f"clean min/max: {clean_min:.6f}, {clean_max:.6f}"
            )

    Y_map.flush()

    metadata_out_path = out_dir / f"bbbc021_lase_train_ready_metadata_{n}cells_44spots.npz"

    source_metadata_params = None
    if "params" in meta.files:
        try:
            source_metadata_params = json.loads(_as_str(meta["params"]))
        except Exception:
            source_metadata_params = _as_str(meta["params"])

    np.savez_compressed(
        metadata_out_path,
        A_npy_path=np.array(str(A_npy_path), dtype=object),
        Y_obs_norm_npy_path=np.array(str(Y_npy_path), dtype=object),
        S=S,
        S_raw=meta["S_raw"].astype(np.float32) if "S_raw" in meta.files else S,
        similarity=similarity,
        pearson_similarity=meta["pearson_similarity"].astype(np.float32) if "pearson_similarity" in meta.files else np.corrcoef(S).astype(np.float32),
        wavelengths=wavelengths,
        split=split,
        cell_ids=cell_ids,
        channel_names=channel_names,
        source_A_file=np.array(str(source_A_file), dtype=object),
        source_metadata_file=np.array(str(metadata_path), dtype=object),
        source_metadata_params=json.dumps(source_metadata_params, ensure_ascii=False),
        A_shape=np.array(A.shape, dtype=np.int64),
        Y_obs_norm_shape=np.array((n, T, L), dtype=np.int64),
        gap=np.array(gap, dtype=np.int32),
        dx_um=np.array(dx_um, dtype=np.float32),
        pixel_size_um=np.array(pixel_size_um, dtype=np.float32),
        sigma_pix=np.array(sigma_pix, dtype=np.float32),
        params=json.dumps(PARAMS, ensure_ascii=False),
        Y_obs_norm_min=np.array(y_min, dtype=np.float32),
        Y_obs_norm_max=np.array(y_max, dtype=np.float32),
        Y_clean_min=np.array(clean_min, dtype=np.float32),
        Y_clean_max=np.array(clean_max, dtype=np.float32),
    )

    print("\nSaved train-ready metadata:")
    print(metadata_out_path)
    print("Saved A npy:")
    print(A_npy_path)
    print("Saved Y_obs_norm npy:")
    print(Y_npy_path)
    print("Y_obs_norm min/max:", y_min, y_max)
    print("Y_clean min/max:", clean_min, clean_max)

    summary = {
        "metadata_file": str(metadata_out_path),
        "A_npy_path": str(A_npy_path),
        "Y_obs_norm_npy_path": str(Y_npy_path),
        "A_shape": list(A.shape),
        "Y_obs_norm_shape": [int(n), int(T), int(L)],
        "S_shape": list(S.shape),
        "S_row_sums": S.sum(axis=1).astype(float).tolist(),
        "S_row_max": S.max(axis=1).astype(float).tolist(),
        "gap": int(gap),
        "dx_um": float(dx_um),
        "pixel_size_um": float(pixel_size_um),
        "sigma_pix": float(sigma_pix),
        "Y_obs_norm_min": float(y_min),
        "Y_obs_norm_max": float(y_max),
        "Y_clean_min": float(clean_min),
        "Y_clean_max": float(clean_max),
        "params": PARAMS,
        "note": (
            "Train-ready BBBC021 LASE dataset. A and Y_obs_norm are saved as .npy arrays. "
            "Y_obs_norm is background-subtracted and divided by peak; negative values are preserved."
        ),
    }

    summary_path = out_dir / "bbbc021_lase_train_ready_summary_44spots.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved summary:")
    print(summary_path)

    preview_path = result_dir / "bbbc021_lase_train_ready_preview_44spots.png"
    save_preview(
        A0=preview_A0,
        X0=preview_X0,
        Y_clean0=preview_Y_clean0,
        Y_obs_norm0=preview_Y_obs_norm0,
        gap=gap,
        wavelengths=wavelengths,
        out_path=preview_path,
    )


if __name__ == "__main__":
    main()
