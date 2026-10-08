from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

bbbc_root = Path(__file__).resolve().parent

input_dir = bbbc_root / "data" / "processed" / "bbbc021_single_cell_patches"

out_dir = bbbc_root / "data" / "processed" / "bbbc021_spectral_metadata"
out_dir.mkdir(parents=True, exist_ok=True)

result_dir = bbbc_root / "results" / "bbbc021_spectral_metadata_preview"
result_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Parameters
# ============================================================

# This step builds a simulated multispectral fluorescence library.
# 16 bands from 450 to 750 nm gives 20 nm sampling interval.
# With Gaussian sigma values of 28-38 nm, each emission peak is sampled by
# about 3-4 bands across its FWHM, which is sufficient for a first simulation.
NUM_BANDS = 16
WAVELENGTH_MIN = 450.0
WAVELENGTH_MAX = 750.0

# IMPORTANT:
# For abundance inversion, unit-sum spectra are more physically interpretable
# than peak-normalized spectra. If A[p, k] is the amount/abundance of dye k,
# then sum_l A[p, k] * S[k, l] = A[p, k]. This makes each dye abundance
# correspond to the total spectral photon contribution across all bands.
# The alternative "peak" option is kept only for ablation/backward checks.
SPECTRAL_NORMALIZATION = "unit_sum"  # choices: "unit_sum", "peak", "none"

# Simulated overlapping fluorescence spectra.
# The first two dyes mimic SYTO16 / DIA-like spectral overlap.
# The third dye is a far-red reference channel to keep NUM_DYES = 3.
DYE_SPECS = [
    {
        "name": "SYTO16_like",
        "peak_nm": 518.0,
        "sigma_left_nm": 28.0,
        "sigma_right_nm": 65.0,
    },
    {
        "name": "DIA_like",
        "peak_nm": 613.0,
        "sigma_left_nm": 70.0,
        "sigma_right_nm": 58.0,
    },
    {
        "name": "DeepRed_like",
        "peak_nm": 665.0,
        "sigma_left_nm": 45.0,
        "sigma_right_nm": 50.0,
    },
]

RANDOM_SEED = 20260517

# Optional: set to a specific npz file if you do not want to use the latest one.
# Example:
# INPUT_A_FILE = input_dir / "A_bbbc021_single_cell_1234cells_128x128x3.npz"
INPUT_A_FILE = None

EPS = 1e-8


# ============================================================
# Functions
# ============================================================

def asymmetric_gaussian_spectrum(wavelengths, peak, sigma_left, sigma_right):
    """
    Create an asymmetric nonnegative fluorescence-like emission spectrum.

    wavelengths: [L]
    peak: emission peak wavelength
    sigma_left: spectral width on the short-wavelength side
    sigma_right: spectral width on the long-wavelength side
    """
    wavelengths = wavelengths.astype(np.float32)

    sigma = np.where(
        wavelengths < float(peak),
        float(sigma_left),
        float(sigma_right),
    ).astype(np.float32)

    s = np.exp(-0.5 * ((wavelengths - float(peak)) / sigma) ** 2)
    return s.astype(np.float32)


def normalize_spectra(S, mode="unit_sum"):
    """
    Normalize spectra row-wise.

    S shape:
        [num_dyes, num_bands]

    mode:
        "unit_sum": each row sums to 1 across spectral bands.
                    This is the preferred convention for abundance recovery.
        "peak":     each row has max value 1. Useful for visualization only.
        "none":     no normalization.
    """
    S = np.asarray(S, dtype=np.float32)
    S = np.clip(S, 0, None)

    if mode == "unit_sum":
        denom = S.sum(axis=1, keepdims=True)
        S = S / (denom + EPS)
    elif mode == "peak":
        denom = S.max(axis=1, keepdims=True)
        S = S / (denom + EPS)
    elif mode == "none":
        pass
    else:
        raise ValueError(
            f"Unknown SPECTRAL_NORMALIZATION={mode!r}. "
            "Use 'unit_sum', 'peak', or 'none'."
        )

    if not np.all(np.isfinite(S)):
        raise ValueError("S contains NaN or Inf after normalization.")

    return S.astype(np.float32)


def cosine_similarity_matrix(S):
    """
    Nonnegative spectral overlap score in [0, 1] for nonnegative spectra.

    This is more appropriate than Pearson correlation for future
    correlation-aware sparse penalties because it does not mean-center spectra.
    """
    S = np.asarray(S, dtype=np.float32)
    denom = np.linalg.norm(S, axis=1, keepdims=True) + EPS
    U = S / denom
    C = U @ U.T
    C = np.clip(C, 0.0, 1.0)
    return C.astype(np.float32)


def pearson_corr_matrix(S):
    """
    Pearson correlation is saved for diagnostics, but cosine similarity is used
    as the default similarity matrix for nonnegative fluorescence spectra.
    """
    C = np.corrcoef(S).astype(np.float32)
    C = np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
    return C.astype(np.float32)


def make_three_dye_spectral_library(wavelengths):
    """
    Build a three-dye simulated spectral library.

    Channel meaning in this simulation:
        0: SYTO16-like overlapping dye
        1: DIA-like overlapping dye
        2: DeepRed-like reference dye
    """
    spectra = []

    for spec in DYE_SPECS:
        spectra.append(
            asymmetric_gaussian_spectrum(
                wavelengths,
                peak=spec["peak_nm"],
                sigma_left=spec["sigma_left_nm"],
                sigma_right=spec["sigma_right_nm"],
            )
        )

    S_raw = np.stack(spectra, axis=0).astype(np.float32)
    S = normalize_spectra(S_raw, mode=SPECTRAL_NORMALIZATION)

    return S.astype(np.float32), S_raw.astype(np.float32)


def make_split(n, seed=20260517):
    rng = np.random.default_rng(seed)

    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    split = np.array(["test"] * n, dtype="<U5")
    split[idx[:n_train]] = "train"
    split[idx[n_train:n_train + n_val]] = "val"
    split[idx[n_train + n_val:]] = "test"

    return split


def find_latest_single_cell_file(input_dir):
    files = sorted(input_dir.glob("A_bbbc021_single_cell_*cells_128x128x3.npz"))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No A_bbbc021_single_cell_*cells_128x128x3.npz found in {input_dir}"
        )

    files = sorted(files, key=lambda p: p.stat().st_mtime)

    return files[-1]


def validate_A(A):
    if A.ndim != 4 or A.shape[-1] != 3:
        raise ValueError(f"Expected A shape [N, 128, 128, 3], got {A.shape}")

    if A.shape[0] <= 0:
        raise ValueError("A contains zero cells.")

    if not np.all(np.isfinite(A)):
        raise ValueError("A contains NaN or Inf.")

    if A.min() < -1e-6:
        raise ValueError(f"A contains negative values: min={A.min()}")

    # The previous segmentation script robust-normalized A to roughly [0, 1].
    # Do not hard-fail if it is slightly above 1, but report it.
    if A.max() > 1.05:
        print(f"Warning: A max is {A.max():.4f}, expected roughly <= 1.")


def validate_wavelengths(wavelengths):
    if wavelengths.ndim != 1 or wavelengths.size != NUM_BANDS:
        raise ValueError(
            f"Expected wavelengths shape ({NUM_BANDS},), got {wavelengths.shape}"
        )

    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError("wavelengths must be strictly increasing.")

    spacing = np.diff(wavelengths)
    if not np.allclose(spacing, spacing[0]):
        print("Warning: wavelength spacing is not uniform.")


def validate_S(S):
    if S.shape != (len(DYE_SPECS), NUM_BANDS):
        raise ValueError(
            f"Expected S shape {(len(DYE_SPECS), NUM_BANDS)}, got {S.shape}"
        )

    if not np.all(np.isfinite(S)):
        raise ValueError("S contains NaN or Inf.")

    if np.min(S) < -1e-8:
        raise ValueError("S contains negative values.")

    if SPECTRAL_NORMALIZATION == "unit_sum":
        row_sums = S.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError(f"Unit-sum normalization failed. Row sums: {row_sums}")


def save_preview(A0, S, wavelengths, out_path):
    """
    Save a compact spectral-metadata preview.

    This preview keeps:
        1. three-dye spectral library S(d, l)
        2. pseudo-color RGB composite from simulated X bands

    Note:
        Pseudo RGB from X is only a visualization.
        It is not a real microscope RGB image.
    """

    X0 = np.einsum("hwd,dl->hwl", A0, S).astype(np.float32)
    X0 = np.clip(X0, 0, None)

    def nearest_band(target_nm):
        """Return index of spectral band closest to target wavelength."""
        return int(np.argmin(np.abs(wavelengths - target_nm)))

    b_dna = nearest_band(DYE_SPECS[0]["peak_nm"])
    b_tubulin = nearest_band(DYE_SPECS[1]["peak_nm"])
    b_actin = nearest_band(DYE_SPECS[2]["peak_nm"])

    def robust_norm(img, percentile=99.5):
        img = img.astype(np.float32)
        vmax = np.percentile(img, percentile)
        if vmax <= EPS:
            vmax = img.max() + EPS
        img = img / (vmax + EPS)
        img = np.clip(img, 0, 1)
        return img

    rgb = np.stack(
        [
            robust_norm(X0[:, :, b_actin]),
            robust_norm(X0[:, :, b_tubulin]),
            robust_norm(X0[:, :, b_dna]),
        ],
        axis=-1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    ax = axes[0]
    labels = [spec["name"] for spec in DYE_SPECS]
    for k, label in enumerate(labels):
        ax.plot(wavelengths, S[k], marker="o", label=label)

    for b in [b_dna, b_tubulin, b_actin]:
        ax.axvline(wavelengths[b], linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Wavelength (nm)")
    if SPECTRAL_NORMALIZATION == "unit_sum":
        ax.set_ylabel("Spectral fraction per band")
    elif SPECTRAL_NORMALIZATION == "peak":
        ax.set_ylabel("Peak-normalized response")
    else:
        ax.set_ylabel("Raw response")

    ax.set_title(
        f"3-dye spectral library\n"
        f"{NUM_BANDS} bands, {WAVELENGTH_MIN:.0f}-{WAVELENGTH_MAX:.0f} nm, "
        f"norm={SPECTRAL_NORMALIZATION}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.imshow(rgb, vmin=0, vmax=1)
    ax.set_title(
        "Pseudo-color composite from simulated X\n"
        f"R={wavelengths[b_actin]:.0f} nm, "
        f"G={wavelengths[b_tubulin]:.0f} nm, "
        f"B={wavelengths[b_dna]:.0f} nm"
    )
    ax.axis("off")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved preview:")
    print(out_path)


def main():
    if INPUT_A_FILE is None:
        in_path = find_latest_single_cell_file(input_dir)
    else:
        in_path = Path(INPUT_A_FILE)
        if not in_path.exists():
            raise FileNotFoundError(f"INPUT_A_FILE does not exist: {in_path}")

    print("Loading single-cell patch file:")
    print(in_path)

    data = np.load(in_path, allow_pickle=True)
    if "A" not in data.files:
        raise KeyError(f"Input file does not contain key 'A': {in_path}")

    A = data["A"].astype(np.float32)
    validate_A(A)
    n = A.shape[0]

    wavelengths = np.linspace(
        WAVELENGTH_MIN,
        WAVELENGTH_MAX,
        NUM_BANDS,
        dtype=np.float32,
    )
    validate_wavelengths(wavelengths)

    S, S_raw = make_three_dye_spectral_library(wavelengths)
    validate_S(S)

    # Use cosine similarity as the default nonnegative spectral overlap matrix.
    similarity = cosine_similarity_matrix(S)
    pearson_similarity = pearson_corr_matrix(S)

    split = make_split(n, seed=RANDOM_SEED)
    cell_ids = np.array([f"bbbc021_cell_{i:05d}" for i in range(n)], dtype=object)
    channel_names = np.array([spec["name"] for spec in DYE_SPECS], dtype=object)

    out_path = out_dir / f"bbbc021_3ch_spectral_metadata_{n}cells.npz"

    params = {
        "num_bands": NUM_BANDS,
        "wavelength_min": WAVELENGTH_MIN,
        "wavelength_max": WAVELENGTH_MAX,
        "wavelength_spacing_nm": float(wavelengths[1] - wavelengths[0]) if NUM_BANDS > 1 else None,
        "spectral_normalization": SPECTRAL_NORMALIZATION,
        "random_seed": RANDOM_SEED,
        "dye_specs": DYE_SPECS,
        "dye_channels": [spec["name"] for spec in DYE_SPECS],
        "similarity_default": "cosine_similarity",
        "note": (
            "Large-data metadata only. Full X is not saved. "
            "S is unit-sum by default so each abundance value corresponds "
            "to total spectral contribution across bands."
        ),
    }

    np.savez_compressed(
        out_path,
        S=S,
        S_raw=S_raw,
        similarity=similarity,
        cosine_similarity=similarity,
        pearson_similarity=pearson_similarity,
        wavelengths=wavelengths,
        split=split,
        cell_ids=cell_ids,
        source_A_file=np.array(str(in_path), dtype=object),
        A_shape=np.array(A.shape, dtype=np.int64),
        channel_names=channel_names,
        params=json.dumps(params),
    )

    print("\nSaved metadata:")
    print(out_path)
    print("A shape:", A.shape)
    print("A min/max:", float(A.min()), float(A.max()))
    print("S shape:", S.shape)
    print("S normalization:", SPECTRAL_NORMALIZATION)
    print("S row sums:", S.sum(axis=1))
    print("S row max:", S.max(axis=1))
    print("wavelengths:")
    print(wavelengths)
    print("similarity shape:", similarity.shape)
    print("split counts:")
    print("  train:", int(np.sum(split == "train")))
    print("  val:", int(np.sum(split == "val")))
    print("  test:", int(np.sum(split == "test")))
    print("S:")
    print(S)
    print("cosine similarity:")
    print(similarity)
    print("pearson similarity:")
    print(pearson_similarity)

    preview_path = result_dir / "bbbc021_spectral_metadata_preview.png"
    save_preview(A[0], S, wavelengths, preview_path)


if __name__ == "__main__":
    main()
