"""
Find abnormal samples in the practical 44-spot LASE abundance recovery experiment.

This script does NOT remove any data. It only reports suspicious sample IDs for
manual inspection, especially:
  1) very large per-dye NRMSE values;
  2) very low abundance-map Pearson correlations.

Outputs are saved to:
    results_main_exp_practical_lase_44spots/outlier_diagnostics_44spots/

Run:
    python 27_find_abnormal_samples_practical_44spots.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
# User settings
# ============================================================

BBBC_ROOT = Path(__file__).resolve().parent
OUT_DIR = BBBC_ROOT / "results_main_exp_practical_lase_44spots"
DIAG_DIR = OUT_DIR / "outlier_diagnostics_44spots"
PREVIEW_DIR = DIAG_DIR / "outlier_previews"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

DYE_NAMES = ["SYTO16-like", "DIA-like", "DeepRed-like"]
EPS = 1e-8
FIG_DPI = 220

# We report both percentile-based and top-K outliers.
NRMSE_PERCENTILE = 99.0
TOP_K_NRMSE = 50
TOP_K_LOW_PEARSON = 50

# Optional absolute criterion for very low Pearson. Used only for flagging.
LOW_PEARSON_THRESHOLD = 0.90

# Number of unique suspicious samples for which preview figures are saved.
MAX_PREVIEW_SAMPLES = 30

METHODS = [
    {
        "key": "ru_nnls",
        "label": "RU-NNLS",
        "metrics_glob": "results/bbbc021_two_step_decode_then_unmix_44spots_practical/test_two_step_nnls_active_set/metrics_two_step_nnls_active_set_test_*samples.json",
    },
    {
        "key": "ru_poisson",
        "label": "RU-Poisson",
        "metrics_glob": "results/bbbc021_two_step_decode_then_unmix_44spots_practical/test_two_step_nnls_init_poisson_kl/metrics_two_step_nnls_init_poisson_kl_test_*samples.json",
    },
    {
        "key": "joint",
        "label": "SA-Joint",
        "metrics_glob": "results/bbbc021_channel_competition_sparse_44spots/test_joint_spectral_ambiguity_tv_final_nosparse_ambtv_2em04_p_1e00/metrics_test_*samples.json",
    },
]


# ============================================================
# Loading utilities
# ============================================================

def _as_python_string(x):
    if hasattr(x, "item"):
        try:
            x = x.item()
        except Exception:
            pass
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.bytes_):
        return x.tobytes().decode("utf-8")
    return str(x)


def _latest_match(pattern: str) -> Path:
    matches = sorted(BBBC_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No file matched pattern: {pattern}")
    return matches[-1]


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if p.exists():
        return p
    # If an old absolute path is stored in JSON/metadata, locate by basename.
    candidates = list(BBBC_ROOT.rglob(p.name))
    if candidates:
        return sorted(candidates, key=lambda x: x.stat().st_mtime)[-1]
    raise FileNotFoundError(f"Cannot resolve path: {p}")


def _load_method(method_info: dict) -> dict:
    metrics_path = _latest_match(method_info["metrics_glob"])
    summary = _load_json(metrics_path)

    a_hat_path = _resolve_path(summary["A_hat_path"])
    real_idx_path = _resolve_path(summary["real_indices_path"])
    metadata_path = _resolve_path(summary["metadata_path"])

    A_hat = np.load(a_hat_path, mmap_mode="r")          # [N, 3, H, W]
    real_indices = np.load(real_idx_path).astype(int)   # indices in full A array

    meta = np.load(metadata_path, allow_pickle=True)
    A_true_path = _resolve_path(str(meta["A_npy_path"].item()))
    A_true_full = np.load(A_true_path, mmap_mode="r")   # [N_all, H, W, 3]
    A_true = np.asarray(A_true_full[real_indices], dtype=np.float32).transpose(0, 3, 1, 2)
    A_hat = np.asarray(A_hat, dtype=np.float32)

    if A_hat.shape != A_true.shape:
        raise ValueError(
            f"Shape mismatch for {method_info['label']}: "
            f"A_hat {A_hat.shape}, A_true {A_true.shape}"
        )

    # Optional cell IDs saved by the metadata builder. These help trace the
    # cropped cell back to the original patch/source.
    if "cell_ids" in meta.files:
        cell_ids_full = np.array([_as_python_string(x) for x in meta["cell_ids"]], dtype=object)
        cell_ids = cell_ids_full[real_indices]
    else:
        cell_ids = np.array(["" for _ in real_indices], dtype=object)

    return {
        "key": method_info["key"],
        "label": method_info["label"],
        "metrics_path": metrics_path,
        "A_hat_path": a_hat_path,
        "real_idx_path": real_idx_path,
        "metadata_path": metadata_path,
        "A_hat": A_hat,
        "A_true": A_true,
        "real_indices": real_indices,
        "cell_ids": cell_ids,
    }


def _align_methods(method_data: list[dict]) -> list[dict]:
    sets = [set(map(int, m["real_indices"])) for m in method_data]
    common = set.intersection(*sets)
    if not common:
        raise RuntimeError("No common real_indices across methods.")

    # Use SA-Joint order as reference if available.
    joint = next((m for m in method_data if m["key"] == "joint"), method_data[0])
    ordered_common = [int(x) for x in joint["real_indices"] if int(x) in common]

    aligned = []
    for m in method_data:
        idx_map = {int(r): i for i, r in enumerate(m["real_indices"])}
        order = np.array([idx_map[r] for r in ordered_common], dtype=int)
        m2 = dict(m)
        m2["A_hat"] = np.asarray(m["A_hat"])[order]
        m2["A_true"] = np.asarray(m["A_true"])[order]
        m2["real_indices"] = np.asarray(m["real_indices"])[order]
        m2["cell_ids"] = np.asarray(m["cell_ids"], dtype=object)[order]
        aligned.append(m2)

    return aligned


# ============================================================
# Metrics
# ============================================================

def compute_true_rms(A_true: np.ndarray) -> np.ndarray:
    """A_true: [N, D, H, W]. Return [N, D]."""
    return np.sqrt(np.mean(A_true ** 2, axis=(2, 3)) + EPS)


def compute_per_dye_rmse(A_hat: np.ndarray, A_true: np.ndarray) -> np.ndarray:
    """A_hat, A_true: [N, D, H, W]. Return [N, D]."""
    return np.sqrt(np.mean((A_hat - A_true) ** 2, axis=(2, 3)) + EPS)


def compute_per_dye_nrmse(A_hat: np.ndarray, A_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rmse = compute_per_dye_rmse(A_hat, A_true)
    true_rms = compute_true_rms(A_true)
    nrmse = rmse / (true_rms + EPS)
    return nrmse, rmse, true_rms


def compute_per_dye_pearson(A_hat: np.ndarray, A_true: np.ndarray) -> np.ndarray:
    """Return per-cell per-dye Pearson: [N, D]."""
    N, D = A_true.shape[:2]
    out = np.full((N, D), np.nan, dtype=np.float32)

    for i in range(N):
        for d in range(D):
            x = A_true[i, d].reshape(-1)
            y = A_hat[i, d].reshape(-1)
            if np.std(x) < EPS or np.std(y) < EPS:
                continue
            r = np.corrcoef(x, y)[0, 1]
            if np.isfinite(r):
                out[i, d] = float(r)
    return out


def _safe_vmax(*imgs, percentile=99.5, minimum=1e-3):
    vals = []
    for img in imgs:
        arr = np.asarray(img)
        if arr.size:
            vals.append(float(np.percentile(arr, percentile)))
    vmax = max(vals) if vals else minimum
    return max(vmax, minimum)


# ============================================================
# Diagnostics tables
# ============================================================

def build_metric_tables(methods: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    true_ref = methods["joint"]["A_true"]
    real_indices = methods["joint"]["real_indices"]
    cell_ids = methods["joint"]["cell_ids"]

    rows_dye = []
    rows_cell = []

    for key, m in methods.items():
        nrmse, rmse, true_rms = compute_per_dye_nrmse(m["A_hat"], true_ref)
        pearson_dye = compute_per_dye_pearson(m["A_hat"], true_ref)
        pearson_overall = np.nanmean(pearson_dye, axis=1)

        for i, ridx in enumerate(real_indices):
            rows_cell.append({
                "real_idx": int(ridx),
                "cell_id": _as_python_string(cell_ids[i]),
                "method_key": key,
                "method": m["label"],
                "pearson_overall": float(pearson_overall[i]) if np.isfinite(pearson_overall[i]) else np.nan,
                "min_per_dye_pearson": float(np.nanmin(pearson_dye[i])) if np.any(np.isfinite(pearson_dye[i])) else np.nan,
                "max_per_dye_nrmse": float(np.nanmax(nrmse[i])) if np.any(np.isfinite(nrmse[i])) else np.nan,
                "true_rms_SYTO16_like": float(true_rms[i, 0]),
                "true_rms_DIA_like": float(true_rms[i, 1]),
                "true_rms_DeepRed_like": float(true_rms[i, 2]),
            })

            for d, dye in enumerate(DYE_NAMES):
                rows_dye.append({
                    "real_idx": int(ridx),
                    "cell_id": _as_python_string(cell_ids[i]),
                    "method_key": key,
                    "method": m["label"],
                    "dye_index": d,
                    "dye": dye,
                    "nrmse": float(nrmse[i, d]),
                    "rmse": float(rmse[i, d]),
                    "true_rms": float(true_rms[i, d]),
                    "pearson_dye": float(pearson_dye[i, d]) if np.isfinite(pearson_dye[i, d]) else np.nan,
                    "true_mean": float(np.mean(true_ref[i, d])),
                    "true_max": float(np.max(true_ref[i, d])),
                    "hat_mean": float(np.mean(m["A_hat"][i, d])),
                    "hat_max": float(np.max(m["A_hat"][i, d])),
                })

    df_dye = pd.DataFrame(rows_dye)
    df_cell = pd.DataFrame(rows_cell)
    return df_dye, df_cell


def select_outliers(df_dye: pd.DataFrame, df_cell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # High per-dye NRMSE cases.
    finite_nrmse = df_dye[np.isfinite(df_dye["nrmse"])].copy()
    nrmse_cut = float(np.nanpercentile(finite_nrmse["nrmse"].to_numpy(), NRMSE_PERCENTILE))

    high_nrmse_p99 = finite_nrmse[finite_nrmse["nrmse"] >= nrmse_cut].copy()
    high_nrmse_top = finite_nrmse.sort_values("nrmse", ascending=False).head(TOP_K_NRMSE).copy()
    high_nrmse = (
        pd.concat([high_nrmse_p99, high_nrmse_top], ignore_index=True)
        .drop_duplicates(["real_idx", "method_key", "dye_index"])
        .sort_values("nrmse", ascending=False)
    )
    high_nrmse["outlier_type"] = "high_per_dye_nrmse"
    high_nrmse["nrmse_p99_cutoff"] = nrmse_cut

    # Low Pearson cases, both bottom-K and below absolute threshold.
    finite_pearson = df_cell[np.isfinite(df_cell["pearson_overall"])].copy()
    low_pearson_thresholded = finite_pearson[finite_pearson["pearson_overall"] <= LOW_PEARSON_THRESHOLD].copy()
    low_pearson_top = finite_pearson.sort_values("pearson_overall", ascending=True).head(TOP_K_LOW_PEARSON).copy()
    low_pearson = (
        pd.concat([low_pearson_thresholded, low_pearson_top], ignore_index=True)
        .drop_duplicates(["real_idx", "method_key"])
        .sort_values("pearson_overall", ascending=True)
    )
    low_pearson["outlier_type"] = "low_pearson"
    low_pearson["low_pearson_threshold"] = LOW_PEARSON_THRESHOLD

    # Unique suspicious sample IDs.
    nrmse_ids = set(map(int, high_nrmse["real_idx"].to_list()))
    pearson_ids = set(map(int, low_pearson["real_idx"].to_list()))
    all_ids = sorted(nrmse_ids | pearson_ids)

    unique_rows = []
    for ridx in all_ids:
        sub_dye = df_dye[df_dye["real_idx"] == ridx]
        sub_cell = df_cell[df_cell["real_idx"] == ridx]
        cell_id = ""
        if len(sub_dye) > 0:
            cell_id = _as_python_string(sub_dye.iloc[0]["cell_id"])
        unique_rows.append({
            "real_idx": int(ridx),
            "cell_id": cell_id,
            "flag_high_nrmse": int(ridx in nrmse_ids),
            "flag_low_pearson": int(ridx in pearson_ids),
            "max_nrmse_all_methods_dyes": float(np.nanmax(sub_dye["nrmse"].to_numpy())) if len(sub_dye) else np.nan,
            "min_pearson_all_methods": float(np.nanmin(sub_cell["pearson_overall"].to_numpy())) if len(sub_cell) else np.nan,
        })

    unique = pd.DataFrame(unique_rows).sort_values(
        ["flag_high_nrmse", "flag_low_pearson", "max_nrmse_all_methods_dyes"],
        ascending=[False, False, False],
    )

    return high_nrmse, low_pearson, unique


# ============================================================
# Preview figures
# ============================================================

def save_outlier_preview(methods: dict[str, dict], row_idx: int, real_idx: int, cell_id: str, reason: str):
    true = methods["joint"]["A_true"][row_idx]  # [3, H, W]
    method_order = ["ru_nnls", "ru_poisson", "joint"]

    # 7 rows x 3 dyes: True, three estimates, three absolute errors.
    fig = plt.figure(figsize=(7.2, 14.0), dpi=FIG_DPI)
    gs = GridSpec(
        7, 3,
        figure=fig,
        left=0.06,
        right=0.99,
        bottom=0.03,
        top=0.94,
        wspace=0.05,
        hspace=0.08,
    )

    row_labels = [
        "True",
        "RU-NNLS",
        "RU-Poisson",
        "Joint",
        "|Err| RU-NNLS",
        "|Err| RU-Poisson",
        "|Err| Joint",
    ]

    abundance_vmax = []
    error_vmax = []
    for d in range(3):
        abundance_imgs = [true[d]] + [methods[k]["A_hat"][row_idx, d] for k in method_order]
        error_imgs = [np.abs(methods[k]["A_hat"][row_idx, d] - true[d]) for k in method_order]
        abundance_vmax.append(_safe_vmax(*abundance_imgs, percentile=99.5))
        error_vmax.append(_safe_vmax(*error_imgs, percentile=99.5))

    for r in range(7):
        for d in range(3):
            ax = fig.add_subplot(gs[r, d])

            if r == 0:
                img = true[d]
                ax.imshow(img, cmap="gray", vmin=0, vmax=abundance_vmax[d], interpolation="nearest")
            elif r in [1, 2, 3]:
                key = method_order[r - 1]
                img = methods[key]["A_hat"][row_idx, d]
                ax.imshow(img, cmap="gray", vmin=0, vmax=abundance_vmax[d], interpolation="nearest")
            else:
                key = method_order[r - 4]
                img = np.abs(methods[key]["A_hat"][row_idx, d] - true[d])
                ax.imshow(img, cmap="magma", vmin=0, vmax=error_vmax[d], interpolation="nearest")

            if r == 0:
                ax.set_title(DYE_NAMES[d], fontsize=10, pad=4)
            if d == 0:
                ax.set_ylabel(row_labels[r], fontsize=9, rotation=90, labelpad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle(
        f"Outlier preview: real_idx={real_idx}\ncell_id={cell_id}\nreason={reason}",
        fontsize=11,
        y=0.985,
    )

    safe_cell = str(cell_id).replace("/", "_").replace("\\", "_").replace(":", "_")[:80]
    out_path = PREVIEW_DIR / f"outlier_realidx_{int(real_idx)}_{safe_cell}.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    return out_path


def save_previews_for_unique_outliers(methods: dict[str, dict], unique: pd.DataFrame):
    idx_map = {int(r): i for i, r in enumerate(methods["joint"]["real_indices"])}
    preview_rows = []

    for _, row in unique.head(MAX_PREVIEW_SAMPLES).iterrows():
        ridx = int(row["real_idx"])
        if ridx not in idx_map:
            continue
        reason_parts = []
        if int(row["flag_high_nrmse"]):
            reason_parts.append("high_NRMSE")
        if int(row["flag_low_pearson"]):
            reason_parts.append("low_Pearson")
        reason = "+".join(reason_parts) if reason_parts else "unknown"
        out_path = save_outlier_preview(
            methods=methods,
            row_idx=idx_map[ridx],
            real_idx=ridx,
            cell_id=_as_python_string(row.get("cell_id", "")),
            reason=reason,
        )
        preview_rows.append({
            "real_idx": ridx,
            "cell_id": _as_python_string(row.get("cell_id", "")),
            "preview_path": str(out_path),
            "reason": reason,
        })

    preview_df = pd.DataFrame(preview_rows)
    preview_csv = DIAG_DIR / "outlier_preview_index.csv"
    preview_df.to_csv(preview_csv, index=False)
    return preview_csv


# ============================================================
# Main
# ============================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("Loading method outputs...")
    loaded = [_load_method(m) for m in METHODS]
    loaded = _align_methods(loaded)
    methods = {m["key"]: m for m in loaded}

    print("Common samples:", len(methods["joint"]["real_indices"]))
    for key, m in methods.items():
        print(f"  {m['label']}: A_hat={m['A_hat'].shape}, metrics={m['metrics_path']}")

    print("\nComputing raw diagnostic metrics...")
    df_dye, df_cell = build_metric_tables(methods)

    long_dye_csv = DIAG_DIR / "all_per_dye_raw_metrics_44spots.csv"
    cell_csv = DIAG_DIR / "all_per_cell_raw_metrics_44spots.csv"
    df_dye.to_csv(long_dye_csv, index=False)
    df_cell.to_csv(cell_csv, index=False)

    print("Selecting suspicious samples...")
    high_nrmse, low_pearson, unique = select_outliers(df_dye, df_cell)

    high_nrmse_csv = DIAG_DIR / "outliers_high_per_dye_nrmse_44spots.csv"
    low_pearson_csv = DIAG_DIR / "outliers_low_pearson_44spots.csv"
    unique_csv = DIAG_DIR / "outlier_unique_sample_ids_44spots.csv"

    high_nrmse.to_csv(high_nrmse_csv, index=False)
    low_pearson.to_csv(low_pearson_csv, index=False)
    unique.to_csv(unique_csv, index=False)

    print("Saving preview figures...")
    preview_csv = save_previews_for_unique_outliers(methods, unique)

    print("\n===== Saved diagnostic outputs =====")
    print("All per-dye raw metrics:", long_dye_csv)
    print("All per-cell raw metrics:", cell_csv)
    print("High per-dye NRMSE outliers:", high_nrmse_csv)
    print("Low Pearson outliers:", low_pearson_csv)
    print("Unique suspicious sample IDs:", unique_csv)
    print("Preview index:", preview_csv)
    print("Preview folder:", PREVIEW_DIR)

    print("\nTop high-NRMSE cases:")
    cols1 = ["real_idx", "cell_id", "method", "dye", "nrmse", "rmse", "true_rms", "pearson_dye"]
    print(high_nrmse[cols1].head(15).to_string(index=False))

    print("\nTop low-Pearson cases:")
    cols2 = ["real_idx", "cell_id", "method", "pearson_overall", "min_per_dye_pearson", "max_per_dye_nrmse"]
    print(low_pearson[cols2].head(15).to_string(index=False))


if __name__ == "__main__":
    main()