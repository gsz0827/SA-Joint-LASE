"""
Plot a publication-style horizontal full representative abundance comparison
for practical 44-spot LASE experiments.

Layout:
    [RU-NNLS block]        [RU-Poisson block]        [Joint block]
    True                   True                      True
    Estimate               Estimate                  Estimate
    |Error|                |Error|                   |Error|

Each block contains the three abundance components:
    DNA / SYTO16-like, beta-tubulin / DIA-like, F-actin / DeepRed-like

The representative sample is selected automatically as the sample whose
SA-Joint-vs-RU-NNLS NRMSE improvement is closest to the median positive
improvement. This avoids cherry-picking an extreme case. You can override
this with MANUAL_REAL_IDX.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# ============================================================
# User-adjustable settings
# ============================================================

BBBC_ROOT = Path(__file__).resolve().parent
OUT_DIR = BBBC_ROOT / "results_main_exp_practical_lase_44spots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIO_NAMES = ["DNA", r"$\beta$-tubulin", "F-actin"]
SPEC_NAMES = ["SYTO16-like", "DIA-like", "DeepRed-like"]

PANEL_NAMES = [
    "DNA\n(SYTO16-like)",
    r"$\beta$-tubulin" + "\n(DIA-like)",
    "F-actin\n(DeepRed-like)",
]

# Set this to an integer real_idx if you want a specific cell.
# Example: MANUAL_REAL_IDX = 3741
MANUAL_REAL_IDX = None

METHODS = [
    {
        "key": "ru_nnls",
        "label": "RU-NNLS",
        "row_label": "RU-NNLS",
        "err_label": "|Err| RU-NNLS",
        "metrics_glob": "results/bbbc021_two_step_decode_then_unmix_44spots_practical/test_two_step_nnls_active_set/metrics_two_step_nnls_active_set_test_*samples.json",
    },
    {
        "key": "ru_poisson",
        "label": "RU-Poisson",
        "row_label": "RU-Poisson",
        "err_label": "|Err| RU-Poisson",
        "metrics_glob": "results/bbbc021_two_step_decode_then_unmix_44spots_practical/test_two_step_nnls_init_poisson_kl/metrics_two_step_nnls_init_poisson_kl_test_*samples.json",
    },
    {
        "key": "joint",
        "label": "SA-Joint",
        "row_label": "SA-Joint",
        "err_label": "|Err| SA-Joint",
        "metrics_glob": "results/bbbc021_channel_competition_sparse_44spots/test_joint_spectral_ambiguity_tv_final_nosparse_ambtv_2em04_p_1e00/metrics_test_*samples.json",
    },
]

EPS = 1e-8
FIG_DPI = 600

# ===== Font settings for publication figures =====
FONT_PANEL = 12        # 每个组分小标题，例如 DNA / SYTO16-like
FONT_ROW = 12          # 左侧行标签，例如 True / RU-NNLS / |Err|
FONT_GROUP = 18        # 每个方法块标题，例如 RU-NNLS / RU-Poisson / SA-Joint
FONT_SUPTITLE = 18     # 整张图总标题

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# Helpers
# ============================================================

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

    A_hat = np.load(a_hat_path, mmap_mode="r")  # [N, 3, H, W]
    real_indices = np.load(real_idx_path)

    meta = np.load(metadata_path, allow_pickle=True)
    A_true_path = _resolve_path(str(meta["A_npy_path"].item()))
    A_true_full = np.load(A_true_path, mmap_mode="r")  # [N_all, H, W, 3]
    A_true = np.asarray(A_true_full[real_indices], dtype=np.float32).transpose(0, 3, 1, 2)

    A_hat = np.asarray(A_hat, dtype=np.float32)
    if A_hat.shape != A_true.shape:
        raise ValueError(
            f"Shape mismatch for {method_info['label']}: A_hat {A_hat.shape}, A_true {A_true.shape}"
        )

    return {
        "key": method_info["key"],
        "label": method_info["label"],
        "row_label": method_info["row_label"],
        "err_label": method_info["err_label"],
        "metrics_path": metrics_path,
        "A_hat": A_hat,
        "A_true": A_true,
        "real_indices": np.asarray(real_indices),
    }


def _align_methods(method_data: list[dict]) -> list[dict]:
    sets = [set(map(int, m["real_indices"])) for m in method_data]
    common = set.intersection(*sets)
    if not common:
        raise RuntimeError("No common real_indices across methods.")

    joint = next((m for m in method_data if m["key"] == "joint"), method_data[0])
    ordered_common = [int(x) for x in joint["real_indices"] if int(x) in common]

    aligned = []
    for m in method_data:
        idx_map = {int(r): i for i, r in enumerate(m["real_indices"])}
        order = np.array([idx_map[r] for r in ordered_common], dtype=int)
        m2 = dict(m)
        m2["A_hat"] = np.asarray(m["A_hat"])[order]
        m2["A_true"] = np.asarray(m["A_true"])[order]
        m2["real_indices"] = np.array(ordered_common, dtype=int)
        aligned.append(m2)
    return aligned


def compute_overall_nrmse(A_hat: np.ndarray, A_true: np.ndarray) -> np.ndarray:
    """A_hat, A_true: [N, D, H, W]. Return [N]."""
    diff = A_hat - A_true
    rmse = np.sqrt(np.mean(diff ** 2, axis=(1, 2, 3)) + EPS)
    denom = np.sqrt(np.mean(A_true ** 2, axis=(1, 2, 3)) + EPS)
    return rmse / (denom + EPS)


def _safe_vmax(*imgs, percentile=99.5, minimum=1e-3):
    vals = []
    for img in imgs:
        arr = np.asarray(img)
        if arr.size:
            vals.append(float(np.percentile(arr, percentile)))
    vmax = max(vals) if vals else minimum
    return max(vmax, minimum)


def _select_representative(methods: dict[str, dict]) -> tuple[int, int, pd.DataFrame]:
    """Return local index, real_idx, and selection table."""
    true = methods["joint"]["A_true"]
    real_indices = methods["joint"]["real_indices"]

    nrmse = {}
    for key, m in methods.items():
        nrmse[key] = compute_overall_nrmse(m["A_hat"], true)

    rows = []
    for i, ridx in enumerate(real_indices):
        rows.append({
            "local_row": i,
            "real_idx": int(ridx),
            "nrmse_ru_nnls": float(nrmse["ru_nnls"][i]),
            "nrmse_ru_poisson": float(nrmse["ru_poisson"][i]),
            "nrmse_sa_joint": float(nrmse["joint"][i]),
            "improvement_ru_nnls_minus_sa_joint": float(nrmse["ru_nnls"][i] - nrmse["joint"][i]),
            "improvement_ru_poisson_minus_sa_joint": float(nrmse["ru_poisson"][i] - nrmse["joint"][i]),
        })
    table = pd.DataFrame(rows)

    if MANUAL_REAL_IDX is not None:
        matches = np.where(real_indices.astype(int) == int(MANUAL_REAL_IDX))[0]
        if len(matches) == 0:
            raise ValueError(f"MANUAL_REAL_IDX={MANUAL_REAL_IDX} not found in common sample IDs.")
        idx = int(matches[0])
        return idx, int(real_indices[idx]), table

    improvement = table["improvement_ru_nnls_minus_sa_joint"].to_numpy()
    positive = improvement[improvement > 0]
    if positive.size > 0:
        target = float(np.median(positive))
    else:
        target = float(np.median(improvement))
    idx = int(np.argmin(np.abs(improvement - target)))
    return idx, int(real_indices[idx]), table


def _plot_grouped(methods: dict[str, dict], row_idx: int, real_idx: int):
    true = methods["joint"]["A_true"][row_idx]  # [3, H, W]

    # Use matched grayscale scaling per dye across true and all estimates.
    abundance_vmax = []
    error_vmax = []
    for d in range(3):
        imgs = [true[d]] + [methods[k]["A_hat"][row_idx, d] for k in ["ru_nnls", "ru_poisson", "joint"]]
        abundance_vmax.append(_safe_vmax(*imgs, percentile=99.5, minimum=1e-3))

        errs = [np.abs(methods[k]["A_hat"][row_idx, d] - true[d]) for k in ["ru_nnls", "ru_poisson", "joint"]]
        error_vmax.append(_safe_vmax(*errs, percentile=99.5, minimum=1e-3))

    # Wide but compact: three method blocks, each containing a 3x3 grid.
    fig = plt.figure(figsize=(20.5, 6.8), dpi=FIG_DPI)
    outer = GridSpec(
        1, 3,
        figure=fig,
        left=0.025,
        right=0.995,
        bottom=0.055,
        top=0.835,
        wspace=0.075,
    )

    method_order = ["ru_nnls", "ru_poisson", "joint"]
    row_names = {
        "ru_nnls": ["True", "RU-NNLS", "|Err| RU-NNLS"],
        "ru_poisson": ["True", "RU-Poisson", "|Err| RU-Poisson"],
        "joint": ["True", "Joint", "|Err| Joint"],
    }

    for g, key in enumerate(method_order):
        m = methods[key]
        inner = GridSpecFromSubplotSpec(
            3, 3,
            subplot_spec=outer[g],
            wspace=0.001,
            hspace=0.075,
        )

        # Group title. Place as figure text above the group, not on an image axis.
        x_center = (g + 0.5) / 3.0
        fig.text(
            x_center,
            0.895,
            m["label"],
            ha="center",
            va="bottom",
            fontsize=FONT_GROUP,
            fontweight="bold",
        )

        for r in range(3):
            for d in range(3):
                ax = fig.add_subplot(inner[r, d])

                if r == 0:
                    img = true[d]
                    ax.imshow(img, cmap="gray", vmin=0.0, vmax=abundance_vmax[d], interpolation="nearest")
                elif r == 1:
                    img = m["A_hat"][row_idx, d]
                    ax.imshow(img, cmap="gray", vmin=0.0, vmax=abundance_vmax[d], interpolation="nearest")
                else:
                    img = np.abs(m["A_hat"][row_idx, d] - true[d])
                    ax.imshow(img, cmap="magma", vmin=0.0, vmax=error_vmax[d], interpolation="nearest")

                if r == 0:
                    ax.set_title(PANEL_NAMES[d], fontsize=FONT_PANEL, pad=5)

                if d == 0:
                    ax.set_ylabel(row_names[key][r], fontsize=FONT_ROW, rotation=90, labelpad=10)

                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    fig.suptitle(
        f"Representative abundance recovery comparison (real_idx={real_idx})",
        fontsize=FONT_SUPTITLE,
        y=0.975,
    )

    out_png = OUT_DIR / "fig_representative_abundance_grouped_full_44spots.png"
    out_svg = OUT_DIR / "fig_representative_abundance_grouped_full_44spots.svg"
    out_pdf = OUT_DIR / "fig_representative_abundance_grouped_full_44spots.pdf"
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"[SAVE] {out_png}")
    print(f"[SAVE] {out_svg}")
    print(f"[SAVE] {out_pdf}")


# ============================================================
# Main
# ============================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    loaded = [_load_method(m) for m in METHODS]
    loaded = _align_methods(loaded)
    methods = {m["key"]: m for m in loaded}

    row_idx, real_idx, table = _select_representative(methods)
    selection_csv = OUT_DIR / "representative_selection_table_grouped_full_44spots.csv"
    table.to_csv(selection_csv, index=False)
    print(f"Selected representative local_row={row_idx}, real_idx={real_idx}")
    print(f"[SAVE] {selection_csv}")

    _plot_grouped(methods, row_idx=row_idx, real_idx=real_idx)


if __name__ == "__main__":
    main()