"""
Plot per-dye NRMSE boxplots for practical 44-spot LASE experiments.

Expected existing outputs from:
    python 10_two_step_bbbc021_decode_then_unmix_44spots.py --unmix nnls
    python 10_two_step_bbbc021_decode_then_unmix_44spots.py --unmix poisson
    python 09_invert_bbbc021_pure_joint_poisson_constant_init_44spots.py

This script reads the saved A_hat .npy files and the corresponding A_true
from the train-ready metadata, computes NRMSE per dye and per cell, and
produces publication-style grouped boxplots.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


# ============================================================
# User-adjustable settings
# ============================================================

BBBC_ROOT = Path(__file__).resolve().parent
OUT_DIR = BBBC_ROOT / "results_main_exp_practical_lase_44spots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DYE_NAMES = ["SYTO16-like", "DIA-like", "DeepRed-like"]

METHODS = [
    {
        "key": "ru_nnls",
        "label": "RU-NNLS",
        "metrics_glob": (
            "results/bbbc021_two_step_decode_then_unmix_44spots_practical/"
            "test_two_step_nnls_active_set/"
            "metrics_two_step_nnls_active_set_test_*samples.json"
        ),
    },
    {
        "key": "ru_poisson",
        "label": "RU-Poisson",
        "metrics_glob": (
            "results/bbbc021_two_step_decode_then_unmix_44spots_practical/"
            "test_two_step_nnls_init_poisson_kl/"
            "metrics_two_step_nnls_init_poisson_kl_test_*samples.json"
        ),
    },
    {
        "key": "joint",
        "label": "SA-Joint",
        "metrics_glob": (
            "results/bbbc021_channel_competition_sparse_44spots/"
            "test_joint_spectral_ambiguity_tv_final_nosparse_ambtv_2em04_p_1e00/"
            "metrics_test_*samples.json"
        ),
    },
]

EPS = 1e-8

# 输出图分辨率。论文插图建议 600 dpi；如果仍觉得不够清楚，可改成 900。
FIG_DPI = 600

# 是否在图上显示 Wilcoxon 显著性检验文字标注。
# 设为 False 后，图 4-2 上方不会再出现 n.s. / p-value 文本。
SHOW_STAT_ANNOTATIONS = False

# A cell-dye pair is treated as active if the true abundance RMS is above this threshold.
# You can later test 0.005, 0.01, 0.02 for sensitivity analysis.
ACTIVE_RMS_THR = 0.01


# ============================================================
# Matplotlib publication settings
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",

    # 全局字体稍微放大
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,

    # 坐标轴线条也稍微加粗
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,

    # 保证 PDF/SVG 文字清晰、可编辑
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

    # If the stored path came from another absolute root, try by basename.
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
            f"Shape mismatch for {method_info['label']}: "
            f"A_hat {A_hat.shape}, A_true {A_true.shape}"
        )

    return {
        "key": method_info["key"],
        "label": method_info["label"],
        "metrics_path": metrics_path,
        "A_hat": A_hat,
        "A_true": A_true,
        "real_indices": np.asarray(real_indices),
    }


def _align_methods(method_data: list[dict]) -> list[dict]:
    """Align all methods by common real_indices and preserve SA-Joint order if possible."""
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


def compute_per_dye_metrics(A_hat: np.ndarray, A_true: np.ndarray) -> dict:
    """
    A_hat, A_true: [N, D, H, W]

    Returns a dict of [N, D] arrays:
        true_rms: true abundance RMS per cell-dye pair
        hat_rms:  estimated abundance RMS per cell-dye pair
        rmse:     RMSE per cell-dye pair
        nrmse:    RMSE / true_rms per cell-dye pair
    """
    A_hat = np.asarray(A_hat, dtype=np.float32)
    A_true = np.asarray(A_true, dtype=np.float32)

    if A_hat.shape != A_true.shape:
        raise ValueError(f"A_hat shape {A_hat.shape} != A_true shape {A_true.shape}")

    diff = A_hat - A_true

    true_rms = np.sqrt(np.mean(A_true ** 2, axis=(2, 3)))
    hat_rms = np.sqrt(np.mean(A_hat ** 2, axis=(2, 3)))
    rmse = np.sqrt(np.mean(diff ** 2, axis=(2, 3)))
    nrmse = rmse / (true_rms + EPS)

    return {
        "true_rms": true_rms.astype(np.float32),
        "hat_rms": hat_rms.astype(np.float32),
        "rmse": rmse.astype(np.float32),
        "nrmse": nrmse.astype(np.float32),
    }


def _format_p(p: float) -> str:
    """
    Format p-value text.

    This function is kept for optional use.
    By default, SHOW_STAT_ANNOTATIONS = False, so p-value text will not appear on figures.
    """
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "p<1e-4"
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def _paired_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """
    Paired Wilcoxon signed-rank test.

    This function is kept for optional use.
    It is only used when SHOW_STAT_ANNOTATIONS = True.
    """
    if wilcoxon is None:
        return np.nan

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if x.size < 3:
        return np.nan

    try:
        return float(
            wilcoxon(
                x,
                y,
                zero_method="wilcox",
                alternative="two-sided",
            ).pvalue
        )
    except ValueError:
        return np.nan


def plot_grouped_metric_by_mask(
    method_data: list[dict],
    per_method_metrics: dict,
    mask: np.ndarray,
    metric_key: str,
    ylabel: str,
    title: str,
    file_stem: str,
) -> None:
    """
    Plot grouped per-dye boxplots for a selected metric under a given mask.

    mask:
        [N, D] boolean array.
        For active-dye NRMSE, mask = active_mask.
        For inactive-dye false-positive RMSE, mask = inactive_mask.

    metric_key:
        One of: "nrmse", "rmse", "hat_rms", "true_rms".
    """

    # Larger figure size for clearer thesis/paper insertion.
    fig, ax = plt.subplots(figsize=(11.5, 6.2))

    base = np.arange(len(DYE_NAMES), dtype=float)
    offsets = np.linspace(-0.25, 0.25, len(method_data))
    width = 0.19

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    rng = np.random.default_rng(2026)

    all_finite_values = []

    for mi, m in enumerate(method_data):
        metric_mat = per_method_metrics[m["key"]][metric_key]  # [N, D]
        positions = base + offsets[mi]

        vals_by_dye = []

        for d in range(len(DYE_NAMES)):
            vals = metric_mat[mask[:, d], d]
            vals = vals[np.isfinite(vals)]
            all_finite_values.append(vals)

            # Matplotlib boxplot does not like empty arrays.
            if vals.size == 0:
                vals_by_dye.append(np.array([np.nan]))
            else:
                vals_by_dye.append(vals)

        bp = ax.boxplot(
            vals_by_dye,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
            boxprops={"linewidth": 0.9},
            whiskerprops={"linewidth": 0.9},
            capprops={"linewidth": 0.9},
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(colors[mi % len(colors)])
            patch.set_alpha(0.38)

        # Scatter raw points.
        for d in range(len(DYE_NAMES)):
            vals = metric_mat[mask[:, d], d]
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                continue

            jitter = rng.normal(0, width * 0.11, size=vals.size)
            ax.scatter(
                np.full(vals.size, positions[d]) + jitter,
                vals,
                s=13,
                alpha=0.55,
                color=colors[mi % len(colors)],
                edgecolors="none",
            )

    # ------------------------------------------------------------
    # Optional paired Wilcoxon annotations.
    # Default: disabled, so no n.s. / p-value text appears on the figure.
    # ------------------------------------------------------------
    joint_metrics = per_method_metrics.get("joint")

    if SHOW_STAT_ANNOTATIONS and joint_metrics is not None:
        valid_finite_lists = [v for v in all_finite_values if v.size > 0]

        if valid_finite_lists:
            finite_concat = np.concatenate(valid_finite_lists)
            y_top = np.nanmax(finite_concat)
            y_min = np.nanmin(finite_concat)
            y_range = max(y_top - y_min, 1e-3)

            for d in range(len(DYE_NAMES)):
                text_lines = []

                for k, label_short in [("ru_nnls", "NNLS"), ("ru_poisson", "Poisson")]:
                    if k in per_method_metrics:
                        valid = mask[:, d]
                        x = per_method_metrics[k][metric_key][valid, d]
                        y = joint_metrics[metric_key][valid, d]
                        p = _paired_pvalue(x, y)
                        text_lines.append(f"{label_short} vs SA-Joint: {_format_p(p)}")

                if text_lines:
                    ax.text(
                        base[d],
                        y_top + 0.05 * y_range,
                        "\n".join(text_lines),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

            ax.set_ylim(top=y_top + 0.22 * y_range)

    # Add sample counts below dye names.
    counts = []
    for d in range(len(DYE_NAMES)):
        counts.append(int(np.sum(mask[:, d])))

    xticklabels = [
        f"{dye}\n(n={counts[i]})"
        for i, dye in enumerate(DYE_NAMES)
    ]

    ax.set_xticks(base)
    ax.set_xticklabels(xticklabels, fontsize=12)

    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    handles = []
    for mi, m in enumerate(method_data):
        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=colors[mi % len(colors)],
                alpha=0.38,
                label=m["label"],
            )
        )

    ax.legend(handles=handles, frameon=False, loc="upper right")

    fig.tight_layout()

    for ext in ["png", "svg", "pdf"]:
        out_path = OUT_DIR / f"{file_stem}.{ext}"
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.04)

    plt.close(fig)

    print(" ", OUT_DIR / f"{file_stem}.png")
    print(" ", OUT_DIR / f"{file_stem}.svg")
    print(" ", OUT_DIR / f"{file_stem}.pdf")


# ============================================================
# Main
# ============================================================

def main() -> None:
    method_data = [_load_method(m) for m in METHODS]
    method_data = _align_methods(method_data)

    # ------------------------------------------------------------
    # Active / inactive mask is determined only by A_true.
    # Since all methods are aligned to the same real_indices,
    # we can use the first method's A_true as the reference.
    # ------------------------------------------------------------
    A_true_ref = method_data[0]["A_true"]  # [N, D, H, W]
    true_rms_ref = np.sqrt(np.mean(A_true_ref ** 2, axis=(2, 3)))  # [N, D]

    active_mask = true_rms_ref > ACTIVE_RMS_THR
    inactive_mask = ~active_mask

    print("\n===== Active / inactive dye definition =====")
    print(f"ACTIVE_RMS_THR = {ACTIVE_RMS_THR}")

    for d, dye in enumerate(DYE_NAMES):
        print(
            f"{dye:12s}: active={int(np.sum(active_mask[:, d])):5d}, "
            f"inactive={int(np.sum(inactive_mask[:, d])):5d}"
        )

    # ------------------------------------------------------------
    # Compute per-dye metrics for each method.
    # ------------------------------------------------------------
    rows = []
    per_method_metrics = {}

    for m in method_data:
        metrics = compute_per_dye_metrics(m["A_hat"], m["A_true"])
        per_method_metrics[m["key"]] = metrics

        for i, ridx in enumerate(m["real_indices"]):
            for d, dye in enumerate(DYE_NAMES):
                is_active = bool(active_mask[i, d])

                rows.append({
                    "sample_id": int(ridx),
                    "method": m["label"],
                    "method_key": m["key"],
                    "dye": dye,
                    "dye_index": d,

                    "true_rms": float(metrics["true_rms"][i, d]),
                    "hat_rms": float(metrics["hat_rms"][i, d]),
                    "rmse": float(metrics["rmse"][i, d]),
                    "nrmse": float(metrics["nrmse"][i, d]),

                    "active_rms_threshold": float(ACTIVE_RMS_THR),
                    "is_active_dye": is_active,
                    "is_inactive_dye": not is_active,

                    # For inactive dye, this is the recommended error metric.
                    "inactive_fp_rmse": float(metrics["rmse"][i, d]),
                    "inactive_hat_rms": float(metrics["hat_rms"][i, d]),
                })

    df = pd.DataFrame(rows)

    csv_path = OUT_DIR / "per_dye_active_inactive_metrics_practical_44spots_long.csv"
    df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------
    # Summary tables.
    # ------------------------------------------------------------
    active_df = df[df["is_active_dye"]].copy()
    inactive_df = df[df["is_inactive_dye"]].copy()

    active_summary = (
        active_df.groupby(["method", "dye"], as_index=False)["nrmse"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )

    inactive_summary = (
        inactive_df.groupby(["method", "dye"], as_index=False)[
            ["inactive_fp_rmse", "inactive_hat_rms"]
        ]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )

    active_summary_path = OUT_DIR / "per_dye_active_nrmse_practical_44spots_summary.csv"
    inactive_summary_path = OUT_DIR / "per_dye_inactive_false_positive_practical_44spots_summary.csv"

    active_summary.to_csv(active_summary_path, index=False)
    inactive_summary.to_csv(inactive_summary_path, index=False)

    print("\n===== Active-dye NRMSE summary =====")
    print(active_summary)

    print("\n===== Inactive-dye false-positive summary =====")
    print(inactive_summary)

    # ------------------------------------------------------------
    # Plot 1: active-dye NRMSE.
    # This is the figure you use as Figure 4-2 in the paper.
    # ------------------------------------------------------------
    print("\n[SAVE] Active-dye NRMSE figures:")
    plot_grouped_metric_by_mask(
        method_data=method_data,
        per_method_metrics=per_method_metrics,
        mask=active_mask,
        metric_key="nrmse",
        ylabel=r"Active-dye NRMSE$_A$ (lower is better)",
        title=(
            "Active-dye abundance NRMSE under the practical 44-spot LASE setting\n"
            rf"Active dye: RMS$(A^{{true}}) > {ACTIVE_RMS_THR}$"
        ),
        file_stem="fig_per_dye_active_nrmse_practical_44spots",
    )

    # ------------------------------------------------------------
    # Plot 2: inactive-dye false-positive RMSE.
    # ------------------------------------------------------------
    print("\n[SAVE] Inactive-dye false-positive RMSE figures:")
    plot_grouped_metric_by_mask(
        method_data=method_data,
        per_method_metrics=per_method_metrics,
        mask=inactive_mask,
        metric_key="rmse",
        ylabel=r"Inactive-dye FP RMSE$_A$ (lower is better)",
        title=(
            "Inactive-dye false-positive error under the practical 44-spot LASE setting\n"
            rf"Inactive dye: RMS$(A^{{true}}) \leq {ACTIVE_RMS_THR}$"
        ),
        file_stem="fig_per_dye_inactive_fp_rmse_practical_44spots",
    )

    # Optional: also plot predicted abundance RMS for inactive dye.
    print("\n[SAVE] Inactive-dye predicted RMS figures:")
    plot_grouped_metric_by_mask(
        method_data=method_data,
        per_method_metrics=per_method_metrics,
        mask=inactive_mask,
        metric_key="hat_rms",
        ylabel=r"Inactive-dye predicted RMS$_A$ (lower is better)",
        title=(
            "Inactive-dye predicted abundance magnitude under the practical 44-spot LASE setting\n"
            rf"Inactive dye: RMS$(A^{{true}}) \leq {ACTIVE_RMS_THR}$"
        ),
        file_stem="fig_per_dye_inactive_hat_rms_practical_44spots",
    )

    print("\nSaved:")
    print(" ", csv_path)
    print(" ", active_summary_path)
    print(" ", inactive_summary_path)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()