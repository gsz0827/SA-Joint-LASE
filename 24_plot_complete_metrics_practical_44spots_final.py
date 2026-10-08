"""
Plot complete metric comparison for BBBC021 practical 44-spots experiment.

Expected result files are in:
    results_main_exp_practical_lase_44spots/

The script reads per-cell metric arrays from .npz files, then plots boxplots
with jittered sample points for:
    MAE_A, RMSE_A, NRMSE_A, Pearson_A, PSNR_A

PSNR_A is computed from per-cell RMSE_A as:
    PSNR_A = 20 * log10(1 / RMSE_A)
assuming abundance maps are normalized to [0, 1].

Run:
    python 24_plot_complete_metrics_practical_44spots.py
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ============================================================
# Config
# ============================================================

OUT_DIR = Path("results_main_exp_practical_lase_44spots")

# Candidate file names are included to make the script robust to small naming changes.
METHODS = [
    {
        "key": "ru_nnls",
        "label": "RU-NNLS",
        "candidates": [
            "two_step_nnls.npz",
        ],
    },
    {
        "key": "ru_poisson",
        "label": "RU-Poisson",
        "candidates": [
            "two_step_nnls_init_poisson_kl.npz",
            "two_step_nnls_init_poisson.npz",
        ],
    },
    {
        "key": "joint",
        "label": "SA-Joint",
        "candidates": [
            "joint_spectral_ambiguity_tv_final_nosparse_ambtv_2em04_p_1e00.npz",
        ],
    },
]

EPS = 1e-12
RNG_SEED = 2026
FIG_DPI = 600

# Whether to draw raw sample points on the boxplot.
SHOW_JITTER_POINTS = True

# Whether to use only samples shared by all methods according to sample_ids.
ALIGN_BY_SAMPLE_ID = True


# ============================================================
# Loading utilities
# ============================================================

def find_existing_file(out_dir: Path, candidates):
    for name in candidates:
        p = out_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the candidate files exists:\n" +
        "\n".join(str(out_dir / name) for name in candidates)
    )


def get_array(data, names):
    """Return the first available metric array from possible names."""
    for name in names:
        if name in data.files:
            arr = np.asarray(data[name], dtype=np.float64)
            return arr
    raise KeyError(f"None of these arrays found in npz: {names}. Available: {data.files}")


def load_method_result(method):
    path = find_existing_file(OUT_DIR, method["candidates"])
    data = np.load(path, allow_pickle=True)

    result = {
        "key": method["key"],
        "label": method["label"],
        "path": path,
        "mae": get_array(data, ["mae", "mae_A", "MAE_A"]),
        "rmse": get_array(data, ["rmse", "rmse_A", "RMSE_A"]),
        "nrmse": get_array(data, ["nrmse", "nrmse_A", "NRMSE_A"]),
        "pearson": get_array(data, ["pearson", "pearson_A", "Pearson_A"]),
    }

    # Optional sample IDs. If absent, use positional indices.
    if "sample_ids" in data.files:
        result["sample_ids"] = np.asarray(data["sample_ids"])
    elif "real_idx" in data.files:
        result["sample_ids"] = np.asarray(data["real_idx"])
    else:
        n = len(result["nrmse"])
        result["sample_ids"] = np.arange(n)

    # Derived metric: PSNR for abundance map, assuming normalized A in [0, 1].
    result["psnr"] = 20.0 * np.log10(1.0 / np.maximum(result["rmse"], EPS))

    n = len(result["sample_ids"])
    for metric in ["mae", "rmse", "nrmse", "pearson", "psnr"]:
        if len(result[metric]) != n:
            raise ValueError(
                f"Length mismatch for {method['key']} metric {metric}: "
                f"len(metric)={len(result[metric])}, len(sample_ids)={n}"
            )

    return result


def align_results_by_sample_id(results):
    """Align all methods to the common set/order of sample IDs."""
    if not ALIGN_BY_SAMPLE_ID:
        return results

    common = set(results[0]["sample_ids"].tolist())
    for r in results[1:]:
        common &= set(r["sample_ids"].tolist())

    if len(common) == 0:
        raise RuntimeError("No common sample_ids across methods.")

    # Keep the order of the first method.
    common_ordered = [sid for sid in results[0]["sample_ids"].tolist() if sid in common]

    aligned = []
    for r in results:
        id_to_pos = {sid: i for i, sid in enumerate(r["sample_ids"].tolist())}
        idx = np.array([id_to_pos[sid] for sid in common_ordered], dtype=int)
        rr = dict(r)
        rr["sample_ids"] = np.asarray(common_ordered)
        for metric in ["mae", "rmse", "nrmse", "pearson", "psnr"]:
            rr[metric] = np.asarray(r[metric])[idx]
        aligned.append(rr)

    return aligned


# ============================================================
# Statistics and plotting
# ============================================================

def format_p_value(p):
    if p < 1e-4:
        return "p<1e-4"
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def paired_wilcoxon(x, y, alternative):
    """
    Paired Wilcoxon test. Returns np.nan if scipy unavailable or test fails.

    alternative:
        'greater' means x tends to be greater than y.
        'less' means x tends to be less than y.
        'two-sided' means distribution differs.
    """
    if not HAS_SCIPY:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    try:
        return float(wilcoxon(x, y, alternative=alternative).pvalue)
    except Exception:
        return np.nan


def make_summary_tables(results):
    rows = []
    long_rows = []

    for r in results:
        for metric in ["mae", "rmse", "nrmse", "pearson", "psnr"]:
            values = np.asarray(r[metric], dtype=float)
            rows.append({
                "method": r["label"].replace("\n", " "),
                "metric": metric,
                "mean": np.nanmean(values),
                "std": np.nanstd(values),
                "median": np.nanmedian(values),
                "q1": np.nanpercentile(values, 25),
                "q3": np.nanpercentile(values, 75),
                "n": np.sum(np.isfinite(values)),
                "source_file": str(r["path"]),
            })

            for sid, val in zip(r["sample_ids"], values):
                long_rows.append({
                    "sample_id": sid,
                    "method": r["label"].replace("\n", " "),
                    "metric": metric,
                    "value": val,
                })

    summary = pd.DataFrame(rows)
    long_df = pd.DataFrame(long_rows)

    summary_path = OUT_DIR / "complete_metrics_summary_practical_44spots.csv"
    long_path = OUT_DIR / "complete_metrics_long_practical_44spots.csv"

    summary.to_csv(summary_path, index=False)
    long_df.to_csv(long_path, index=False)

    print("\n===== Summary =====")
    print(summary)
    print(f"\n[SAVE] {summary_path}")
    print(f"[SAVE] {long_path}")

    return summary, long_df


def plot_complete_metrics(results):
    metric_specs = [
        {
            "key": "mae",
            "title": "MAE$_A$",
            "ylabel": "MAE (lower is better)",
            "better": "lower",
        },
        {
            "key": "rmse",
            "title": "RMSE$_A$",
            "ylabel": "RMSE (lower is better)",
            "better": "lower",
        },
        {
            "key": "nrmse",
            "title": "NRMSE$_A$",
            "ylabel": "NRMSE (lower is better)",
            "better": "lower",
        },
        {
            "key": "pearson",
            "title": "Pearson$_A$",
            "ylabel": "Pearson r (higher is better)",
            "better": "higher",
        },
        {
            "key": "psnr",
            "title": "PSNR$_A$",
            "ylabel": "PSNR (dB, higher is better)",
            "better": "higher",
        },
    ]

    tick_labels = [r["label"] for r in results]
    rng = np.random.default_rng(RNG_SEED)

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(4.0 * len(metric_specs), 4.2))
    if len(metric_specs) == 1:
        axes = [axes]

    joint = results[-1]

    for ax, spec in zip(axes, metric_specs):
        metric = spec["key"]
        values = [np.asarray(r[metric], dtype=float) for r in results]

        ax.boxplot(
            values,
            tick_labels=tick_labels,
            showmeans=True,
            meanline=True,
            widths=0.55,
            patch_artist=False,
            showfliers=False,
        )

        if SHOW_JITTER_POINTS:
            for i, arr in enumerate(values, start=1):
                arr = arr[np.isfinite(arr)]
                jitter = rng.normal(loc=0.0, scale=0.035, size=len(arr))
                ax.scatter(
                    np.full(len(arr), i, dtype=float) + jitter,
                    arr,
                    s=14,
                    alpha=0.45,
                    linewidths=0,
                )

        # Paired Wilcoxon: compare each two-step baseline against pure joint.
        p_texts = []
        for r in results[:-1]:
            if spec["better"] == "lower":
                # test whether baseline metric > joint metric
                p = paired_wilcoxon(r[metric], joint[metric], alternative="greater")
            else:
                # test whether baseline metric < joint metric
                p = paired_wilcoxon(r[metric], joint[metric], alternative="less")
            p_texts.append(f"{r['label'].splitlines()[0]} vs SA-Joint: {format_p_value(p)}")

        ax.set_title(spec["title"])
        ax.set_ylabel(spec["ylabel"])
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

        # --------------------------------------------------
        # Zoom Pearson panel to show the main distribution
        # --------------------------------------------------
        if metric == "pearson":

            all_vals = np.concatenate([
                arr[np.isfinite(arr)]
                for arr in values
            ])

            # 涓讳綋鍒嗗竷
            q1 = np.percentile(all_vals, 1)
            q99 = np.percentile(all_vals, 99)

            margin = 0.01

            ymin = max(0.0, q1 - margin)
            ymax = min(1.005, q99 + margin)

            ax.set_ylim(ymin, ymax)

            ax.set_title("Pearson$_A$ (zoomed)")

            low_thr = ymin

            for i, arr in enumerate(values, start=1):

                arr = np.asarray(arr)

                n_low = np.sum(arr < low_thr)

                if n_low > 0:
                    ax.text(
                        i,
                        ymin + 0.002,
                        f"{n_low} out",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="red",
                    )

        # Put p-values inside each panel without overcrowding the plot.
        if HAS_SCIPY:
            ax.text(
                0.02,
                0.98,
                "\n".join(p_texts),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
            )

    fig.suptitle(
        "Abundance recovery metrics under the practical 44-spot LASE setting",
        y=1.03,
        fontsize=14,
    )
    fig.tight_layout()

    png_path = OUT_DIR / "fig_complete_metrics_boxplot_practical_44spots.png"
    svg_path = OUT_DIR / "fig_complete_metrics_boxplot_practical_44spots.svg"
    pdf_path = OUT_DIR / "fig_complete_metrics_boxplot_practical_44spots.pdf"

    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[SAVE] {png_path}")
    print(f"[SAVE] {svg_path}")
    print(f"[SAVE] {pdf_path}")

    save_full_pearson_plot(results)


def save_full_pearson_plot(results):

    tick_labels = [r["label"] for r in results]

    rng = np.random.default_rng(RNG_SEED)

    fig, ax = plt.subplots(figsize=(5,4))

    values = [
        np.asarray(r["pearson"], dtype=float)
        for r in results
    ]

    ax.boxplot(
        values,
        tick_labels=tick_labels,
        showmeans=True,
        meanline=True,
        widths=0.55,
        showfliers=False,
    )

    for i, arr in enumerate(values, start=1):

        arr = arr[np.isfinite(arr)]

        jitter = rng.normal(
            loc=0,
            scale=0.035,
            size=len(arr)
        )

        ax.scatter(
            np.full(len(arr), i) + jitter,
            arr,
            s=12,
            alpha=0.35,
        )

    ax.set_ylim(0.0, 1.01)

    ax.set_ylabel("Pearson r")
    ax.set_title("Pearson$_A$ (full range)")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "fig_pearson_full_range_44spots.png",
        dpi=FIG_DPI,
    )

    plt.close(fig)


def main():
    if not OUT_DIR.exists():
        raise FileNotFoundError(f"OUT_DIR does not exist: {OUT_DIR.resolve()}")

    print("Loading results from:", OUT_DIR.resolve())

    results = [load_method_result(m) for m in METHODS]
    results = align_results_by_sample_id(results)

    print("\nLoaded methods:")
    for r in results:
        print(f"  {r['label'].replace(chr(10), ' '):35s} n={len(r['sample_ids'])} file={r['path'].name}")

    make_summary_tables(results)
    plot_complete_metrics(results)


if __name__ == "__main__":
    main()