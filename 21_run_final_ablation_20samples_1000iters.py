#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the final 20-sample / 1000-iteration ablation suite for
spectral-ambiguity-aware Poisson inversion.

Put this file in the same folder as:
    09_invert_bbbc021_spectral_ambiguity_tv_44spots_final_ablation.py

Usage:
    python 21_run_final_ablation_20samples_1000iters.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path


MAX_SAMPLES = 20
NUM_ITERS = 1000

# Core ablation suite. These isolate the contribution of each final component.
EXPERIMENTS = [
    {
        "name": "poisson_upper_only",
        "variant": "poisson_only",
        "description": "Poisson/KL + upper-bound only",
    },
    {
        "name": "standard_tv_no_ambiguity_tv",
        "variant": "no_ambiguity_tv",
        "description": "Poisson/KL + standard TV + upper-bound",
    },
    {
        "name": "ambiguity_tv_only_no_standard_tv",
        "variant": "ambiguity_tv_only",
        "description": "Poisson/KL + spectral-ambiguity TV + upper-bound",
    },
    {
        "name": "final_standard_tv_plus_ambiguity_tv",
        "variant": "final",
        "description": "Poisson/KL + standard TV + spectral-ambiguity TV + upper-bound",
    },
    {
        "name": "final_no_upper_bound",
        "variant": "no_upper",
        "description": "Poisson/KL + standard TV + spectral-ambiguity TV, no upper-bound penalty",
    },

    # Optional legacy sparse comparisons.
    # Uncomment these two if you want the ablation table to explicitly show
    # that channel-competition sparse priors are not the final choice.
    # {
    #     "name": "legacy_fixed_channel_sparse",
    #     "variant": "fixed_sparse",
    #     "description": "Poisson/KL + fixed channel-competition sparse + standard TV + upper-bound",
    # },
    # {
    #     "name": "legacy_adaptive_spectral_sparse",
    #     "variant": "adaptive_sparse",
    #     "description": "Poisson/KL + adaptive spectral-correlation sparse + standard TV + upper-bound",
    # },
]


def main() -> None:
    root = Path(__file__).resolve().parent
    script = root / "09_invert_bbbc021_spectral_ambiguity_tv_44spots_final_ablation.py"

    if not script.exists():
        raise FileNotFoundError(f"Cannot find script: {script}")

    result_root = root / "results" / "bbbc021_channel_competition_sparse_44spots"
    summary_dir = result_root
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for exp in EXPERIMENTS:
        start_time = time.time() - 1.0

        cmd = [
            sys.executable,
            str(script),
            "--variant", exp["variant"],
            "--max-samples", str(MAX_SAMPLES),
            "--num-iters", str(NUM_ITERS),
        ]

        print("\n" + "=" * 100)
        print(f"Running ablation: {exp['name']}")
        print(exp["description"])
        print(" ".join(cmd))
        print("=" * 100)

        subprocess.run(cmd, cwd=str(root), check=True)

        metrics_path = find_latest_metrics_json(
            result_root=result_root,
            max_samples=MAX_SAMPLES,
            start_time=start_time,
        )

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = data.get("config", {})

        row = {
            "experiment": exp["name"],
            "variant": exp["variant"],
            "description": exp["description"],
            "method": data.get("method"),
            "n_processed": data.get("n_processed"),
            "num_iters": config.get("num_iters"),
            "sparse_mode": config.get("sparse_mode"),
            "lambda_tv": config.get("lambda_tv"),
            "lambda_ambiguity_tv": config.get("lambda_ambiguity_tv"),
            "lambda_upper": config.get("lambda_upper"),
            "lambda_channel_sparse": config.get("lambda_channel_sparse"),
            "lambda_adaptive_sparse": config.get("lambda_adaptive_sparse"),
            "mean_mae": data.get("mean_mae_vs_eval_A"),
            "mean_rmse": data.get("mean_rmse_vs_eval_A"),
            "mean_nrmse": data.get("mean_nrmse_vs_eval_A"),
            "mean_pearson": data.get("mean_pearson_vs_eval_A"),
            "metrics_path": str(metrics_path),
        }
        rows.append(row)

        partial_path = summary_dir / "final_ablation_20samples_1000iters_summary_partial.csv"
        write_csv(rows, partial_path)

    summary_path = summary_dir / "final_ablation_20samples_1000iters_summary.csv"
    write_csv(rows, summary_path)

    print("\nDone. Summary:")
    print(summary_path)
    for row in rows:
        print(row)


def find_latest_metrics_json(result_root: Path, max_samples: int, start_time: float) -> Path:
    pattern = f"metrics_test_{max_samples}samples.json"
    candidates = [
        p for p in result_root.rglob(pattern)
        if p.is_file() and p.stat().st_mtime >= start_time
    ]

    if not candidates:
        # Fallback: choose the latest matching file if timestamp filtering fails.
        candidates = [p for p in result_root.rglob(pattern) if p.is_file()]

    if not candidates:
        raise FileNotFoundError(f"Cannot find {pattern} under {result_root}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "experiment",
        "variant",
        "description",
        "method",
        "n_processed",
        "num_iters",
        "sparse_mode",
        "lambda_tv",
        "lambda_ambiguity_tv",
        "lambda_upper",
        "lambda_channel_sparse",
        "lambda_adaptive_sparse",
        "mean_mae",
        "mean_rmse",
        "mean_nrmse",
        "mean_pearson",
        "metrics_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()