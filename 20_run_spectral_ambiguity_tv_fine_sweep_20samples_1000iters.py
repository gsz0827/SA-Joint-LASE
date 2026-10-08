#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine sweep around the current best spectral-ambiguity-direction TV weight.

Previous coarse sweep suggested lambda_ambiguity_tv ~= 3e-4 was best.
This script refines the search in the 1e-4 to 1e-3 range.

Usage:
    python 20_run_spectral_ambiguity_tv_fine_sweep_20samples_1000iters.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


# Fine sweep around the previous best 3e-4.
# Keep 0.0 in the table so the output remains self-contained versus baseline.
LAMBDAS = [0.0, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 7e-4, 1e-3]

MAX_SAMPLES = 20
NUM_ITERS = 1000

# Keep the same ambiguity-TV shape parameters as the coarse sweep.
AMBIGUITY_ETA = 1e-3
AMBIGUITY_POWER = 1.0
AMBIGUITY_W_MAX = 20.0


def lambda_to_tag(x: float) -> str:
    if float(x) == 0:
        return "0"
    return f"{float(x):.0e}".replace("+", "").replace("-", "m")


def expected_method_name(lam: float) -> str:
    name = "joint_spectral_ambiguity_tv_nosparse"
    if float(lam) > 0:
        name += f"_ambtv_{lambda_to_tag(lam)}_p_{lambda_to_tag(AMBIGUITY_POWER)}"
    return name


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lambda_ambiguity_tv",
        "method",
        "n_processed",
        "num_iters",
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


def main() -> None:
    root = Path(__file__).resolve().parent
    script = root / "09_invert_bbbc021_spectral_ambiguity_tv_44spots_sweepable.py"

    if not script.exists():
        raise FileNotFoundError(f"Cannot find script: {script}")

    rows = []
    summary_dir = root / "results" / "bbbc021_channel_competition_sparse_44spots"

    for lam in LAMBDAS:
        method_name = expected_method_name(lam)
        out_dir = summary_dir / f"test_{method_name}"
        metrics_path = out_dir / f"metrics_test_{MAX_SAMPLES}samples.json"

        cmd = [
            sys.executable,
            str(script),
            "--variant", "ambiguity_tv",
            "--sparse-mode", "none",
            "--max-samples", str(MAX_SAMPLES),
            "--num-iters", str(NUM_ITERS),
            "--lambda-ambiguity-tv", str(lam),
            "--ambiguity-eta", str(AMBIGUITY_ETA),
            "--ambiguity-power", str(AMBIGUITY_POWER),
            "--ambiguity-w-max", str(AMBIGUITY_W_MAX),
        ]

        print("\n" + "=" * 96)
        print(f"Running lambda_ambiguity_tv={lam:g}")
        print(" ".join(cmd))
        print("=" * 96)

        subprocess.run(cmd, cwd=str(root), check=True)

        if not metrics_path.exists():
            candidates = sorted(out_dir.glob("metrics_*samples.json"))
            if candidates:
                metrics_path = candidates[-1]
            else:
                raise FileNotFoundError(f"Cannot find metrics JSON for lambda={lam:g}: {out_dir}")

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        row = {
            "lambda_ambiguity_tv": lam,
            "method": data.get("method"),
            "n_processed": data.get("n_processed"),
            "num_iters": data.get("config", {}).get("num_iters"),
            "mean_mae": data.get("mean_mae_vs_eval_A"),
            "mean_rmse": data.get("mean_rmse_vs_eval_A"),
            "mean_nrmse": data.get("mean_nrmse_vs_eval_A"),
            "mean_pearson": data.get("mean_pearson_vs_eval_A"),
            "metrics_path": str(metrics_path),
        }
        rows.append(row)

        partial_path = summary_dir / "spectral_ambiguity_tv_fine_sweep_20samples_1000iters_summary_partial.csv"
        write_csv(rows, partial_path)

    summary_path = summary_dir / "spectral_ambiguity_tv_fine_sweep_20samples_1000iters_summary.csv"
    write_csv(rows, summary_path)

    print("\nDone. Summary:")
    print(summary_path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()