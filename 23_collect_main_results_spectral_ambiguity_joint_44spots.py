import os
import numpy as np
import pandas as pd

from main_exp_config_practical_44spots import OUT_DIR

methods = [
    ("two_step_nnls", "Two-step + NNLS"),
    ("two_step_nnls_init_poisson_kl", "Two-step + NNLS-init Poisson/KL"),
    ("joint_spectral_ambiguity_tv_final_nosparse_ambtv_2em04_p_1e00", "SA-Joint Poisson inversion"),
]

rows = []

for file_name, display_name in methods:
    path = os.path.join(OUT_DIR, f"{file_name}.npz")

    if not os.path.exists(path):
        print(f"[WARN] Missing {path}")
        continue

    data = np.load(path, allow_pickle=True)

    row = {
        "Method": display_name,
        "MAE_A mean": np.nanmean(data["mae"]),
        "MAE_A std": np.nanstd(data["mae"]),
        "RMSE_A mean": np.nanmean(data["rmse"]),
        "RMSE_A std": np.nanstd(data["rmse"]),
        "NRMSE_A mean": np.nanmean(data["nrmse"]),
        "NRMSE_A std": np.nanstd(data["nrmse"]),
        "Pearson_A mean": np.nanmean(data["pearson"]),
        "Pearson_A std": np.nanstd(data["pearson"]),
    }

    rows.append(row)

df = pd.DataFrame(rows)

print("\n===== Main Experiment Results =====")
print(df)

save_csv = os.path.join(OUT_DIR, "main_results_summary_spectral_ambiguity_joint_44spots.csv")
df.to_csv(save_csv, index=False)

print(f"\n[SAVE] {save_csv}")