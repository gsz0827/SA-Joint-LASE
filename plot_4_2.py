import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 1. Basic settings
# =========================
OUT_DIR = Path("./figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 修改为你的 npz 文件路径
META_PATH = Path("./data/processed/bbbc021_spectral_metadata/bbbc021_3ch_spectral_metadata_8269cells.npz")

if not META_PATH.exists():
    raise FileNotFoundError(f"Cannot find metadata file: {META_PATH}")


# =========================
# 2. Load spectral metadata
# =========================
data = np.load(META_PATH, allow_pickle=True)

S = data["S"].astype(float)                         # shape: [K, L]
wavelengths = data["wavelengths"].astype(float)     # shape: [L]
dye_names = [str(x) for x in data["channel_names"]]

# For better display in the figure
dye_names = [name.replace("_", "-") for name in dye_names]

# Use the cosine similarity matrix saved in your metadata file
if "cosine_similarity" in data.files:
    C = data["cosine_similarity"].astype(float)
else:
    eps = 1e-12
    S_l2 = S / (np.linalg.norm(S, axis=1, keepdims=True) + eps)
    C = S_l2 @ S_l2.T

print("Loaded metadata from:", META_PATH)
print("S shape:", S.shape)
print("wavelengths:", wavelengths)
print("dye names:", dye_names)
print("C shape:", C.shape)


# =========================
# 3. Check consistency
# =========================
eps = 1e-12
S_l2 = S / (np.linalg.norm(S, axis=1, keepdims=True) + eps)
C_recomputed = S_l2 @ S_l2.T

print("\nMax difference between saved C and recomputed C:")
print(np.max(np.abs(C - C_recomputed)))


# =========================
# 4. Eigen-decomposition and direction weights
# =========================
eigvals, eigvecs = np.linalg.eigh(C)

# Sort eigenvalues from large to small for clearer display
sort_idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[sort_idx]
eigvecs = eigvecs[:, sort_idx]

# Direction weights used in spectral-ambiguity-aware TV
eta = 1e-3
p = 1.0
w_max = 20.0

sigma_max = eigvals.max()
weights = ((sigma_max + eta) / (eigvals + eta)) ** p
weights = np.clip(weights, 1.0, w_max)


# =========================
# 5. Plot right panel:
#    Gram matrix + spectral-direction weights
# =========================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(
    1, 2,
    figsize=(7.2, 3.2),
    gridspec_kw={"width_ratios": [1.05, 1.0]}
)


# ---------- 5.1 Gram matrix heatmap ----------
ax = axes[0]

im = ax.imshow(C, vmin=0, vmax=1)

ax.set_title("Spectral Gram matrix $C$", fontsize=11)
ax.set_xticks(np.arange(len(dye_names)))
ax.set_yticks(np.arange(len(dye_names)))
ax.set_xticklabels(dye_names, rotation=35, ha="right", fontsize=8)
ax.set_yticklabels(dye_names, fontsize=8)

# Annotate matrix values
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        ax.text(
            j, i,
            f"{C[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=9
        )

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Correlation", fontsize=9)
cbar.ax.tick_params(labelsize=8)


# ---------- 5.2 Spectral-direction weights ----------
ax = axes[1]

direction_labels = [f"Dir. {i+1}" for i in range(len(weights))]
x = np.arange(len(weights))

bars = ax.bar(x, weights, width=0.6)

ax.set_title("Spectral-direction weights", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(direction_labels, fontsize=9)
ax.set_ylabel("Weight $w_m$", fontsize=10)
ax.set_xlabel("Eigen-direction", fontsize=10)

# Add eigenvalue and weight labels above bars
for bar, sigma, weight in zip(bars, eigvals, weights):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"$\\sigma$={sigma:.3f}\n$w$={weight:.2f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

ax.set_ylim(0, max(weights) * 1.35)
ax.grid(axis="y", linestyle="--", alpha=0.35)

fig.suptitle(
    "Spectral correlation structure",
    fontsize=12,
    y=1.03
)

fig.tight_layout()


# =========================
# 6. Save figures
# =========================
png_path = OUT_DIR / "fig4_2_right_spectral_correlation_structure.png"
svg_path = OUT_DIR / "fig4_2_right_spectral_correlation_structure.svg"
pdf_path = OUT_DIR / "fig4_2_right_spectral_correlation_structure.pdf"

fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

plt.show()

print("\nSaved:")
print(png_path)
print(svg_path)
print(pdf_path)


# =========================
# 7. Print numerical values
# =========================
print("\nSpectral library S:")
print(S)

print("\nSpectral Gram matrix C:")
print(C)

print("\nEigenvalues sigma:")
print(eigvals)

print("\nSpectral-direction weights w:")
print(weights)