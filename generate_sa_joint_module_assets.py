from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

out_dir = Path("sa_joint_module_assets")
out_dir.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "savefig.transparent": True,
})

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

def save(fig, name):
    fig.savefig(out_dir / f"{name}.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_dir / f"{name}.svg", transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

rng = np.random.default_rng(12)
t = np.linspace(0, 100, 1200)
base = 4200 * np.exp(-t / 13.5) + 55 * np.exp(-((t - 45) / 16) ** 2) + 5
spikes = np.zeros_like(t)
centers = rng.uniform(2, 60, 28)
amps = rng.uniform(100, 1900, len(centers)) * np.exp(-centers / 55)
widths = rng.uniform(0.08, 0.45, len(centers))
for c, a, w in zip(centers, amps, widths):
    spikes += a * np.exp(-0.5 * ((t - c) / w) ** 2)
lam = np.clip(base + spikes, 0.5, None)
y = rng.poisson(lam).astype(float)
fig, ax = plt.subplots(figsize=(3.0, 2.1))
ax.plot(t, y + 1, lw=1.2)
ax.set_yscale("log")
ax.set_xlim(0, 100)
ax.set_ylim(1, 1e4)
ax.set_xlabel("Time (μs)", fontsize=10)
ax.set_ylabel("Counts", fontsize=10)
clean_axes(ax)
save(fig, "01_raw_pmt_counts_Y")

# =========================
# 2. Load your spectral metadata
# =========================
# 修改为你的 npz 文件路径
META_PATH = Path("./data/processed/bbbc021_spectral_metadata/bbbc021_3ch_spectral_metadata_8269cells.npz")

if not META_PATH.exists():
    raise FileNotFoundError(f"Cannot find metadata file: {META_PATH}")

data = np.load(META_PATH, allow_pickle=True)

# 已知光谱响应矩阵 S: shape = [K, L]
S = data["S"].astype(float)

# 光谱波长: shape = [L]
wavelengths = data["wavelengths"].astype(float)

# 荧光组分名称
dye_names = [str(x).replace("_", "-") for x in data["channel_names"]]

eps = 1e-12

# =========================
# 3. L2 spectral normalization: S -> S_tilde
# =========================
# 这里的 S 就是你的已知光谱响应矩阵 S
# S_tilde 是逐行 L2 归一化后的光谱响应矩阵，用于构建 Gram 矩阵
S_tilde = S / (np.linalg.norm(S, axis=1, keepdims=True) + eps)

# ---------- 3.1 光谱归一化模块图 ----------
fig, ax = plt.subplots(figsize=(3.1, 2.05))

for k in range(S_tilde.shape[0]):
    ax.plot(wavelengths, S_tilde[k], lw=2.0, label=dye_names[k])

ax.set_xlim(wavelengths.min(), wavelengths.max())
ax.set_ylim(0, S_tilde.max() * 1.12)
ax.set_xlabel("Wavelength (nm)", fontsize=10)
ax.set_ylabel(r"L2-normalized response", fontsize=10)

# 如果你不想图里有图例，可以注释掉这一行
ax.legend(frameon=False, fontsize=7, loc="upper right", handlelength=1.6)

clean_axes(ax)
save(fig, "02_spectral_normalization")


# =========================
# 4. Spectral structure construction: C = S_tilde S_tilde^T
# =========================
# 推荐直接用 S_tilde 重新计算 C，保证和图中的 S_tilde 完全一致
C = S_tilde @ S_tilde.T

# 如果你想使用 npz 里保存的 cosine_similarity，也可以改成：
# if "cosine_similarity" in data.files:
#     C = data["cosine_similarity"].astype(float)
# else:
#     C = S_tilde @ S_tilde.T

# ---------- 4.1 光谱结构构建模块图：Gram 矩阵 ----------
fig, ax = plt.subplots(figsize=(2.65, 2.35))

im = ax.imshow(C, vmin=0, vmax=1)

ax.set_xticks(np.arange(len(dye_names)))
ax.set_yticks(np.arange(len(dye_names)))
ax.set_xticklabels(dye_names, rotation=35, ha="right", fontsize=7)
ax.set_yticklabels(dye_names, fontsize=7)

# 写入矩阵数值
for_i_range = range(C.shape[0])
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        ax.text(
            j, i,
            f"{C[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

ax.text(
    0.04, 1.05,
    r"$C=\tilde{\mathbf{S}}\tilde{\mathbf{S}}^{T}$",
    transform=ax.transAxes,
    fontsize=12,
    weight="bold"
)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)

save(fig, "02b_spectral_structure_gram_C")


# =========================
# 5. Eigen-decomposition based on your C
# =========================
eigvals, eigvecs = np.linalg.eigh(C)

# 从大到小排序，方便展示
sort_idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[sort_idx]
eigvecs = eigvecs[:, sort_idx]

m = np.arange(1, len(eigvals) + 1)

# ---------- 5.1 特征分解模块图 ----------
fig, ax = plt.subplots(figsize=(3.1, 2.05))

ax.bar(m, eigvals, width=0.62)

ax.set_xlim(0.4, len(eigvals) + 0.6)
ax.set_ylim(0, eigvals.max() * 1.15)
ax.set_xticks(m)
ax.set_xlabel(r"$m$", fontsize=11)
ax.set_ylabel(r"$\sigma_m$", fontsize=11)

# 如果想显示具体数值，保留这段；不想显示可删除
for x, val in zip(m, eigvals):
    ax.text(
        x, val,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

clean_axes(ax)
save(fig, "03_eigen_decomposition")


# =========================
# 6. Direction-related spatial weights
# =========================
eta = 1e-3
p = 1.0
w_max = 20.0

sigma_max = eigvals.max()
weights = ((sigma_max + eta) / (eigvals + eta)) ** p
weights = np.clip(weights, 1.0, w_max)

# ---------- 6.1 方向相关空间权重模块图 ----------
fig, ax = plt.subplots(figsize=(3.1, 2.05))

ax.bar(m, weights, width=0.62)

ax.set_xlim(0.4, len(weights) + 0.6)
ax.set_ylim(0, weights.max() * 1.25)
ax.set_xticks(m)
ax.set_xlabel(r"$m$", fontsize=11)
ax.set_ylabel(r"$w_m$", fontsize=11)

ax.text(
    0.08, 0.86,
    r"$w_m=f(\sigma_m)$",
    transform=ax.transAxes,
    fontsize=14,
    weight="bold"
)

# 如果想显示具体权重，保留这段；不想显示可删除
for x, val in zip(m, weights):
    ax.text(
        x, val,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

clean_axes(ax)
save(fig, "05_direction_spatial_weights")

rng = np.random.default_rng(23)
t2 = np.linspace(0, 100, 420)
pred = 1.8 + 6.5*np.exp(-((t2 - 18)/9)**2) + 3.7*np.exp(-((t2 - 43)/15)**2) + 1.4*np.exp(-((t2 - 72)/13)**2)
obs = np.clip(pred + rng.normal(0, 0.35, len(t2)) + 0.16*np.sin(t2/2.3), 0, None)
fig, ax = plt.subplots(figsize=(3.2, 2.1))
ax.plot(t2, obs, lw=1.0, alpha=0.95, label=r"$\mathbf{Y}$ observed")
ax.plot(t2, pred, lw=2.1, label=r"$\mathbf{\Lambda}(\mathbf{A})$ predicted")
ax.set_xlim(0, 100)
ax.set_ylim(0, max(pred)*1.18)
ax.set_xlabel("Time (μs)", fontsize=10)
ax.set_ylabel("Intensity", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper right", handlelength=2.2)
clean_axes(ax)
save(fig, "06_observation_consistency")
