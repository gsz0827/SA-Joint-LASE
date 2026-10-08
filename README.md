# BBBC021 — SA-Joint 多光谱 LASE 联合丰度反演实验

本目录对应论文 **《Spectral-Correlation-Aware Joint Abundance Inversion for LASE-Based Multispectral Imaging Flow Cytometry》（SA-Joint）** 的完整实验代码与数据。

## 目录结构

```
BBBC021_download/
├── README.md                              # 本说明
├── main_exp_config_practical_44spots.py   # 主配置：种子 / 数据集划分 / 结果输出目录
├── main_exp_metrics.py                    # 指标计算（MAE/RMSE/NRMSE/Pearson）
│
├── 01_list_files.py                       # 数据准备：列出 BBBC021 文件清单
├── 02_download_many_zips.py               # 数据准备：批量下载 BBBC021 原始 zip
├── 03_unzip_all_zips.py                   # 数据准备：解压全部 zip
├── 04_find_triplets.py                    # 数据准备：查找荧光三通道 triplet
├── 05_preview_triplet.py                  # 数据准备：triplet 预览
├── 06_segment_and_crop_single_cells.py    # 数据准备：分割并裁剪单细胞 patches
├── 07_build_bbbc021_spectral_metadata_large_revised.py   # 生成光谱元数据（S 矩阵、cosine 相似度）
├── 08_simulate_bbbc021_lase_waveforms_practical_44spots.py # LASE 正演模拟（44 光斑）
├── 09_invert_bbbc021_spectral_ambiguity_tv_44spots_final_ablation_fulltest.py # SA-Joint 联合反演（核心）
├── 10_two_step_bbbc021_decode_then_unmix_44spots.py      # RU 两阶段对比基线
├── 19_run_spectral_ambiguity_tv_sweep_20samples_1000iters.py  # 调参扫描（20 样本粗扫）
├── 20_run_spectral_ambiguity_tv_fine_sweep_20samples_1000iters.py # 调参扫描（20 样本精扫）
├── 21_run_final_ablation_20samples_1000iters.py          # 消融实验运行器（20 样本试跑）
├── 22_run_final_ablation_full_test_1000iters.py          # 消融实验运行器（全量测试）
├── 23_collect_main_results_spectral_ambiguity_joint_44spots.py # 汇总主结果
│
├── 24_plot_complete_metrics_practical_44spots_final.py   # 图 4-1：整体指标箱线图
├── 25_plot_per_dye_nrmse_practical_44spots_final.py      # 图 4-2：各荧光组分 NRMSE
├── 26_plot_representative_abundance_grouped_full_44spots_final.py # 图 4-3：代表性样本丰度图
├── 27_find_abnormal_samples_practical_44spots_final.py   # 异常样本诊断
├── 28_plot_joint_ablation_metrics_44spots_final.py       # 图 4-4：消融实验对比
├── plot_4_2.py                             # 图 4-2 光谱相关结构图（单独出图）
├── generate_sa_joint_module_assets.py      # 方法示意图资产生成
│
├── data/processed/
│   ├── bbbc021_hyperspectral_dataset/      # 16 波段高光谱图像（X，294 cells，128×128×16）
│   ├── bbbc021_single_cell_patches/        # 裁剪的单细胞丰度 patches（A，294/8269 cells）
│   ├── bbbc021_lase_waveforms/             # LASE 波形（Y，294 cells，Tx16）
│   ├── bbbc021_spectral_metadata/          # 光谱元数据（S 矩阵等）
│   └── bbbc021_lase_train_ready_44spots/   # 核心输入数据（44 光斑 LASE 波形）
│
├── results_main_exp_practical_lase_44spots/ # 最终结果 + 论文图（唯一正式结果目录）
├── figures/                                # 图 4-2 光谱相关结构图
├── sa_joint_module_assets/                 # 方法示意图资产
│
├── results/                                # 数据准备阶段的可视化预览
│   ├── bbbc021_lase_train_ready_preview_44spots/
│   ├── bbbc021_single_cell_preview/
│   └── bbbc021_spectral_metadata_preview/
│
└── _archive/                               # 历史遗留（已归档，非当前实验）
    └── exploratory_scripts/                # 诊断脚本（00）
```

## 实验流程

```
BBBC021 原始图像 (02 下载 → 03 解压)
   └─ 06_segment_and_crop_single_cells.py  → single_cell_patches (A)
         └─ 07_build_bbbc021_spectral_metadata...  → spectral_metadata (S 矩阵)
               └─ 08_simulate_bbbc021_lase_waveforms... → lase_train_ready_44spots
                     ├─ 09_invert... (SA-Joint 联合反演)  ─┐
                     ├─ 10_two_step... (RU 两阶段基线)     ├→ results_main_exp_practical_lase_44spots
                     └─ 21/22 消融运行器                   ┘
                           └─ 23 汇总 → 24/25/26/28 绘图
```

## 复现要点

1. **数据**：直接用 `data/processed/` 下的成品数据即可复现，无需重新下载 BBBC021 原始图像。完整数据处理链（`01`~`08`）脚本均保留在主目录，原始 TIF 图像可按需重新下载生成。
2. **反演**：`09` 是 SA-Joint 主方法，`10` 是两阶段对比基线，二者共用 `main_exp_config_practical_44spots.py` 的种子与样本划分。
3. **消融**：`21/22` 运行消融设置（Poisson only / Standard TV only / SA-TV only / SA-Joint）。
4. **绘图**：`23` 汇总 npz 后，`24/25/26/28` 生成论文图 4-1 ~ 4-4。
5. **运行环境**：`.venv/`（Python 3.11，依赖：`torch` 2.12、`numpy` 2.4、`scipy` 1.17、`matplotlib` 3.11、`pandas` 3.0、`pillow`）。

## 关键参数

| 参数 | 值 |
|---|---|
| 单细胞图像尺寸 | 128 × 128 |
| 光斑数 | 44 |
| 光谱波段数 | 16 |
| 荧光组分数 | 3（SYTO16-like / DIA-like / DeepRed-like） |
| 视场 | 30 μm |
| 迭代次数 | 1000 |
| 随机种子 | 2026 |
