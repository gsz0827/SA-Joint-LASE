import os
import random
import numpy as np
import torch

SEED = 2026
EVAL_SPLIT = "test"
MAX_SAMPLES = None

OUT_DIR = "results_main_exp_practical_lase_44spots"
os.makedirs(OUT_DIR, exist_ok=True)

EVAL_IDS_PATH = os.path.join(
    OUT_DIR,
    f"eval_ids_{EVAL_SPLIT}_{MAX_SAMPLES}_44spots_practical.npy"
)

EPS = 1e-8


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_or_create_eval_ids(all_ids):
    """
    all_ids: 当前 split 里面所有样本 id
    返回固定 eval ids，保证 two-step 和 joint inversion 使用同一批样本
    """
    all_ids = np.array(all_ids)

    if os.path.exists(EVAL_IDS_PATH):
        eval_ids = np.load(EVAL_IDS_PATH, allow_pickle=True)
        print(f"[INFO] Loaded eval ids from {EVAL_IDS_PATH}")
    else:
        rng = np.random.default_rng(SEED)

        if MAX_SAMPLES is None:
            eval_ids = all_ids
        elif len(all_ids) > MAX_SAMPLES:
            eval_ids = rng.choice(all_ids, size=MAX_SAMPLES, replace=False)
        else:
            eval_ids = all_ids

        np.save(EVAL_IDS_PATH, eval_ids)
        print(f"[INFO] Saved eval ids to {EVAL_IDS_PATH}")

    return list(eval_ids)