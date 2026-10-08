import numpy as np


def compute_abundance_metrics(A_hat, A_true, eps=1e-8):
    """
    A_hat: [H, W, K]
    A_true: [H, W, K]

    return:
        MAE_A
        RMSE_A
        NRMSE_A
        Pearson_A
    """

    A_hat = np.asarray(A_hat, dtype=np.float32)
    A_true = np.asarray(A_true, dtype=np.float32)

    assert A_hat.shape == A_true.shape, (
        f"A_hat shape {A_hat.shape} != A_true shape {A_true.shape}"
    )

    diff = A_hat - A_true

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    # 推荐用 true abundance 的 RMS 做归一化
    denom = np.sqrt(np.mean(A_true ** 2)) + eps
    nrmse = rmse / denom

    # 每个 dye 单独算 Pearson，然后平均
    K = A_true.shape[-1]
    pearsons = []

    for k in range(K):
        x = A_true[..., k].reshape(-1)
        y = A_hat[..., k].reshape(-1)

        x_std = np.std(x)
        y_std = np.std(y)

        if x_std < eps or y_std < eps:
            continue

        r = np.corrcoef(x, y)[0, 1]
        pearsons.append(r)

    if len(pearsons) == 0:
        pearson = np.nan
    else:
        pearson = float(np.nanmean(pearsons))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "nrmse": float(nrmse),
        "pearson": float(pearson),
    }