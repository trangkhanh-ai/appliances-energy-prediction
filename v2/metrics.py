"""Sai số nhóm cao và ngưỡng cảnh báo được chọn ngoài tập đánh giá."""
import numpy as np
from sklearn.metrics import mean_pinball_loss
import appliances_core as v1


def regression(y, prediction, threshold):
    y, p = np.asarray(y, float), np.asarray(prediction, float)
    error = p - y
    high = y > threshold
    result = {"n": len(y), "mae": float(np.mean(np.abs(error))),
              "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean()),
              "high_n": int(high.sum()), "high_episodes": len(v1.runs(high))}
    for name, values in (("mae", np.abs(error)), ("rmse", error ** 2),
                         ("bias", error), ("underprediction_rate", (p < y).astype(float))):
        value = float(values[high].mean()) if high.any() else None
        result["high_" + name] = float(np.sqrt(value)) if name == "rmse" and value is not None else value
    return result


def threshold_for_fpr(y, score, energy_threshold, target_fpr=.10):
    negatives = np.asarray(score)[np.asarray(y) <= energy_threshold]
    if not len(negatives):
        raise ValueError("Không có mẫu âm để chọn ngưỡng cảnh báo")
    return float(np.quantile(negatives, 1 - target_fpr, method="higher"))


def threshold_for_recall(y, score, energy_threshold, target_recall=.70):
    positives = np.sort(np.asarray(score)[np.asarray(y) > energy_threshold])
    if not len(positives):
        raise ValueError("Không có mẫu cao để chọn ngưỡng recall")
    count = max(1, int(np.ceil(len(positives) * target_recall)))
    return float(np.nextafter(positives[-count], -np.inf))


def alerts(y, score, energy_threshold, score_threshold=None):
    score_threshold = energy_threshold if score_threshold is None else score_threshold
    y, score = np.asarray(y), np.asarray(score)
    result = v1.alert_metrics(y, score, energy_threshold)
    truth, alarm = y > energy_threshold, score > score_threshold
    tp, fp = int((truth & alarm).sum()), int((~truth & alarm).sum())
    fn, tn = int((truth & ~alarm).sum()), int((~truth & ~alarm).sum())
    precision, recall = tp/max(tp+fp, 1), tp/max(tp+fn, 1)
    result.update(tp=tp, fp=fp, fn=fn, tn=tn, alerts=tp+fp, precision=precision,
                  recall=recall, f1=2*precision*recall/max(precision+recall, 1e-12))
    result["energy_threshold"] = float(energy_threshold)
    result["score_threshold"] = float(score_threshold)
    result["fpr"] = result["fp"] / max(result["fp"] + result["tn"], 1)
    result["episodes"] = v1.episode_metrics(np.asarray(y) > energy_threshold, np.asarray(score) > score_threshold)
    result["repeated_alert_origins"] = result["alerts"] - result["episodes"]["alert_episodes"]
    return result


def interval(y, quantiles, margin, energy_threshold):
    y, q = np.asarray(y), np.asarray(quantiles)
    low, high = np.maximum(q[:, 0] - margin, 0), q[:, 2] + margin
    covered = (y >= low) & (y <= high)
    subset = y > energy_threshold
    return {"n": len(y), "coverage": float(covered.mean()), "mean_width": float(np.mean(high-low)),
            "high_coverage": float(covered[subset].mean()) if subset.any() else None,
            "high_mean_width": float(np.mean((high-low)[subset])) if subset.any() else None,
            "pinball": [float(mean_pinball_loss(y, q[:, i], alpha=a)) for i, a in enumerate(v1.QUANTILES)]}


def block_bootstrap(y, candidate, baseline, block=144, repeats=1000, seed=42):
    e = np.abs(np.asarray(y) - candidate)
    b = np.abs(np.asarray(y) - baseline)
    rng = np.random.default_rng(seed)
    maes, gains = [], []
    for _ in range(repeats):
        starts = rng.integers(0, len(e), int(np.ceil(len(e)/block)))
        index = ((starts[:, None] + np.arange(block)) % len(e)).ravel()[:len(e)]
        mae, base = e[index].mean(), b[index].mean()
        maes.append(mae)
        gains.append(100 * (1 - mae/base))
    return {"block": block, "replicates": repeats,
            "mae_percentile95": np.quantile(maes, [.025, .975]).tolist(),
            "gain_percent_percentile95": np.quantile(gains, [.025, .975]).tolist(),
            "scope": "Descriptive paired block resampling; frozen models; overlapping targets; not an external validation"}
