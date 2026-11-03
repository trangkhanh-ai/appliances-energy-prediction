"""All reported quantities are recomputed from measurements and predictions."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    mean_pinball_loss,
)
from .config import HORIZONS, QUANTILES


def point_metrics(actual, predicted, threshold):
    actual, predicted = np.asarray(actual), np.asarray(predicted, dtype=float)
    if (
        actual.shape != predicted.shape
        or actual.shape[1] != 6
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("Expected six finite predictions per origin")
    error = predicted - actual
    total_error = error.sum(axis=1)
    high = actual.sum(axis=1) > threshold
    return {
        "rows": len(actual),
        "mae_step_kWh": float(abs(error).mean()),
        "rmse_step_kWh": float(np.sqrt(np.mean(error**2, axis=0)).mean()),
        "mae_hour_kWh": float(abs(total_error).mean()),
        "rmse_hour_kWh": float(np.sqrt(np.mean(total_error**2))),
        "bias_hour_kWh": float(total_error.mean()),
        "high_rows": int(high.sum()),
        "high_mae_hour_kWh": (
            float(abs(total_error[high]).mean()) if high.any() else None
        ),
        "horizons": [
            {
                "minutes": m,
                "mae_kWh": float(abs(error[:, i]).mean()),
                "rmse_kWh": float(np.sqrt(np.mean(error[:, i] ** 2))),
            }
            for i, m in enumerate(HORIZONS)
        ],
    }


def calibration_margin(actual, quantiles):
    score = np.maximum.reduce(
        [quantiles[:, 0] - actual, actual - quantiles[:, 2], np.zeros(len(actual))]
    )
    level = min(1.0, np.ceil((len(score) + 1) * 0.8) / len(score))
    return float(np.quantile(score, level, method="higher"))


def interval_metrics(actual, quantiles, margin, threshold):
    lower = np.maximum(0, quantiles[:, 0] - margin)
    upper = quantiles[:, 2] + margin
    inside = (actual >= lower) & (actual <= upper)
    high = actual > threshold
    return {
        "coverage": float(inside.mean()),
        "width_kWh": float(np.mean(upper - lower)),
        "high_coverage": float(inside[high].mean()) if high.any() else None,
        "pinball": {
            str(q): float(mean_pinball_loss(actual, quantiles[:, i], alpha=q))
            for i, q in enumerate(QUANTILES)
        },
        "nominal_coverage": 0.8,
        "guaranteed": False,
    }


def alert_metrics(actual, upper, threshold):
    positive, alert = actual > threshold, upper > threshold
    tp = int(np.sum(positive & alert))
    fp = int(np.sum(~positive & alert))
    fn = int(np.sum(positive & ~alert))
    tn = int(np.sum(~positive & ~alert))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    pr, re, _ = precision_recall_curve(positive, upper)
    both = len(np.unique(positive)) == 2
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_discovery_rate": fp / (tp + fp) if tp + fp else 0.0,
        "positive_prevalence": float(positive.mean()),
        "average_precision": (
            float(average_precision_score(positive, upper)) if both else None
        ),
        "pr_auc_trapezoid": float(auc(re, pr)) if both else None,
        "roc_auc_ranking": float(roc_auc_score(positive, upper)) if both else None,
        "definition": "Overlapping forecast origins, not independent physical events; no verified intervention lead time",
    }
