"""Small fixed search; CPU deployment also works for CUDA-trained XGBoost."""

import json
import time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

CANDIDATES = (
    {"name": "ridge_1", "family": "ridge", "alpha": 1.0},
    {"name": "ridge_100", "family": "ridge", "alpha": 100.0},
    {"name": "hgb_15", "family": "hgb", "leaves": 15},
    {"name": "hgb_31", "family": "hgb", "leaves": 31},
    {"name": "xgb_3", "family": "xgb", "depth": 3},
    {"name": "xgb_5", "family": "xgb", "depth": 5},
)


def fit_point(x, y, candidate, device="cuda"):
    x = np.asarray(x, dtype=float)
    models, device_logs = [], []
    start = time.perf_counter()
    if candidate["family"] == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=candidate["alpha"]))
        model.fit(x, y)
        models = [model]
    else:
        for i in range(y.shape[1]):
            if candidate["family"] == "hgb":
                model = HistGradientBoostingRegressor(
                    loss="absolute_error",
                    max_iter=180,
                    learning_rate=0.08,
                    max_leaf_nodes=candidate["leaves"],
                    l2_regularization=3.0,
                    early_stopping=False,
                    random_state=42,
                )
            else:
                from xgboost import XGBRegressor

                model = XGBRegressor(
                    objective="reg:absoluteerror",
                    n_estimators=180,
                    learning_rate=0.08,
                    max_depth=candidate["depth"],
                    min_child_weight=20,
                    reg_lambda=5.0,
                    n_jobs=1,
                    tree_method="hist",
                    device=device,
                    random_state=42,
                )
            model.fit(x, y[:, i])
            if candidate["family"] == "xgb":
                config = json.loads(model.get_booster().save_config())
                actual_device = config["learner"]["generic_param"]["device"]
                if device.startswith("cuda") and not actual_device.startswith("cuda"):
                    raise RuntimeError("CUDA training requested but CPU fallback occurred")
                device_logs.append(actual_device)
                model.set_params(device="cpu")
            models.append(model)
    return {"family": candidate["family"], "models": models}, {
        "fit_seconds": time.perf_counter() - start,
        "devices": device_logs,
    }


def predict_point(bundle, x):
    x = np.asarray(x, dtype=float)
    if bundle["family"] == "ridge":
        prediction = bundle["models"][0].predict(x)
    else:
        prediction = np.column_stack([model.predict(x) for model in bundle["models"]])
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (len(x), 6) or not np.isfinite(prediction).all():
        raise ValueError("Invalid model output")
    return np.maximum(prediction, 0.0)


def fit_quantiles(x, actual):
    models = []
    for q in (0.1, 0.5, 0.9):
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=3.0,
            early_stopping=False,
            random_state=42,
        )
        model.fit(np.asarray(x, dtype=float), actual)
        models.append(model)
    return models


def predict_quantiles(models, x):
    raw = np.column_stack([model.predict(np.asarray(x, dtype=float)) for model in models])
    if not np.isfinite(raw).all():
        raise ValueError("Invalid quantile output")
    return np.sort(np.maximum(raw, 0.0), axis=1)
