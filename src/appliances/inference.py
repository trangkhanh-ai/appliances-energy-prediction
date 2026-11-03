"""Measured-history inference, with artifact integrity checks."""

import json
import time
from pathlib import Path
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from .config import MODEL_DIR, RAW_SHA256, HORIZONS
from .data import sha256
from .features import features_at_issue
from .modeling import predict_point, predict_quantiles


class Forecaster:
    def __init__(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        self.manifest = json.loads(
            (model_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if sha256(model_dir / "forecast.joblib") != self.manifest["model_sha256"]:
            raise ValueError("Model hash mismatch")
        self.bundle = joblib.load(model_dir / "forecast.joblib")
        if (
            self.bundle["contract"] != "appliances_6x10min_v1"
            or self.bundle["raw_sha256"] != RAW_SHA256
        ):
            raise ValueError("Incompatible model or dataset")
        if self.bundle["features"] != self.manifest["features"]:
            raise ValueError("Feature contract mismatch")

    def predict(self, history):
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            x = features_at_issue(history, self.bundle["features"])
            pred = predict_point(self.bundle["point"], x)[0]
            quant = predict_quantiles(self.bundle["quantiles"], x)[0]
        t = pd.Timestamp(history.timestamp.iloc[-1])
        lower = max(0.0, float(quant[0] - self.bundle["margin"]))
        upper = float(quant[2] + self.bundle["margin"])
        return {
            "issue_time": t.isoformat(),
            "forecast_end": (t + pd.Timedelta(minutes=60)).isoformat(),
            "horizons": [
                {
                    "time": (t + pd.Timedelta(minutes=m)).isoformat(),
                    "minutes": m,
                    "predicted_kWh": float(v),
                }
                for m, v in zip(HORIZONS, pred)
            ],
            "predicted_hour_kWh": float(pred.sum()),
            "p10_hour_kWh": float(quant[0]),
            "p50_hour_kWh": float(quant[1]),
            "p90_hour_kWh": float(quant[2]),
            "lower_hour_kWh": lower,
            "upper_hour_kWh": upper,
            "threshold_kWh": self.bundle["threshold"],
            "high_consumption_warning": upper > self.bundle["threshold"],
            "actual_hour_kWh": None,
            "model": self.manifest["winner"],
            "model_sha256": self.manifest["model_sha256"],
            "inference_ms": (time.perf_counter() - started) * 1000,
            "scope": "Research threshold from TRAIN; calibrated interval has no guaranteed time-series coverage",
        }
