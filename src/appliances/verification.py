"""Independent reconciliation of source, labels, metrics and online inference."""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ROOT, RELEASE, MODEL_DIR, HORIZONS
from .data import load_data, sha256
from .features import supervised, split_rows, make_features, features_at_issue
from .inference import Forecaster
from .metrics import point_metrics, interval_metrics, alert_metrics


def assert_nested_close(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_close(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_close(a, b)
    elif isinstance(left, (float, int)) and not isinstance(left, bool):
        np.testing.assert_allclose(left, right, atol=1e-9, rtol=1e-9)
    else:
        assert left == right, (left, right)


def verify(output=RELEASE, model_dir=MODEL_DIR):
    output, model_dir = Path(output), Path(model_dir)
    lock = json.loads((output / "results_lock.json").read_text(encoding="utf-8"))
    for relative, expected in lock.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen artifact changed: {relative}")
    raw, audit = load_data()
    rows, columns = supervised(raw)
    parts, counts = split_rows(rows)
    model = Forecaster(model_dir)
    threshold = model.bundle["threshold"]
    stored = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    meter = raw.set_index("timestamp").energy_kWh
    predictions = {}
    for split, part in parts.items():
        frame = pd.read_csv(
            output / "predictions" / f"{split}.csv",
            parse_dates=["timestamp", "forecast_end"],
            float_precision="round_trip",
        )
        pd.testing.assert_series_equal(
            frame.timestamp, part.timestamp, check_names=False
        )
        actual = np.column_stack(
            [
                meter.reindex(frame.timestamp + pd.Timedelta(minutes=m)).to_numpy()
                for m in HORIZONS
            ]
        )
        saved = frame[[f"actual_{m}m_kWh" for m in HORIZONS]].to_numpy()
        np.testing.assert_allclose(actual, saved, rtol=0, atol=1e-12)
        pred = frame[[f"predicted_{m}m_kWh" for m in HORIZONS]].to_numpy()
        quant = frame[["p10_hour_kWh", "p50_hour_kWh", "p90_hour_kWh"]].to_numpy()
        computed = {
            "point": point_metrics(actual, pred, threshold),
            "interval": interval_metrics(
                actual.sum(axis=1), quant, model.bundle["margin"], threshold
            ),
            "alert": alert_metrics(
                actual.sum(axis=1), frame.upper_hour_kWh.to_numpy(), threshold
            ),
        }
        assert_nested_close(computed, stored[split])
        predictions[split] = frame
    leakage = []
    for index in (1200, 7000, 15000):
        changed = raw.copy()
        changed.loc[index + 1, "energy_kWh"] += 1.0
        before = make_features(raw).iloc[index][columns].to_numpy()
        after = make_features(changed).iloc[index][columns].to_numpy()
        np.testing.assert_array_equal(before, after)
        original_rows, _ = supervised(raw)
        changed_rows, _ = supervised(changed)
        np.testing.assert_allclose(
            changed_rows.actual_hour_kWh.iloc[index]
            - original_rows.actual_hour_kWh.iloc[index],
            1.0,
        )
        leakage.append(
            {
                "index": index,
                "future_input_unchanged": True,
                "target_changes_correctly": True,
            }
        )
    timings = []
    test = predictions["test"]
    sample_indices = np.unique(np.linspace(0, len(test) - 1, 100, dtype=int))
    for n, index in enumerate(sample_indices):
        t = test.timestamp.iloc[index]
        history = raw.loc[raw.timestamp <= t].tail(1009)
        begin = time.perf_counter()
        payload = model.predict(history)
        elapsed = (time.perf_counter() - begin) * 1000
        if n >= 5:
            timings.append(elapsed)
        np.testing.assert_allclose(
            [r["predicted_kWh"] for r in payload["horizons"]],
            test.iloc[index][[f"predicted_{m}m_kWh" for m in HORIZONS]].to_numpy(
                dtype=float
            ),
            rtol=1e-6,
            atol=1e-9,
        )
        interval_columns = [
            "p10_hour_kWh",
            "p50_hour_kWh",
            "p90_hour_kWh",
            "lower_hour_kWh",
            "upper_hour_kWh",
        ]
        np.testing.assert_allclose(
            [payload[c] for c in interval_columns],
            test.iloc[index][interval_columns].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )
        assert payload["actual_hour_kWh"] is None
        full = (
            make_features(raw)
            .loc[raw.timestamp.eq(t), model.bundle["features"]]
            .to_numpy()
        )
        np.testing.assert_allclose(
            features_at_issue(history, model.bundle["features"]).to_numpy(),
            full,
            rtol=1e-12,
            atol=1e-12,
        )
    result = {
        "passed": True,
        "frozen_files_checked": len(lock),
        "all_label_rows_reconciled": counts,
        "causality_checks": leakage,
        "inference_origins_checked": len(sample_indices),
        "inference_p50_ms": float(np.quantile(timings, 0.5)),
        "inference_p95_ms": float(np.quantile(timings, 0.95)),
        "benchmark_scope": "Features plus point and quantile inference, warm CPU, excludes HTTP/UI/SQLite/model loading",
        "test_used_for_selection": False,
        "real_savings_measured": False,
        "human_group_acceptance": "pending",
    }
    return result
