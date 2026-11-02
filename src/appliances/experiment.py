"""Train once, freeze predictions, and select without using TEST outcomes."""

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
from .config import FOLDS, HORIZONS, BOUNDARIES, RELEASE, MODEL_DIR, RAW_SHA256
from .data import download_data, load_data, write_json, sha256
from .features import supervised, split_rows, feature_columns
from .modeling import CANDIDATES, fit_point, predict_point, fit_quantiles, predict_quantiles
from .metrics import point_metrics, calibration_margin, interval_metrics, alert_metrics

LABELS = [f"actual_{m}m_kWh" for m in HORIZONS]


def baseline(raw, rows, name):
    energy = raw.set_index("timestamp").energy_kWh
    times = pd.DatetimeIndex(rows.timestamp)
    if name == "persistence":
        return np.repeat(energy.reindex(times).to_numpy()[:, None], 6, axis=1)
    lag = pd.Timedelta(days=1 if name == "previous_day" else 7)
    return np.column_stack(
        [energy.reindex(times + pd.Timedelta(minutes=m) - lag).to_numpy() for m in HORIZONS]
    )


def run_experiment(output=RELEASE, model_dir=MODEL_DIR, device="cuda"):
    output, model_dir = Path(output), Path(model_dir)
    if output.exists() or model_dir.exists():
        raise FileExistsError(
            "Choose fresh result/model directories to preserve frozen experiments"
        )
    download_data()
    raw, audit = load_data()
    rows, all_columns = supervised(raw)
    parts, counts = split_rows(rows)
    output.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    write_json(output / "data_audit.json", audit)
    write_json(
        output / "protocol.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": RAW_SHA256,
            "boundaries": BOUNDARIES,
            "split_rows": counts,
            "folds": FOLDS,
            "horizons_minutes": HORIZONS,
            "seed": 42,
            "candidates": CANDIDATES,
            "groups": ["G0", "G1"],
            "selection": "Best OOF hourly MAE per family/group; among six winners choose lowest validation hourly MAE",
            "calibration": "Separate CAL, nonnegative CQR-style widening, nominal 80%; dependent time series, no guarantee",
            "threshold": "95th percentile of TRAIN actual next-hour energy, research only",
            "weather": "excluded because upstream interpolation and availability are not certified",
            "G1": "18 indoor/outdoor house sensor fields and lights, lagged one completed interval",
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "device": device,
            "test_used_for_selection": False,
            "savings_measured": False,
            "human_acceptance": "pending",
        },
    )
    rows[["timestamp", "forecast_end", "ready"]].to_csv(output / "row_manifest.csv", index=False)
    processed = output / "processed"
    processed.mkdir()
    for name, part in parts.items():
        part.to_csv(processed / f"{name}.csv", index=False)
    write_json(
        output / "features.json", {g: feature_columns(rows[all_columns], g) for g in ("G0", "G1")}
    )
    search, oof_frames, final_models = [], [], {}
    with threadpool_limits(limits=1):
        for group in ("G0", "G1"):
            columns = feature_columns(rows[all_columns], group)
            for candidate in CANDIDATES:
                name = group + "_" + candidate["name"]
                fold_actual, fold_pred = [], []
                for fold, (start, end) in enumerate(FOLDS, 1):
                    train = rows.loc[rows.ready & (rows.forecast_end < pd.Timestamp(start))]
                    valid = rows.loc[
                        rows.ready
                        & (rows.timestamp >= pd.Timestamp(start))
                        & (rows.forecast_end < pd.Timestamp(end))
                    ]
                    model, timing = fit_point(
                        train[columns], train[LABELS].to_numpy(), candidate, device
                    )
                    pred = predict_point(model, valid[columns])
                    actual = valid[LABELS].to_numpy()
                    fold_actual.append(actual)
                    fold_pred.append(pred)
                    frame = pd.DataFrame(
                        {"timestamp": valid.timestamp.to_numpy(), "model": name, "fold": fold}
                    )
                    for i, m in enumerate(HORIZONS):
                        frame[f"actual_{m}m_kWh"] = actual[:, i]
                        frame[f"predicted_{m}m_kWh"] = pred[:, i]
                    oof_frames.append(frame)
                    threshold = float(train.actual_hour_kWh.quantile(0.95))
                    search.append(
                        {
                            "model": name,
                            "family": candidate["family"],
                            "group": group,
                            "split": f"fold_{fold}",
                            **point_metrics(actual, pred, threshold),
                            **timing,
                        }
                    )
                train, val = parts["train"], parts["validation"]
                model, timing = fit_point(
                    train[columns], train[LABELS].to_numpy(), candidate, device
                )
                final_models[name] = (model, columns, candidate)
                threshold = float(train.actual_hour_kWh.quantile(0.95))
                score = point_metrics(
                    val[LABELS].to_numpy(), predict_point(model, val[columns]), threshold
                )
                search.append(
                    {
                        "model": name,
                        "family": candidate["family"],
                        "group": group,
                        "split": "validation",
                        **score,
                        **timing,
                    }
                )
                pooled = point_metrics(
                    np.concatenate(fold_actual), np.concatenate(fold_pred), threshold
                )
                search.append(
                    {
                        "model": name,
                        "family": candidate["family"],
                        "group": group,
                        "split": "oof",
                        **pooled,
                        "fit_seconds": None,
                        "devices": [],
                    }
                )
                print(
                    f'{name}: OOF hour MAE={pooled["mae_hour_kWh"]:.5f}; VAL={score["mae_hour_kWh"]:.5f}',
                    flush=True,
                )
        table = pd.DataFrame(
            [{k: v for k, v in s.items() if k not in ("horizons", "devices")} for s in search]
        )
        shortlist = (
            table[table.split.eq("oof")]
            .sort_values(["mae_hour_kWh", "model"])
            .groupby(["family", "group"])
            .head(1)
            .model.tolist()
        )
        winner = (
            table[table.model.isin(shortlist) & table.split.eq("validation")]
            .sort_values(["mae_hour_kWh", "model"])
            .iloc[0]
            .model
        )
        model, columns, candidate = final_models[winner]
        selection = {
            "winner": winner,
            "candidate": candidate,
            "features": columns,
            "shortlist": shortlist,
            "locked_before_test_scoring": True,
            "criterion": "OOF shortlist, then minimum validation hourly MAE",
        }
        write_json(output / "selection.json", selection)
        write_json(output / "search_full.json", search)
        table.to_csv(output / "search_metrics.csv", index=False)
        pd.concat(oof_frames, ignore_index=True).to_csv(output / "oof_predictions.csv", index=False)
        ablations = []
        for label, excluded in [
            ("without_day_lag", {"energy_lag_144"}),
            ("without_week_lag", {"energy_lag_1008"}),
            ("without_rolling", {"sum_1h", "mean_4h", "std_4h", "mean_24h", "max_24h"}),
        ]:
            subset = [c for c in columns if c not in excluded]
            fitted, timing = fit_point(
                parts["train"][subset], parts["train"][LABELS].to_numpy(), candidate, device
            )
            score = point_metrics(
                parts["validation"][LABELS].to_numpy(),
                predict_point(fitted, parts["validation"][subset]),
                float(parts["train"].actual_hour_kWh.quantile(0.95)),
            )
            ablations.append({"ablation": label, "features": len(subset), **score, **timing})
        write_json(output / "ablation.json", ablations)
        if device == "cuda":
            benchmark = {}
            gpu_candidate = next(c for c in CANDIDATES if c["name"] == "xgb_3")
            for backend in ("cpu", "cuda"):
                fitted, timing = fit_point(
                    parts["train"][columns],
                    parts["train"][LABELS].to_numpy(),
                    gpu_candidate,
                    backend,
                )
                benchmark[backend] = {
                    "timing": timing,
                    "validation": point_metrics(
                        parts["validation"][LABELS].to_numpy(),
                        predict_point(fitted, parts["validation"][columns]),
                        float(parts["train"].actual_hour_kWh.quantile(0.95)),
                    ),
                }
            benchmark["scope"] = (
                "One paired measurement on the same data/configuration; does not measure electrical energy or financial cost"
            )
            write_json(output / "gpu_benchmark.json", benchmark)
        quantiles = fit_quantiles(
            parts["train"][columns], parts["train"].actual_hour_kWh.to_numpy()
        )
        q_cal = predict_quantiles(quantiles, parts["calibration"][columns])
        margin = calibration_margin(parts["calibration"].actual_hour_kWh.to_numpy(), q_cal)
        threshold = float(parts["train"].actual_hour_kWh.quantile(0.95))
        bundle = {
            "contract": "appliances_6x10min_v1",
            "point": model,
            "quantiles": quantiles,
            "features": columns,
            "selection": selection,
            "margin": margin,
            "threshold": threshold,
            "raw_sha256": RAW_SHA256,
            "versions": {"sklearn": sklearn.__version__},
        }
        joblib.dump(bundle, model_dir / "forecast.joblib", compress=3)
        model_hash = sha256(model_dir / "forecast.joblib")
        write_json(
            model_dir / "manifest.json",
            {
                "model_sha256": model_hash,
                "raw_sha256": RAW_SHA256,
                "contract": bundle["contract"],
                "features": columns,
                "winner": winner,
            },
        )
        predictions = output / "predictions"
        predictions.mkdir()
        metrics = {}
        for split, part in parts.items():
            actual = part[LABELS].to_numpy()
            pred = predict_point(model, part[columns])
            q = predict_quantiles(quantiles, part[columns])
            frame = part[["timestamp", "forecast_end"] + LABELS].copy()
            for i, m in enumerate(HORIZONS):
                frame[f"predicted_{m}m_kWh"] = pred[:, i]
            frame["actual_hour_kWh"] = actual.sum(axis=1)
            frame["predicted_hour_kWh"] = pred.sum(axis=1)
            for i, level in enumerate((10, 50, 90)):
                frame[f"p{level}_hour_kWh"] = q[:, i]
            frame["lower_hour_kWh"] = np.maximum(0, q[:, 0] - margin)
            frame["upper_hour_kWh"] = q[:, 2] + margin
            frame.to_csv(predictions / f"{split}.csv", index=False)
            metrics[split] = {
                "point": point_metrics(actual, pred, threshold),
                "interval": interval_metrics(actual.sum(axis=1), q, margin, threshold),
                "alert": alert_metrics(
                    actual.sum(axis=1), frame.upper_hour_kWh.to_numpy(), threshold
                ),
            }
        comparison = []
        for split in ("validation", "test"):
            part = parts[split]
            actual = part[LABELS].to_numpy()
            for name in ("persistence", "previous_day", "previous_week"):
                comparison.append(
                    {
                        "split": split,
                        "model": name,
                        **point_metrics(actual, baseline(raw, part, name), threshold),
                    }
                )
            for name, (other, cols, _) in final_models.items():
                comparison.append(
                    {
                        "split": split,
                        "model": name,
                        **point_metrics(actual, predict_point(other, part[cols]), threshold),
                    }
                )
        write_json(output / "metrics.json", metrics)
        pd.DataFrame([{k: v for k, v in s.items() if k != "horizons"} for s in comparison]).to_csv(
            output / "comparison.csv", index=False
        )
        write_json(output / "comparison_full.json", comparison)
        residual = abs(
            parts["calibration"].actual_hour_kWh.to_numpy()
            - predict_point(model, parts["calibration"][columns]).sum(axis=1)
        )
        write_json(
            model_dir / "monitor_reference.json",
            {
                "calibration_mae_kWh": float(residual.mean()),
                "calibration_coverage": metrics["calibration"]["interval"]["coverage"],
                "window_rows": 144,
                "rule": "Review after two consecutive nonoverlapping daily windows with MAE > 1.5x CAL or coverage < CAL - .1; heuristic",
            },
        )
        write_json(
            output / "summary.json",
            {
                "winner": winner,
                "features": len(columns),
                "threshold_kWh": threshold,
                "calibration_margin_kWh": margin,
                "split_rows": counts,
                "test": metrics["test"],
                "model_sha256": model_hash,
                "raw_sha256": RAW_SHA256,
                "training": "Ridge/HGB CPU; XGBoost " + device,
                "savings_measured": False,
            },
        )
    freeze_results(output, model_dir)
    print(
        json.dumps(json.loads((output / "summary.json").read_text(encoding="utf-8")), indent=2),
        flush=True,
    )


def freeze_results(output=RELEASE, model_dir=MODEL_DIR):
    output, model_dir = Path(output), Path(model_dir)
    paths = [p for p in output.rglob("*") if p.is_file() and p.name != "results_lock.json"]
    paths += [p for p in model_dir.rglob("*") if p.is_file()]
    from .config import ROOT

    write_json(
        output / "results_lock.json",
        {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(paths)},
    )
