"""Predeclared feature removals and paired hardware timings on TRAIN/VAL only."""

import json
import subprocess
from threadpoolctl import threadpool_limits
from src.appliances.config import RELEASE, HORIZONS
from src.appliances.data import load_data, write_json
from src.appliances.features import supervised, split_rows
from src.appliances.modeling import CANDIDATES, fit_point, predict_point
from src.appliances.metrics import point_metrics


def run():
    raw, _ = load_data()
    parts, _ = split_rows(supervised(raw)[0])
    choice = json.loads((RELEASE / "selection.json").read_text(encoding="utf-8"))
    labels = [f"actual_{m}m_kWh" for m in HORIZONS]
    columns = choice["features"]
    candidate = choice["candidate"]
    threshold = float(parts["train"].actual_hour_kWh.quantile(0.95))
    with threadpool_limits(limits=1):
        if not (RELEASE / "ablation.json").exists():
            results = []
            for label, excluded in [
                ("without_day_lag", {"energy_lag_144"}),
                ("without_week_lag", {"energy_lag_1008"}),
                (
                    "without_rolling",
                    {"sum_1h", "mean_4h", "std_4h", "mean_24h", "max_24h"},
                ),
            ]:
                subset = [c for c in columns if c not in excluded]
                fitted, timing = fit_point(
                    parts["train"][subset],
                    parts["train"][labels].to_numpy(),
                    candidate,
                    "cuda",
                )
                result = {
                    "ablation": label,
                    "features": len(subset),
                    **point_metrics(
                        parts["validation"][labels].to_numpy(),
                        predict_point(fitted, parts["validation"][subset]),
                        threshold,
                    ),
                    **timing,
                }
                results.append(result)
                print(label, result["mae_hour_kWh"], flush=True)
            write_json(RELEASE / "ablation.json", results)
        if not (RELEASE / "gpu_benchmark.json").exists():
            benchmark = {}
            c = next(c for c in CANDIDATES if c["name"] == "xgb_3")
            for backend in ("cpu", "cuda"):
                fitted, timing = fit_point(
                    parts["train"][columns],
                    parts["train"][labels].to_numpy(),
                    c,
                    backend,
                )
                benchmark[backend] = {
                    "timing": timing,
                    "validation": point_metrics(
                        parts["validation"][labels].to_numpy(),
                        predict_point(fitted, parts["validation"][columns]),
                        threshold,
                    ),
                }
            benchmark["scope"] = (
                "One paired measurement, same data and configuration, includes fit only; does not measure electrical energy or rental cost"
            )
            write_json(RELEASE / "gpu_benchmark.json", benchmark)
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu = "not queried"
    write_json(
        RELEASE / "hardware.json",
        {
            "gpu": gpu,
            "study_scope": "Validation diagnostics after point selection, no modification to the selected model or test predictions",
        },
    )


if __name__ == "__main__":
    run()
