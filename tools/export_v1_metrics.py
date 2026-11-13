"""Xuất số liệu đối chiếu báo cáo v1, không fit hay chọn lại mô hình.

Chạy: python tools/export_v1_metrics.py
Đầu ra: out/v1_report_metrics.json (không sửa các kết quả gốc).
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

from tools.check_project import check_frozen_files


def require_close(actual, expected, label):
    if abs(float(actual) - float(expected)) > 1e-9:
        raise ValueError(f"Số liệu không khớp: {label}: {actual} != {expected}")


def export_metrics():
    # Kiểm đúng phiên trước khi nạp model; chỉ dùng joblib đi kèm repo.
    frozen = check_frozen_files(ROOT)
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_pinball_loss
    import appliances_core as c

    results = json.loads((ROOT / "out/results.json").read_text(encoding="utf-8"))
    bundle = joblib.load(ROOT / "out/bundle.joblib")
    df, audit = c.load_raw(ROOT / "data/energydata_complete.csv")
    features, labels = c.build_features(df), c.build_labels(df)
    columns = bundle["features"]
    masks = {name: c.origin_mask(df.t, *bounds, features[columns].to_numpy(), labels)
             for name, bounds in c.SPLITS.items()}
    for name, mask in masks.items():
        require_close(mask.sum(), results["split_rows"][name], f"số origin {name}")
    train_total = labels[masks["train"]].sum(axis=1)
    threshold = float(np.quantile(train_total, 0.95))
    require_close(threshold, results["threshold_Wh_per_hour"], "ngưỡng TRAIN")
    require_close(threshold, bundle["threshold"], "ngưỡng trong model")
    require_close(bundle["margin"], results["calibration_margin_Wh"], "margin")

    # CAL: bổ sung các ô còn thiếu trong results.json bằng model đã lưu.
    cal_x = features.loc[masks["calibration"], columns].to_numpy(float)
    q, crossing = c.predict_quantiles(bundle["quantile_models"], cal_x)
    cal_y = labels[masks["calibration"]].sum(axis=1)
    margin = float(bundle["margin"])
    lower, upper = np.maximum(q[:, 0] - margin, 0), q[:, 2] + margin
    calibration = {
        "n": len(cal_y),
        "raw_coverage": float(np.mean((cal_y >= q[:, 0]) & (cal_y <= q[:, 2]))),
        "calibrated_coverage": float(np.mean((cal_y >= lower) & (cal_y <= upper))),
        "calibrated_mean_width": float(np.mean(upper - lower)),
        "fractions": [float(np.mean(cal_y <= q[:, i])) for i in range(3)],
        "pinball": [float(mean_pinball_loss(cal_y, q[:, i], alpha=a))
                    for i, a in enumerate(c.QUANTILES)],
        "crossing_rows": int(crossing.sum()),
    }
    for key in ("raw_coverage", "calibrated_coverage"):
        require_close(calibration[key], results["quantile_calibration_set"][key], key)
    require_close(c.calibration_margin(cal_y, q), margin, "tính lại margin trên CAL")

    # TEST: đối chiếu CSV dự báo đã chốt, không chọn cấu hình hay ngưỡng mới.
    saved = pd.read_csv(ROOT / "out/test_predictions.csv")
    expected_times = pd.DatetimeIndex(df.loc[masks["test"], "t"])
    if not pd.DatetimeIndex(pd.to_datetime(saved.issue_time)).equals(expected_times):
        raise ValueError("Timestamp của CSV dự báo khác TEST")
    actual = saved[[f"actual_{k}m" for k in range(10, 61, 10)]].to_numpy(float)
    predicted = saved[[f"pred_{k}m" for k in range(10, 61, 10)]].to_numpy(float)
    if not np.array_equal(actual, labels[masks["test"]]):
        raise ValueError("Nhãn trong CSV dự báo khác dữ liệu UCI")
    point = c.reg_metrics(actual, predicted)
    for key, value in point.items():
        require_close(value, results["point"]["test"][bundle["point_name"]][key], key)
    total = actual.sum(axis=1)
    for column, expected in (("actual_hour_total", total),
                             ("point_hour_total", predicted.sum(axis=1)),
                             ("lower_cal", np.maximum(saved.p10.to_numpy() - margin, 0)),
                             ("upper_cal", saved.p90.to_numpy() + margin)):
        if not np.allclose(saved[column], expected, rtol=0, atol=1e-9):
            raise ValueError(f"CSV không khớp: {column}")
    alerts = {}
    for name, score in (("ai_raw_p90", saved.p90.to_numpy()),
                        ("ai_calibrated_upper", saved.upper_cal.to_numpy())):
        alerts[name] = c.alert_metrics(total, score, threshold)
        for key, value in alerts[name].items():
            if value is not None:
                require_close(value, results["alert"][name][key], f"{name}.{key}")
    high = total > threshold
    high_metrics = {
        "n": int(high.sum()), "episodes": len(c.runs(high)),
        "mae_hour": float(np.mean(np.abs(predicted.sum(axis=1)[high] - total[high]))),
        "underprediction_rate": float(np.mean(predicted.sum(axis=1)[high] < total[high])),
        "coverage": float(np.mean((total[high] >= saved.lower_cal.to_numpy()[high]) &
                                  (total[high] <= saved.upper_cal.to_numpy()[high]))),
    }
    baseline_mae = results["point"]["test"]["persistence"]["mae_hour"]
    if check_frozen_files(ROOT) != frozen:
        raise ValueError("Tệp v1 bị thay đổi trong quá trình đối chiếu")
    return {
        "passed": True, "version": "v1", "experiment_commit": "8d6c1ba0b5368426236a72f7d7240da90052410d",
        "scope": "Frozen-model verification; no fitting, tuning or new model selection. TEST is retrospective.",
        "data_sha256": audit["sha256"], "model_sha256": frozen["model_sha256"],
        "predictions_sha256": hashlib.sha256((ROOT / "out/test_predictions.csv").read_bytes()).hexdigest(),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "feature_set": results["selected_feature_set"], "n_features": len(columns),
        "point_method": bundle["point_name"], "threshold_Wh": threshold, "margin_Wh": margin,
        "validation_is_fold_B": c.FOLDS["walk_B_april"][1] == c.SPLITS["validation"],
        "calibration": calibration, "point_test": point, "high_group_test": high_metrics,
        "mae_hour_reduction_percent": 100 * (1 - point["mae_hour"] / baseline_mae), "alerts_test": alerts,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out/v1_report_metrics.json")
    args = parser.parse_args()
    output = args.out.resolve()
    # Không cho ghi nhầm vào dữ liệu, mã nguồn, model hoặc kết quả gốc.
    if output.suffix != ".json" or (output.exists() and output.name != "v1_report_metrics.json"):
        parser.error("Chọn tệp JSON mới hoặc v1_report_metrics.json; không ghi đè tệp gốc")
    metrics = export_metrics()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK: đối chiếu v1, không huấn luyện lại. Kết quả: {output}")


if __name__ == "__main__":
    main()
