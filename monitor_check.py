"""monitor_check.py - Giám sát sau triển khai (bước 8 của lifecycle) trên dữ liệu test đã lưu.

Mô phỏng cách theo dõi hằng tuần khi nhãn (actual) đến trễ 60 phút:
  - MAE tổng giờ và độ phủ dải so với mức tham chiếu của tập CALIBRATION;
  - PSI của phân bố Appliances so với TRAIN (10 nhóm theo decile của TRAIN).
Quy tắc rà soát là HEURISTIC của nhóm (không phải chuẩn công nghiệp): MAE > 1,2 x tham chiếu HOẶC độ phủ < tham chiếu - 0,10
ở hai cửa sổ 7 ngày liên tiếp -> yêu cầu con người rà soát. Không tự retrain.
Chạy:  python monitor_check.py        (cần out/results.json và out/test_predictions.csv)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import appliances_core as c

ROOT = Path(__file__).parent
OUT = ROOT / "out"
R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
df, _ = c.load_raw(ROOT / "data" / "energydata_complete.csv")
pred = pd.read_csv(OUT / "test_predictions.csv", parse_dates=["issue_time"])

point = R["selected_point_method"]
ref_mae = R["point"]["calibration"][point]["mae_hour"]
ref_cov = R["quantile_calibration_set"]["calibrated_coverage"]
edges = np.quantile(df.Appliances[df.t < c.SPLITS["train"][1]], np.linspace(0, 1, 11))
edges[0], edges[-1] = -np.inf, np.inf
edges = np.unique(edges)  # các phân vị trùng nhau không tạo nhóm rỗng


def psi(expected, actual):
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


train_vals = df.Appliances[df.t < c.SPLITS["train"][1]].to_numpy()
rows, flagged = [], []
start = pred.issue_time.min().normalize()
for w in range(4):
    lo, hi = start + pd.Timedelta(days=7 * w), start + pd.Timedelta(days=7 * (w + 1))
    m = (pred.issue_time >= lo) & (pred.issue_time < hi)
    if m.sum() < c.DAY:
        continue
    p = pred[m]
    mae = float(np.mean(np.abs(p.actual_hour_total - p.point_hour_total)))
    cov = float(np.mean((p.actual_hour_total >= p.lower_cal) & (p.actual_hour_total <= p.upper_cal)))
    app = df.Appliances[(df.t >= lo) & (df.t < hi)].to_numpy()
    bad = mae > 1.2 * ref_mae or cov < ref_cov - 0.10
    flagged.append(bad)
    rows.append({"window": f"{lo.date()} → {(hi - pd.Timedelta(days=1)).date()}", "origins": int(m.sum()), "mae_hour": mae,
                 "mae_vs_ref": mae / ref_mae, "coverage": cov, "psi_appliances": psi(train_vals, app), "breach": bool(bad)})
review = any(a and b for a, b in zip(flagged, flagged[1:]))
# PSI tự nhiên giữa các tuần đầy đủ trong TRAIN (so với toàn TRAIN) để biết mức dao động bình thường
nat = []
t_tr0 = pd.Timestamp("2016-01-12")
for w in range(11):
    lo, hi = t_tr0 + pd.Timedelta(days=7 * w), t_tr0 + pd.Timedelta(days=7 * (w + 1))
    nat.append(psi(train_vals, df.Appliances[(df.t >= lo) & (df.t < hi)].to_numpy()))
out = {"reference": {"mae_hour_calibration": ref_mae, "coverage_calibration": ref_cov, "mae_factor": 1.2, "coverage_drop": 0.10},
       "windows": rows, "human_review_triggered": review,
       "psi_natural_weekly_in_train": {"n_weeks": len(nat), "min": float(np.min(nat)), "median": float(np.median(nat)), "max": float(np.max(nat))},
       "note": "heuristic của prototype; PSI chỉ mô tả dịch chuyển phân bố đầu vào, không chứng minh drift hay lỗi cảm biến"}
(OUT / "monitoring.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
