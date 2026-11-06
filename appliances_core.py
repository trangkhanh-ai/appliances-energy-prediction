"""appliances_core.py - Thư viện lõi cho đồ án: dự báo điện năng thiết bị trong giờ tới
(UCI Appliances Energy Prediction) + dải bất định + cảnh báo tiêu thụ cao.

Quy ước: một hàng = 10 phút, Appliances = Wh trong 10 phút. Tại thời điểm phát dự báo t
chỉ dùng số đo <= t. Nhãn = Appliances tại t+10, ..., t+60 phút (6 bước);
"tổng giờ tới" = tổng 6 nhãn (Wh) = công suất trung bình của giờ tới tính bằng W.
"""
import json
import hashlib
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STEP = pd.Timedelta(minutes=10)
HORIZON = 6                      # 6 bước x 10 phút = 60 phút
DAY = 144                        # số bước trong 1 ngày
QUANTILES = (0.1, 0.5, 0.9)

# Mốc chia theo thời gian (nửa mở). Origin hợp lệ: t >= start và t + 60 phút < end (purge nhãn vượt biên).
SPLITS = {"train": ("2016-01-11 17:00", "2016-04-01"), "validation": ("2016-04-01", "2016-04-15"),
          "calibration": ("2016-04-15", "2016-05-01"), "test": ("2016-05-01", "2016-05-28")}
FOLDS = {"walk_A_march": (("2016-01-11 17:00", "2016-03-01"), ("2016-03-01", "2016-04-01")),
         "walk_B_april": (("2016-01-11 17:00", "2016-04-01"), ("2016-04-01", "2016-04-15"))}

WEATHER = ["T_out", "RH_out", "Windspeed", "Press_mm_hg", "Visibility", "Tdewpoint"]
HOUSE = [f"T{i}" for i in range(1, 10)] + [f"RH_{i}" for i in range(1, 10)]
AR_LAGS = (0, 1, 2, 3, 6, 144)
AR_FEATURES = ([f"app_lag_{k}" for k in AR_LAGS] + ["app_yday_window_sum", "app_sum_1h", "app_max_1h", "app_mean_3h",
               "app_std_3h", "app_mean_24h", "app_max_24h", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"])
FEATURE_SETS = {"A_ar_calendar": AR_FEATURES,
                "B_plus_lights": AR_FEATURES + ["lights_lag_0", "lights_sum_1h"],
                "C_plus_weather": AR_FEATURES + ["lights_lag_0", "lights_sum_1h"] + WEATHER,
                "D_plus_house": AR_FEATURES + ["lights_lag_0", "lights_sum_1h"] + WEATHER + HOUSE,
                "E_D_plus_rv1_control": AR_FEATURES + ["lights_lag_0", "lights_sum_1h"] + WEATHER + HOUSE + ["rv1"]}

POINT_PARAMS = {
    "hgb_l1": dict(loss="absolute_error", max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                   min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=42),
    "hgb_l2": dict(loss="squared_error", max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                   min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=42),
    "xgb_l1": dict(objective="reg:absoluteerror", tree_method="hist", n_estimators=300, max_depth=4,
                   learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                   random_state=42, n_jobs=2),
    "rf": dict(n_estimators=300, min_samples_leaf=10, max_features=0.5, random_state=42, n_jobs=2),
}


# ----------------------------------------------------------------------------- 1. Dữ liệu
def load_raw(path):
    """Đọc CSV, kiểm lưới 10 phút/thiếu/trùng/âm. Không impute, không cắt outlier."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    expected = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
    if digest != expected:
        raise ValueError("CSV không trùng bản UCI đã xác minh; cần kiểm tra lại nguồn.")
    df = pd.read_csv(path)
    df["t"] = pd.to_datetime(df["date"])
    audit = {"rows": int(len(df)), "start": str(df.t.min()), "end": str(df.t.max()),
             "was_sorted": bool(df.t.is_monotonic_increasing)}
    df = df.sort_values("t", kind="stable").reset_index(drop=True)
    if not df.t.is_unique or not df.t.diff().dropna().eq(STEP).all():
        raise ValueError("Lưới 10 phút bị thiếu/trùng")
    numbers = df.drop(columns=["date", "t"]).to_numpy(float)
    if not np.isfinite(numbers).all() or (df.Appliances < 0).any():
        raise ValueError("Dữ liệu thiếu, vô hạn hoặc điện năng âm")
    audit["sha256"] = digest
    audit.update(missing_cells=0, grid_gaps=0, duplicates=0,
                 appliances_multiple_of_10=bool((df.Appliances % 10 == 0).all()),
                 rv1_equals_rv2=bool((df.rv1 == df.rv2).all()))
    return df, audit


def _win(x, w, op):
    out = np.full(len(x), np.nan)
    if len(x) < w:
        return out
    out[w - 1:] = getattr(sliding_window_view(x, w), op)(axis=1)    # cửa sổ kết thúc tại t, ddof=0
    return out


def build_features(df):
    """Mọi feature tại t chỉ dùng hàng <= t và lịch của khoảng dự báo (biết trước)."""
    a = df.Appliances.to_numpy(float)
    s = pd.Series(a)
    cols = {f"app_lag_{k}": s.shift(k).to_numpy() for k in AR_LAGS}
    # cùng cửa sổ t+10..t+60 của ngày hôm trước: giá trị tại i+k-144 => shift(144-k), k=1..6
    cols["app_yday_window_sum"] = sum(s.shift(DAY - k).to_numpy() for k in range(1, HORIZON + 1))
    cols.update(app_sum_1h=_win(a, 6, "sum"), app_max_1h=_win(a, 6, "max"), app_mean_3h=_win(a, 18, "mean"),
                app_std_3h=_win(a, 18, "std"), app_mean_24h=_win(a, DAY, "mean"), app_max_24h=_win(a, DAY, "max"))
    start = pd.DatetimeIndex(df.t) + STEP
    hour = np.asarray(start.hour + start.minute / 60)
    dow = np.asarray(start.dayofweek)
    cols.update(hour_sin=np.sin(2 * np.pi * hour / 24), hour_cos=np.cos(2 * np.pi * hour / 24),
                dow_sin=np.sin(2 * np.pi * dow / 7), dow_cos=np.cos(2 * np.pi * dow / 7),
                is_weekend=(dow >= 5).astype(float))
    cols["lights_lag_0"] = df.lights.to_numpy(float)
    cols["lights_sum_1h"] = _win(df.lights.to_numpy(float), 6, "sum")
    for c in WEATHER + HOUSE + ["rv1"]:
        cols[c] = df[c].to_numpy(float)
    return pd.DataFrame(cols)


def build_labels(df):
    """Nhãn OFFLINE: Appliances tại t+10..t+60. Inference không gọi hàm này."""
    s = pd.Series(df.Appliances.to_numpy(float))
    return np.column_stack([s.shift(-k).to_numpy() for k in range(1, HORIZON + 1)])


def origin_mask(times, start, end, X=None, Y=None):
    t = pd.DatetimeIndex(times)
    m = (t >= pd.Timestamp(start)) & (t + HORIZON * STEP < pd.Timestamp(end))
    if X is not None:
        m &= np.isfinite(X).all(axis=1)
    if Y is not None:
        m &= np.isfinite(Y).all(axis=1)
    return np.asarray(m)


# ----------------------------------------------------------------------------- 2. Mô hình
def fit_point(name, X, Y):
    """Hồi quy đa đầu ra 6 bước. Siêu tham số cố định trước (không tuning theo test)."""
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, Y)
    if name == "rf":
        return RandomForestRegressor(**POINT_PARAMS["rf"]).fit(X, Y)
    if name in ("hgb_l1", "hgb_l2"):
        return [HGB(**POINT_PARAMS[name]).fit(X, Y[:, i]) for i in range(HORIZON)]
    if name == "xgb_l1":
        import xgboost as xgb
        return [xgb.XGBRegressor(**POINT_PARAMS[name]).fit(X, Y[:, i]) for i in range(HORIZON)]
    raise ValueError(name)


def predict_point(name, model, X):
    if name in ("ridge", "rf"):
        p = model.predict(X)
    else:
        p = np.column_stack([m.predict(X) for m in model])
    return np.maximum(p, 0)


def fit_quantiles(X, y_total):
    return [HGB(loss="quantile", quantile=q, max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=42).fit(X, y_total)
            for q in QUANTILES]


def predict_quantiles(models, X):
    """Cắt số âm rồi sắp thứ tự P10<=P50<=P90 (hậu xử lý công bố trước đánh giá)."""
    raw = np.maximum(np.column_stack([m.predict(X) for m in models]), 0)
    return np.sort(raw, axis=1), bool_crossing(raw)


def bool_crossing(raw):
    return np.any(np.diff(raw, axis=1) < 0, axis=1)


def calibration_margin(y, q, nominal=0.8):
    """Margin kiểu CQR trên tập calibration để dải P10-P90 phủ ~nominal. Không có bảo đảm cho chuỗi thời gian."""
    score = np.maximum.reduce([q[:, 0] - y, y - q[:, 2], np.zeros(len(y))])
    level = min(1.0, np.ceil((len(y) + 1) * nominal) / len(y))
    return float(np.quantile(score, level, method="higher"))


# ----------------------------------------------------------------------------- 3. Baseline không học
def baseline_predictions(df, idx, train_idx):
    """Ba baseline: persistence, ngày hôm trước, trung vị theo giờ trong ngày (chỉ học từ TRAIN)."""
    a = df.Appliances.to_numpy(float)
    out = {"persistence": np.repeat(a[idx][:, None], HORIZON, axis=1),
           "seasonal_day": np.column_stack([a[idx + k - DAY] for k in range(1, HORIZON + 1)])}
    t = pd.DatetimeIndex(df.t)
    slot = (t.hour * 6 + t.minute // 10).to_numpy()                     # 0..143
    med = pd.Series(a[train_idx]).groupby(slot[train_idx]).median().reindex(range(DAY)).to_numpy()
    out["diurnal_median"] = np.column_stack([med[(slot[idx] + k) % DAY] for k in range(1, HORIZON + 1)])
    return out


# ----------------------------------------------------------------------------- 4. Chỉ số
def reg_metrics(Y, P):
    yh, ph = Y.sum(1), P.sum(1)
    return {"mae_step": float(mean_absolute_error(Y, P)), "rmse_step": float(root_mean_squared_error(Y, P)),
            "mae_hour": float(mean_absolute_error(yh, ph)), "rmse_hour": float(root_mean_squared_error(yh, ph)),
            "bias_hour": float(np.mean(ph - yh)), "r2_hour": float(1 - np.sum((yh - ph) ** 2) / np.sum((yh - yh.mean()) ** 2))}


def alert_metrics(y_total, score, T):
    """Đúng: tổng giờ tới > T. Cảnh báo: score > T."""
    truth, alert = y_total > T, score > T
    tp, fp = int((truth & alert).sum()), int((~truth & alert).sum())
    fn, tn = int((truth & ~alert).sum()), int((~truth & ~alert).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    ok = 0 < truth.sum() < len(truth)
    return {"n": int(len(truth)), "positives": int(truth.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "alerts": tp + fp, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "auc_roc": float(roc_auc_score(truth, score)) if ok else None,
            "auc_pr": float(average_precision_score(truth, score)) if ok else None}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def runs(flags):
    d = np.diff(np.r_[0, np.asarray(flags, bool).astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def episode_metrics(truth, alert):
    """Gom origin liên tiếp (cách 10 phút) thành cụm. lead > 0: cảnh báo sớm hơn origin đầu của cụm thật (phút)."""
    tr, al = runs(truth), runs(alert)
    ov = lambda a, b: a[0] <= b[1] and b[0] <= a[1]
    det = [r for r in tr if any(ov(r, a) for a in al)]
    lead = [10 * (r[0] - min(a[0] for a in al if ov(r, a))) for r in det]
    return {"true_episodes": len(tr), "alert_episodes": len(al), "detected": len(det),
            "valid_alert_episodes": int(sum(any(ov(a, r) for r in tr) for a in al)), "lead_minutes": lead}


def cost_curve(y, score, T, ratios):
    m = alert_metrics(y, score, T)
    return {str(r): m["fp"] + r * m["fn"] for r in ratios}


# ----------------------------------------------------------------------------- 5. Kiểm thử độ bền
def perturb(df, kind, level, seed=0):
    d = df.copy()
    rng = np.random.default_rng(seed)
    if kind == "noise":                       # nhiễu nhân tính của công tơ
        d["Appliances"] = np.maximum(d.Appliances * (1 + rng.normal(0, level, len(d))), 0)
    elif kind == "dropout":                   # mất mốc đo, giữ giá trị cũ (forward-fill)
        drop = rng.random(len(d)) < level
        drop[0] = False
        for c in ("Appliances", "lights"):
            v = d[c].to_numpy(float)
            v[drop] = np.nan
            d[c] = pd.Series(v).ffill().to_numpy()
    elif kind == "meter_scale":               # công tơ lệch hiệu chuẩn: lịch sử bị nhân hệ số, nhãn thật giữ nguyên
        d["Appliances"] = d.Appliances * level
    elif kind == "temp_shift":                # lệch nhiệt độ ngoài trời/điểm sương (độ C)
        d["T_out"] = d.T_out + level
        d["Tdewpoint"] = d.Tdewpoint + level
    elif kind == "weather_stuck":             # cảm biến thời tiết kẹt ở giá trị trung bình
        for c in WEATHER:
            d[c] = df.loc[df.t < SPLITS["train"][1], c].mean()  # chỉ tham chiếu TRAIN
    return d


# ----------------------------------------------------------------------------- 6. Suy luận một điểm (demo/latency)
def forecast_one(bundle, history):
    """history: DataFrame >= 145 hàng gần nhất, kết thúc đúng thời điểm t. Trả về dict dự báo."""
    if len(history) < DAY + 1:
        raise ValueError("Cần ít nhất 145 số đo liên tiếp")
    history = history.tail(DAY + 1).reset_index(drop=True)
    times = pd.DatetimeIndex(history.t)
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing or not pd.Series(times).diff().dropna().eq(STEP).all():
        raise ValueError("Lịch sử phải tăng liên tục mỗi 10 phút, không thiếu/trùng")
    needed = ["Appliances"] + [x for x in bundle["features"] if x in history.columns]
    if not np.isfinite(history[needed].to_numpy(float)).all() or (history.Appliances < 0).any():
        raise ValueError("Số đo sử dụng bị thiếu, vô hạn hoặc điện năng âm")
    feats = build_features(history.tail(DAY + 1).reset_index(drop=True)).iloc[[-1]][bundle["features"]].to_numpy(float)
    if not np.isfinite(feats).all():
        raise ValueError("Đặc trưng lỗi/thiếu lịch sử")
    pts = predict_point(bundle["point_name"], bundle["point_model"], feats)[0]
    q, crossing = predict_quantiles(bundle["quantile_models"], feats)
    lo, med, hi = map(float, q[0])
    m, T = bundle["margin"], bundle["threshold"]
    return {"issue_time": str(history.t.iloc[-1]), "point_steps_Wh": [float(x) for x in pts],
            "point_sum_Wh": float(pts.sum()), "p10": lo, "p50": med, "p90": hi,
            "lower_cal": max(lo - m, 0.0), "upper_cal": hi + m, "threshold": T,
            "alert": bool(hi + m > T), "crossing_reordered": bool(crossing[0])}


def measure_latency(bundle, history, n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(DAY + 1, len(history)), n, replace=False)
    times = []
    for i in idx:
        t0 = time.perf_counter()
        forecast_one(bundle, history.iloc[i - DAY: i + 1])
        times.append((time.perf_counter() - t0) * 1000)
    return {"n": n, "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
            "max_ms": float(np.max(times))}


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def save_bundle(path, bundle):
    joblib.dump(bundle, path)
    return Path(path).stat().st_size
