"""Đặc trưng tăng tải và hồi quy có trọng số, chỉ học từ TRAIN."""
import numpy as np
import appliances_core as v1
from sklearn.ensemble import HistGradientBoostingRegressor

RISE_FEATURES = ["app_diff_1", "app_diff_3", "app_std_1h", "app_ratio_max1h_mean24h"]
FEATURE_SETS = {"A": list(v1.AR_FEATURES), "F": list(v1.AR_FEATURES) + RISE_FEATURES}


def build_features(df):
    x = v1.build_features(df)
    x["app_diff_1"] = x.app_lag_0 - x.app_lag_1
    x["app_diff_3"] = x.app_lag_0 - x.app_lag_3
    x["app_std_1h"] = v1._win(df.Appliances.to_numpy(float), 6, "std")
    # +1 Wh tránh chia cho 0, không phải một tham số đã được tối ưu.
    x["app_ratio_max1h_mean24h"] = x.app_max_1h / (x.app_mean_24h + 1.0)
    return x


def training_weights(y, weight_high=1.0, threshold=None):
    y = np.asarray(y, dtype=float)
    total = y.sum(axis=1) if y.ndim == 2 else y
    if not np.isfinite(total).all() or len(total) == 0:
        raise ValueError("Nhãn TRAIN phải hữu hạn và không rỗng")
    if not np.isfinite(weight_high) or weight_high < 1:
        raise ValueError("weight_high phải hữu hạn và >= 1")
    threshold = float(np.quantile(total, .95)) if threshold is None else float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("Ngưỡng TRAIN không hữu hạn")
    return np.where(total > threshold, float(weight_high), 1.0), threshold


def fit_point(name, x, y, weight_high=1.0, params=None):
    """Mặc định gọi đúng fit_point v1; chỉ HGB hỗ trợ thử trọng số ở đây."""
    weights, threshold = training_weights(y, weight_high)
    if params is None and weight_high == 1:
        return v1.fit_point(name, x, y), threshold
    if name not in ("hgb_l1", "hgb_l2"):
        raise ValueError("Thử trọng số chỉ áp dụng cho HGB")
    config = {**v1.POINT_PARAMS[name], **(params or {})}
    models = [HistGradientBoostingRegressor(**config).fit(x, y[:, i], sample_weight=weights)
              for i in range(v1.HORIZON)]
    return models, threshold


def fit_direct(name, x, y, weight_high=1.0, params=None):
    """Một HGB học trực tiếp tổng 6 nhãn, không tạo số đo mới."""
    if name not in ("hgb_l1", "hgb_l2"):
        raise ValueError(name)
    weights, threshold = training_weights(y, weight_high)
    config = {**v1.POINT_PARAMS[name], **(params or {})}
    total = np.asarray(y).sum(axis=1)
    model = HistGradientBoostingRegressor(**config).fit(x, total, sample_weight=weights)
    return model, threshold


def predict_hour(config, model, x):
    if config["target"] == "direct":
        return np.maximum(model.predict(x), 0)
    return v1.predict_point(config["method"], model, x).sum(axis=1)
