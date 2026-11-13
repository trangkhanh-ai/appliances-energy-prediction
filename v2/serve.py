"""Suy luận một lần: kiểm đầu vào, tạo đặc trưng một lần, chạy đúng model cần dùng.

Tối ưu đường chạy, giữ nguyên model/ngưỡng/lựa chọn đã khóa. runtime.py là bản
tham chiếu đối chiếu; không cần chạy lại huấn luyện hoặc tìm cấu hình.
"""
import numpy as np
import pandas as pd
import appliances_core as v1
from v2 import core, runtime


def forecast_one(bundle, history):
    if len(history) < 145:
        raise ValueError('Cần ít nhất 145 số đo liên tiếp')
    history = history.tail(145).reset_index(drop=True)
    times = pd.DatetimeIndex(history.t)
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing or not pd.Series(times).diff().dropna().eq(v1.STEP).all():
        raise ValueError('Lịch sử phải tăng liên tục mỗi 10 phút, không thiếu/trùng')
    if not np.isfinite(history.Appliances.to_numpy(float)).all() or (history.Appliances < 0).any():
        raise ValueError('Điện năng bị thiếu, âm hoặc vô hạn')
    legacy, config = bundle['legacy'], bundle['config']
    x = core.build_features(history).iloc[[-1]]
    xa = x[legacy['features']].to_numpy(float)
    xp = x[core.FEATURE_SETS[config['features']]].to_numpy(float)
    if not np.isfinite(xa).all() or not np.isfinite(xp).all():
        raise ValueError('Đặc trưng không hợp lệ')
    if config['method'] == 'lstm':
        from v2.lstm import predict_lstm
        mask = np.zeros(len(history), bool); mask[-1] = True
        steps = predict_lstm(bundle['candidate_model'], history, mask)
        hour = steps.sum(1)
    elif config['target'] == 'direct':
        steps = v1.predict_point(legacy['point_name'], legacy['point_model'], xa)
        hour = core.predict_hour(config, bundle['candidate_model'], xp)
    else:
        model = legacy['point_model'] if config['name'] == 'v1' else bundle['candidate_model']
        steps = v1.predict_point(config['method'], model, xp)
        hour = steps.sum(1)
    q, crossing = v1.predict_quantiles(legacy['quantile_models'], xa)
    margin = legacy['margin']
    score = float(runtime.score_for_policy(bundle['policy']['name'], hour, q, margin)[0])
    return {'issue_time': str(history.t.iloc[-1]), 'point_steps_Wh': steps[0].tolist(),
            'point_sum_Wh': float(steps[0].sum()), 'hour_prediction_Wh': float(hour[0]),
            'hour_target': config['target'], 'model_version': bundle['version'],
            'p10': float(q[0, 0]), 'p50': float(q[0, 1]), 'p90': float(q[0, 2]),
            'lower_cal': max(float(q[0, 0])-margin, 0.), 'upper_cal': float(q[0, 2])+margin,
            'threshold': legacy['threshold'], 'alert_score': score,
            'alert_score_threshold': bundle['policy']['threshold'],
            'alert': bool(score > bundle['policy']['threshold']), 'crossing_reordered': bool(crossing[0])}
