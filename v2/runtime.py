"""Suy luận v2, tách tổng trực tiếp khỏi tổng sáu bước và khỏi cảnh báo."""
import numpy as np
import appliances_core as v1
from v2 import core


def forecast_batch(bundle, df, mask):
    config, legacy = bundle['config'], bundle['legacy']
    x = core.build_features(df)
    xa = x.loc[mask, legacy['features']].to_numpy(float)
    steps = v1.predict_point(legacy['point_name'], legacy['point_model'], xa)
    if config['method'] == 'lstm':
        from v2.lstm import predict_lstm
        steps = predict_lstm(bundle['candidate_model'], df, mask)
        hour = steps.sum(1)
    elif config['target'] == 'direct':
        hour = core.predict_hour(config, bundle['candidate_model'], x.loc[mask, core.FEATURE_SETS[config['features']]].to_numpy(float))
    elif config['name'] != 'v1':
        steps = v1.predict_point(config['method'], bundle['candidate_model'], x.loc[mask, core.FEATURE_SETS[config['features']]].to_numpy(float))
        hour = steps.sum(1)
    else:
        hour = steps.sum(1)
    q, crossing = v1.predict_quantiles(legacy['quantile_models'], xa)
    return steps, hour, q, crossing


def score_for_policy(name, hour, quantiles, margin):
    if name == 'point':
        return np.asarray(hour)
    if name == 'raw_p90':
        return np.asarray(quantiles)[:, 2]
    if name in ('v1', 'upper'):
        return np.asarray(quantiles)[:, 2] + margin
    raise ValueError(name)


def forecast_one(bundle, history):
    # Dùng hợp đồng đầu vào v1 trước khi tính đặc trưng hoặc dự báo mở rộng.
    rec = v1.forecast_one(bundle['legacy'], history)
    tail = history.tail(v1.DAY+1).reset_index(drop=True)
    mask = np.zeros(len(tail), bool)
    mask[-1] = True
    steps, hour, quantiles, _ = forecast_batch(bundle, tail, mask)
    rec['point_steps_Wh'] = steps[0].tolist()
    rec['point_sum_Wh'] = float(steps[0].sum())
    rec['hour_prediction_Wh'] = float(hour[0])
    rec['hour_target'] = bundle['config']['target']
    rec['model_version'] = bundle['version']
    rec['alert_score'] = float(score_for_policy(bundle['policy']['name'], hour, quantiles, bundle['legacy']['margin'])[0])
    rec['alert_score_threshold'] = bundle['policy']['threshold']
    rec['alert'] = rec['alert_score'] > rec['alert_score_threshold']
    return rec
