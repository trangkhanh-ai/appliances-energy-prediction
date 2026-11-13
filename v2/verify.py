"""Kiểm lại mô hình đã lưu, không fit và không đổi lựa chọn sau TEST."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import appliances_core as v1
from tools.check_project import check_frozen_files
from v2 import core, metrics, runtime
from v2.run import sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'out_v2'


def verify(include_lstm=False):
    check_frozen_files(ROOT)
    lock_path = OUT/'selection_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    if sources() != lock['source_sha256'] or lock['test_used_for_selection']:
        raise ValueError('Mã nguồn hoặc quy trình lựa chọn không khớp phiên đã khóa')
    stamp = json.loads((OUT/'test_evaluation_started.json').read_text(encoding='utf-8'))
    assert stamp['model_sha256'] == hashlib.sha256((OUT/'bundle.joblib').read_bytes()).hexdigest()
    assert stamp['selection_lock_sha256'] == hashlib.sha256(lock_path.read_bytes()).hexdigest()
    bundle = joblib.load(OUT/'bundle.joblib')
    results = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    saved = pd.read_csv(OUT/'test_predictions.csv')
    df, audit = v1.load_raw(ROOT/'data/energydata_complete.csv')
    x, y = core.build_features(df), v1.build_labels(df)
    mt = v1.origin_mask(df.t, *v1.SPLITS['train'], x[core.FEATURE_SETS['A']].to_numpy(), y)
    me = v1.origin_mask(df.t, *v1.SPLITS['test'], x[core.FEATURE_SETS['A']].to_numpy(), y)
    assert pd.DatetimeIndex(pd.to_datetime(saved.issue_time)).equals(pd.DatetimeIndex(df.t[me]))
    np.testing.assert_array_equal(y[me], saved[[f'actual_{i}m' for i in range(10, 61, 10)]].to_numpy())
    np.testing.assert_allclose(y[me].sum(1), saved.actual_hour_total, atol=1e-10, rtol=0)
    threshold = float(np.quantile(y[mt].sum(1), .95))
    assert threshold == results['threshold'] == bundle['legacy']['threshold']
    steps, hour, q, _ = runtime.forecast_batch(bundle, df, me)
    tolerance = 1e-3 if bundle['config']['method'] == 'lstm' else 1e-9
    np.testing.assert_allclose(hour, saved.selected_hour_prediction, atol=tolerance, rtol=0)
    np.testing.assert_allclose(steps, saved[[f'pred_{i}m' for i in range(10, 61, 10)]], atol=tolerance, rtol=0)
    np.testing.assert_allclose(q, saved[['p10', 'p50', 'p90']], atol=1e-9, rtol=0)
    score = runtime.score_for_policy(bundle['policy']['name'], hour, q, bundle['legacy']['margin'])
    expected_alerts = score > bundle['policy']['threshold']
    np.testing.assert_array_equal(expected_alerts, saved.alert.to_numpy())
    a = metrics.alerts(y[me].sum(1), score, threshold, bundle['policy']['threshold'])
    assert all(a[key] == results['alert'][key] for key in ('tp', 'fp', 'fn', 'tn'))
    point = metrics.regression(y[me].sum(1), hour, threshold)
    for key, value in point.items():
        if value is not None:
            np.testing.assert_allclose(value, results['selected'][key], atol=tolerance, rtol=0)
    # The optional finalists archive includes PyTorch version metadata.
    # Main HGB verification above must also work without research dependencies.
    artifact = joblib.load(OUT/'finalists.joblib') if include_lstm else None
    finalist_checks = {}
    for label, config in lock['finalists'].items():
        if artifact is None:
            finalist_checks[label] = 'saved outputs only; install research dependencies and use --include-lstm for independent finalist inference'
            continue
        model = artifact['models'][config['name']]
        if config['method'] == 'lstm':
            from v2.lstm import predict_lstm
            prediction = predict_lstm(model, df, me).sum(1)
        else:
            prediction = core.predict_hour(config, model, x.loc[me, core.FEATURE_SETS[config['features']]].to_numpy(float))
        np.testing.assert_allclose(prediction, saved['hour_'+label], atol=1e-3 if config['method'] == 'lstm' else 1e-9, rtol=0)
        finalist_checks[label] = 'fresh inference matches saved predictions'
    online_error = 0.
    origins = np.flatnonzero(me)
    for j in np.random.default_rng(42).choice(len(origins), 200, replace=False):
        i = origins[j]
        one = runtime.forecast_one(bundle, df.iloc[i-v1.DAY:i+1])
        online_error = max(online_error, abs(one['hour_prediction_Wh']-hour[j]))
        assert one['alert'] == bool(expected_alerts[j])
    assert online_error <= tolerance
    # Nhãn đã đến mới được dùng để giám sát, theo cùng các cửa sổ 7 ngày.
    cal_ref = results['split_metrics']['calibration']['mae']
    coverage_ref = results['calibration_interval']['coverage']
    monitoring, consecutive = [], 0
    train_values = df.Appliances[df.t < v1.SPLITS['train'][1]].to_numpy(float)
    edges = np.quantile(train_values, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    expected_frequency = np.clip(np.histogram(train_values, edges)[0]/len(train_values), 1e-4, None)
    times = pd.DatetimeIndex(pd.to_datetime(saved.issue_time))
    for start in pd.date_range(times.min().normalize(), times.max().normalize(), freq='7D'):
        end = start + pd.Timedelta(days=7)
        mask = np.asarray((times >= start) & (times < end))
        actual = saved.actual_hour_total.to_numpy()[mask]
        predicted = saved.selected_hour_prediction.to_numpy()[mask]
        coverage = float(np.mean((actual >= saved.lower_cal.to_numpy()[mask]) & (actual <= saved.upper_cal.to_numpy()[mask])))
        mae = float(np.mean(np.abs(actual-predicted)))
        breach = mae > 1.2*cal_ref or coverage < coverage_ref-.10
        current_values = df.Appliances[(df.t >= start) & (df.t < end)].to_numpy(float)
        frequency = np.clip(np.histogram(current_values, edges)[0]/len(current_values), 1e-4, None)
        psi = float(np.sum((frequency-expected_frequency)*np.log(frequency/expected_frequency)))
        consecutive = consecutive+1 if breach else 0
        monitoring.append({'start': str(start), 'end_exclusive': str(end), 'n': int(mask.sum()),
                           'mae': mae, 'mae_vs_ref': mae/cal_ref, 'coverage': coverage, 'psi_appliances': psi,
                           'breach': breach, 'review_requested': consecutive >= 2})
    (OUT/'monitoring.json').write_text(json.dumps({'calibration_mae': cal_ref, 'calibration_coverage': coverage_ref, 'weeks': monitoring}, indent=2), encoding='utf-8')
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT/'v2').glob('*.py'))}
    hashes.update({str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [OUT/'bundle.joblib', OUT/'finalists.joblib', OUT/'results.json', OUT/'selection_lock.json', OUT/'test_predictions.csv']})
    summary = {'passed': True, 'version': bundle['version'], 'data_sha256': audit['sha256'],
               'labels_checked': int(y[me].size), 'online_origins_checked': 200,
               'online_max_abs_diff': online_error, 'tolerance_Wh': tolerance,
               'counts': {key: a[key] for key in ('tp', 'fp', 'fn', 'tn')},
               'finalists': finalist_checks, 'sha256': hashes, 'test_status': 'retrospective; no retuning'}
    (OUT/'verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-lstm', action='store_true')
    print(json.dumps(verify(parser.parse_args().include_lstm), ensure_ascii=False, indent=2))
