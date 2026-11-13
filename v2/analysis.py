"""Phân tích các ứng viên đã khóa; không chọn lại từ TEST hoặc sửa kết quả gốc."""
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import appliances_core as v1
from v2 import core, metrics, runtime, serve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'out_v2'


def main():
    bundle = joblib.load(OUT/'bundle.joblib')
    artifacts = joblib.load(OUT/'finalists.joblib')
    result = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    saved = pd.read_csv(OUT/'test_predictions.csv')
    df, _ = v1.load_raw(ROOT/'data/energydata_complete.csv')
    x, labels = core.build_features(df), v1.build_labels(df)
    mc = v1.origin_mask(df.t, *v1.SPLITS['calibration'], x[core.FEATURE_SETS['A']].to_numpy(), labels)
    me = v1.origin_mask(df.t, *v1.SPLITS['test'], x[core.FEATURE_SETS['A']].to_numpy(), labels)
    cal_y = labels[mc].sum(1)
    threshold = result['threshold']
    supplement = {'scope': 'Diagnostics of locked finalists; no model or policy reselection', 'finalists': {}}
    for name, config in artifacts['configs'].items():
        model = artifacts['models'][config['name']]
        if config['method'] == 'lstm':
            from v2.lstm import predict_lstm
            cal_prediction = predict_lstm(model, df, mc).sum(1)
        else:
            cal_prediction = core.predict_hour(config, model, x.loc[mc, core.FEATURE_SETS[config['features']]].to_numpy(float))
        prediction = saved['hour_'+name].to_numpy()
        policy = {}
        for mode in ('fixed_energy', 'fixed_fpr', 'fixed_recall'):
            cut = threshold if mode == 'fixed_energy' else (
                metrics.threshold_for_fpr(cal_y, cal_prediction, threshold, .10) if mode == 'fixed_fpr' else
                metrics.threshold_for_recall(cal_y, cal_prediction, threshold, .70))
            policy[mode] = metrics.alerts(saved.actual_hour_total.to_numpy(), prediction, threshold, cut)
        supplement['finalists'][name] = {'point': result['finalists'][name]['metrics'], 'alert_policies': policy}
    (OUT/'supplement.json').write_text(json.dumps(supplement, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    # Benchmark trên cùng 200 origin: đầu ra bắt buộc khớp đường tham chiếu.
    timings, reference_timings, differences = [], [], []
    import torch
    torch.set_num_threads(2)
    for position in np.random.default_rng(42).choice(np.flatnonzero(me), 200, replace=False):
        history = df.iloc[position-144:position+1]
        started = time.perf_counter()
        expected = runtime.forecast_one(bundle, history)
        reference_timings.append((time.perf_counter()-started)*1000)
        started = time.perf_counter()
        actual = serve.forecast_one(bundle, history)
        timings.append((time.perf_counter()-started)*1000)
        differences.append(max(abs(expected[key]-actual[key]) for key in ('hour_prediction_Wh', 'point_sum_Wh', 'p10', 'p50', 'p90', 'lower_cal', 'upper_cal')))
        assert expected['alert'] == actual['alert']
    assert max(differences) <= 1e-3 if bundle['config']['method'] == 'lstm' else max(differences) <= 1e-9
    performance = {'n': len(timings), 'p50_ms': float(np.percentile(timings, 50)), 'p95_ms': float(np.percentile(timings, 95)),
                   'max_ms': max(timings), 'reference_max_abs_diff_Wh': max(differences),
                   'reference_path_p95_ms_recorded': result['system']['p95_ms'],
                   'reference_p50_ms_same_origins': float(np.percentile(reference_timings, 50)),
                   'reference_p95_ms_same_origins': float(np.percentile(reference_timings, 95)),
                   'model_sha256': hashlib.sha256((OUT/'bundle.joblib').read_bytes()).hexdigest(),
                   'scope': 'serve.forecast_one; loaded model; input validation + one feature build + required models; 2 threads; excludes file/SQLite/UI',
                   'note': 'Equivalent inference refactor after model lock; no fitting or threshold change'}
    (OUT/'performance.json').write_text(json.dumps(performance, indent=2), encoding='utf-8')
    print(json.dumps(performance, indent=2))


if __name__ == '__main__':
    main()
