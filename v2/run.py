"""Hai giai đoạn tách biệt: development không chấm TEST; final dùng lựa chọn đã khóa.

python -m v2.run development --include-lstm
python -m v2.run final
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
import appliances_core as v1
from tools.check_project import check_frozen_files
from v2 import core, metrics, runtime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'out_v2'
PROTOCOL = {
    'seed': 42, 'folds': v1.FOLDS, 'high_quantile_train': .95,
    'high_mae_improvement_min': .05, 'overall_mae_increase_max': .03,
    'alert_fp_increase_max': .20, 'policy_target_fpr': .10, 'policy_target_recall': .70,
    'policy_development_split': 'First half for calibration, second half for assessment, purge 60min at boundary',
    'random_search_candidates': 24, 'random_search_target': 'direct hour',
    'lstm_max_epochs': 30, 'external_validation': 'Not available: one house and no unused later period supplied',
    'test_status': 'Retrospective: TEST already observed during v1 development; no new generalization claim',
    'production_rule': 'Point must pass high MAE and overall MAE gates on BOTH folds. Policy must not reduce recall or increase FP by >20% versus v1 on BOTH development assessment halves. Unchanged v1 policy is an admissible fallback, not an alert improvement.',
}


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float, allow_nan=False)+'\n', encoding='utf-8')


def sources():
    paths = [ROOT/'appliances_core.py', ROOT/'data/energydata_complete.csv',
             *[ROOT/'v2'/name for name in ('__init__.py', 'core.py', 'metrics.py', 'lstm.py', 'runtime.py', 'run.py')]]
    return {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def config(name, method='hgb_l1', features='A', target='multi', weight=1, params=None, stage='baseline'):
    return dict(name=name, method=method, features=features, target=target, weight=weight,
                params=params or {}, stage=stage)


def candidates():
    rows = [config('v1'), *[config(name, method=name) for name in ('persistence', 'ridge', 'hgb_l2', 'xgb_l1', 'rf')]]
    for target in ('multi', 'direct'):
        for method in ('hgb_l1', 'hgb_l2'):
            for feature in ('A', 'F'):
                for weight in (1, 2, 4, 8):
                    if target == 'multi' and feature == 'A' and weight == 1:
                        continue
                    rows.append(config(f'{target}_{method}_{feature}_w{weight}', method, feature, target, weight,
                                       stage='rise_weight' if target == 'multi' else 'direct'))
    space = dict(learning_rate=[.03, .05, .1], max_iter=[200, 400], max_leaf_nodes=[7, 15, 31],
                 min_samples_leaf=[10, 20, 40, 80], l2_regularization=[.1, 1., 5.],
                 weight=[1, 2, 4, 8], method=['hgb_l1', 'hgb_l2'], features=['A', 'F'])
    for i, sample in enumerate(ParameterSampler(space, n_iter=24, random_state=42)):
        sample = dict(sample)
        weight, method, feature = (sample.pop(key) for key in ('weight', 'method', 'features'))
        rows.append(config(f'tune_{i:02}', method, feature, 'direct', weight, sample, 'tuning'))
    return rows


def fit(config, x, y, df, train_mask, train_end):
    if config['method'] == 'lstm':
        from v2.lstm import fit_lstm
        return fit_lstm(df, train_mask, train_end, max_epochs=PROTOCOL['lstm_max_epochs'])
    if config['target'] == 'direct':
        return core.fit_direct(config['method'], x, y, config['weight'], config['params'])[0]
    return core.fit_point(config['method'], x, y, config['weight'], config['params'] or None)[0]


def predict(config, model, x, df, mask):
    if config['method'] == 'lstm':
        from v2.lstm import predict_lstm
        return predict_lstm(model, df, mask).sum(1)
    return core.predict_hour(config, model, x)


def development(include_lstm):
    check_frozen_files(ROOT)
    if (OUT/'test_evaluation_started.json').exists():
        raise ValueError('TEST đã được chấm cho phiên này; dùng thư mục phiên mới nếu thay đổi lựa chọn')
    save('protocol.json', {**PROTOCOL, 'source_sha256': sources()})
    df, _ = v1.load_raw(ROOT/'data/energydata_complete.csv')
    # Không tạo nhãn/đặc trưng TEST hoặc CAL trong vòng chọn cấu hình.
    df = df[df.t < v1.SPLITS['calibration'][0]].reset_index(drop=True)
    x, y = core.build_features(df), v1.build_labels(df)
    configs = candidates()
    if include_lstm:
        configs.append(config('lstm_small', 'lstm', stage='deep_learning'))
    rows, predictions, folds = [], {}, {}
    for name, (train, valid) in v1.FOLDS.items():
        mt = v1.origin_mask(df.t, *train, x[core.FEATURE_SETS['A']].to_numpy(), y)
        me = v1.origin_mask(df.t, *valid, x[core.FEATURE_SETS['A']].to_numpy(), y)
        total = y[me].sum(1)
        threshold = float(np.quantile(y[mt].sum(1), .95))
        midpoint = pd.Timestamp(valid[0]) + (pd.Timestamp(valid[1])-pd.Timestamp(valid[0]))/2
        times = pd.DatetimeIndex(df.t[me])
        calibration = np.asarray(times + pd.Timedelta(minutes=60) < midpoint)
        assessment = np.asarray(times >= midpoint)
        qmodels = v1.fit_quantiles(x.loc[mt, core.FEATURE_SETS['A']].to_numpy(float), y[mt].sum(1))
        q, _ = v1.predict_quantiles(qmodels, x.loc[me, core.FEATURE_SETS['A']].to_numpy(float))
        margin = v1.calibration_margin(total[calibration], q[calibration])
        folds[name] = {'train': train, 'evaluation': valid, 'threshold': threshold,
                       'train_n': int(mt.sum()), 'eval_n': int(me.sum()),
                       'policy_calibration_n': int(calibration.sum()), 'policy_assessment_n': int(assessment.sum()),
                       'policy_split_time': str(midpoint), 'margin': margin}
        predictions[name+'_truth'] = total
        predictions[name+'_quantiles'] = q
        predictions[name+'_policy_calibration'] = calibration
        predictions[name+'_policy_assessment'] = assessment
        print(f'FOLD {name}: TRAIN={mt.sum()}, evaluation={me.sum()}, T={threshold}', flush=True)
        for item in configs:
            columns = core.FEATURE_SETS[item['features']]
            started = time.perf_counter()
            if item['method'] == 'persistence':
                prediction = 6*df.Appliances.to_numpy(float)[me]
                training = {}
            else:
                model = fit(item, x.loc[mt, columns].to_numpy(float), y[mt], df, mt, train[1])
                prediction = predict(item, model, x.loc[me, columns].to_numpy(float), df, me)
                training = model.get('training', {}) if item['method'] == 'lstm' else {}
            rows.append({'config': item, 'fold': name, 'metrics': metrics.regression(total, prediction, threshold),
                         'seconds': time.perf_counter()-started, 'training': training})
            predictions[name+'_'+item['name']] = prediction
            print(f"  {item['name']}: MAE={rows[-1]['metrics']['mae']:.2f}; high={rows[-1]['metrics']['high_mae']:.2f}", flush=True)
        save('development_partial.json', {'folds': folds, 'rows': rows})
    baseline = {row['fold']: row['metrics'] for row in rows if row['config']['name'] == 'v1'}
    ranking = []
    for item in configs:
        per_fold = [row for row in rows if row['config']['name'] == item['name']]
        gate = [{'fold': row['fold'],
                 'high_mae_reduction': 1-row['metrics']['high_mae']/baseline[row['fold']]['high_mae'],
                 'overall_mae_increase': row['metrics']['mae']/baseline[row['fold']]['mae']-1} for row in per_fold]
        passed = item['name'] != 'v1' and all(g['high_mae_reduction'] >= .05 and g['overall_mae_increase'] <= .03 for g in gate)
        ranking.append({'config': item, 'point_gate_passed': passed, 'gates': gate,
                        'mean_mae': float(np.mean([row['metrics']['mae'] for row in per_fold])),
                        'mean_relative_high_mae': float(np.mean([1-g['high_mae_reduction'] for g in gate]))})
    acceptable = [row for row in ranking if row['point_gate_passed']]
    chosen = min(acceptable, key=lambda row: (row['mean_relative_high_mae'], row['mean_mae']))['config'] if acceptable else configs[0]
    # Các ứng viên cuối được khóa trước TEST, kể cả đối chứng không được chọn.
    finalists = {'v1': configs[0], 'selected': chosen}
    for target in ('multi', 'direct'):
        subset = [row for row in ranking if row['config']['target'] == target and row['config']['method'] not in ('persistence', 'lstm')]
        eligible = [row for row in subset if row['point_gate_passed']]
        finalists['best_'+target] = min(eligible or subset, key=lambda row: (row['mean_relative_high_mae'] if eligible else row['mean_mae'], row['mean_mae']))['config']
    if include_lstm:
        finalists['lstm'] = configs[-1]
    policy_rows = []
    for fold, info in folds.items():
        actual = predictions[fold+'_truth']
        q = predictions[fold+'_quantiles']
        mc, ma = predictions[fold+'_policy_calibration'], predictions[fold+'_policy_assessment']
        point = predictions[fold+'_'+chosen['name']]
        for policy in ('v1', 'raw_p90', 'upper', 'point'):
            score = runtime.score_for_policy(policy, point, q, info['margin'])
            for mode in (('fixed',) if policy == 'v1' else ('fixed_fpr', 'fixed_recall')):
                threshold = info['threshold'] if mode == 'fixed' else (
                    metrics.threshold_for_fpr(actual[mc], score[mc], info['threshold'], .10) if mode == 'fixed_fpr' else
                    metrics.threshold_for_recall(actual[mc], score[mc], info['threshold'], .70))
                policy_rows.append({'fold': fold, 'name': policy, 'mode': mode,
                                    'calibration': metrics.alerts(actual[mc], score[mc], info['threshold'], threshold),
                                    'assessment': metrics.alerts(actual[ma], score[ma], info['threshold'], threshold)})
    policy_eligible = []
    for policy in ('v1', 'raw_p90', 'upper', 'point'):
        selected_rows = [r for r in policy_rows if r['name'] == policy and r['mode'] == ('fixed' if policy == 'v1' else 'fixed_fpr')]
        passed = True
        for row in selected_rows:
            b = next(r['assessment'] for r in policy_rows if r['fold'] == row['fold'] and r['name'] == 'v1')
            a = row['assessment']
            passed &= a['recall'] >= b['recall'] and a['fp'] <= b['fp']*1.20
        if passed:
            policy_eligible.append((policy, float(np.mean([r['assessment']['recall'] for r in selected_rows])),
                                    sum(r['assessment']['fp'] for r in selected_rows)))
    policy = sorted(policy_eligible, key=lambda r: (-r[1], r[2], r[0] != 'v1'))[0][0]
    lock = {'protocol': PROTOCOL, 'source_sha256': sources(), 'selected_point': chosen,
            'selected_policy': policy, 'finalists': finalists,
            'point_candidates_passing_gate': len(acceptable),
            'production_version': 'v1' if chosen['name'] == 'v1' and policy == 'v1' else 'v2',
            'test_used_for_selection': False, 'validation_is_fold_B': True,
            'reason': 'Both-fold point gates and chronological policy assessment; unchanged v1 is fallback'}
    save('development.json', {'protocol': PROTOCOL, 'folds': folds, 'rows': rows, 'ranking': ranking, 'policies': policy_rows, 'selection': lock})
    np.savez_compressed(OUT/'development_predictions.npz', **predictions)
    save('high_group_models.json', {'folds': folds, 'rows': [r for r in rows if r['config']['stage'] == 'baseline']})
    save('selection_lock.json', lock)
    print('LOCKED:', chosen['name'], 'policy=', policy, 'version=', lock['production_version'], flush=True)


def final():
    check_frozen_files(ROOT)
    lock_path = OUT/'selection_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    if sources() != lock['source_sha256']:
        raise ValueError('Mã thực nghiệm thay đổi sau khi khóa lựa chọn; cần chạy lại development trước TEST')
    if (OUT/'test_evaluation_started.json').exists():
        raise ValueError('Phiên này đã bắt đầu đánh giá TEST. Dùng python -m v2.verify để kiểm lại, không chọn lại cấu hình.')
    df, audit = v1.load_raw(ROOT/'data/energydata_complete.csv')
    x, y = core.build_features(df), v1.build_labels(df)
    masks = {name: v1.origin_mask(df.t, *bounds, x[core.FEATURE_SETS['F']].to_numpy(), y) for name, bounds in v1.SPLITS.items()}
    legacy = joblib.load(ROOT/'out/bundle.joblib')
    threshold = float(np.quantile(y[masks['train']].sum(1), .95))
    models, seconds = {}, {}
    for item in lock['finalists'].values():
        name = item['name']
        if name in models:
            continue
        started = time.perf_counter()
        if name == 'v1':
            model = legacy['point_model']
        else:
            columns = core.FEATURE_SETS[item['features']]
            model = fit(item, x.loc[masks['train'], columns].to_numpy(float), y[masks['train']], df, masks['train'], v1.SPLITS['train'][1])
        models[name], seconds[name] = model, time.perf_counter()-started
    selected = lock['selected_point']
    bundle = {'version': lock['production_version'], 'legacy': legacy, 'config': selected,
              'candidate_model': models[selected['name']], 'policy': {'name': lock['selected_policy'], 'threshold': threshold},
              'selection_lock_sha256': hashlib.sha256(lock_path.read_bytes()).hexdigest()}
    # CAL chỉ hiệu chỉnh dải và ngưỡng của chính sách đã chọn, không đổi mô hình.
    _, cal_hour, cal_q, _ = runtime.forecast_batch(bundle, df, masks['calibration'])
    cal_y = y[masks['calibration']].sum(1)
    margin = v1.calibration_margin(cal_y, cal_q)
    np.testing.assert_allclose(margin, legacy['margin'], atol=1e-10)
    if lock['selected_policy'] != 'v1':
        cal_score = runtime.score_for_policy(lock['selected_policy'], cal_hour, cal_q, margin)
        bundle['policy']['threshold'] = metrics.threshold_for_fpr(cal_y, cal_score, threshold, .10)
    joblib.dump(bundle, OUT/'bundle.joblib')
    joblib.dump({'models': models, 'configs': lock['finalists']}, OUT/'finalists.joblib')
    save('test_evaluation_started.json', {'selection_lock_sha256': bundle['selection_lock_sha256'], 'protocol': PROTOCOL, 'model_sha256': hashlib.sha256((OUT/'bundle.joblib').read_bytes()).hexdigest()})
    steps, hour, q, crossing = runtime.forecast_batch(bundle, df, masks['test'])
    actual = y[masks['test']].sum(1)
    results = {'version': bundle['version'], 'selection': lock, 'threshold': threshold, 'margin': margin,
               'n_test': len(actual), 'final_fit_seconds': seconds,
               'selected': metrics.regression(actual, hour, threshold), 'finalists': {},
               'interval': metrics.interval(actual, q, margin, threshold),
               'calibration_interval': metrics.interval(cal_y, cal_q, margin, threshold),
               'crossing_rows': int(crossing.sum()), 'external_validation': PROTOCOL['external_validation']}
    saved = pd.DataFrame({'issue_time': df.loc[masks['test'], 't'].to_numpy(), 'actual_hour_total': actual,
                          'selected_hour_prediction': hour, 'point_hour_total': steps.sum(1),
                          'p10': q[:, 0], 'p50': q[:, 1], 'p90': q[:, 2],
                          'lower_cal': np.maximum(q[:, 0]-margin, 0), 'upper_cal': q[:, 2]+margin})
    for i, k in enumerate(range(10, 61, 10)):
        saved[f'actual_{k}m'] = y[masks['test']][:, i]
        saved[f'pred_{k}m'] = steps[:, i]
    for label, item in lock['finalists'].items():
        prediction = predict(item, models[item['name']], x.loc[masks['test'], core.FEATURE_SETS[item['features']]].to_numpy(float), df, masks['test'])
        saved['hour_'+label] = prediction
        results['finalists'][label] = {'config': item, 'metrics': metrics.regression(actual, prediction, threshold),
                                      'training': models[item['name']].get('training', {}) if item['method'] == 'lstm' else {}}
    score = runtime.score_for_policy(bundle['policy']['name'], hour, q, margin)
    saved['alert_score'] = score
    saved['alert_score_threshold'] = bundle['policy']['threshold']
    saved['alert'] = score > bundle['policy']['threshold']
    saved['truth_high'] = actual > threshold
    results['alert'] = metrics.alerts(actual, score, threshold, bundle['policy']['threshold'])
    results['policy_comparison'] = {}
    for policy in ('v1', 'raw_p90', 'upper', 'point'):
        cs = runtime.score_for_policy(policy, cal_hour, cal_q, margin)
        ts = runtime.score_for_policy(policy, hour, q, margin)
        policy_results = {}
        for mode in ('fixed_energy', 'fixed_fpr', 'fixed_recall'):
            operating_threshold = threshold if mode == 'fixed_energy' else (
                metrics.threshold_for_fpr(cal_y, cs, threshold, .10) if mode == 'fixed_fpr' else
                metrics.threshold_for_recall(cal_y, cs, threshold, .70))
            policy_results[mode] = metrics.alerts(actual, ts, threshold, operating_threshold)
        results['policy_comparison'][policy] = policy_results
    results['bootstrap'] = metrics.block_bootstrap(actual, hour, saved.hour_v1.to_numpy())
    results['split_metrics'] = {}
    for name, mask in masks.items():
        prediction = runtime.forecast_batch(bundle, df, mask)[1]
        results['split_metrics'][name] = metrics.regression(y[mask].sum(1), prediction, threshold)
    saved.to_csv(OUT/'test_predictions.csv', index=False)
    # Đo toàn luồng dự báo khi model đã nạp, không gồm tệp/SQLite/UI.
    timings = []
    for i in np.random.default_rng(42).choice(np.flatnonzero(masks['test']), 200, replace=False):
        started = time.perf_counter()
        runtime.forecast_one(bundle, df.iloc[i-v1.DAY:i+1])
        timings.append((time.perf_counter()-started)*1000)
    results['system'] = {'model_bytes': (OUT/'bundle.joblib').stat().st_size, 'latency_n': len(timings),
                         'p50_ms': float(np.percentile(timings, 50)), 'p95_ms': float(np.percentile(timings, 95)),
                         'scope': 'Loaded model; validation + features + prediction; excludes model loading, file I/O, SQLite, UI'}
    results['robustness'] = []
    for kind, level in [('noise', .1), ('dropout', .1), ('meter_scale', .9), ('meter_scale', 1.1)]:
        trials = []
        for seed in range(5 if kind in ('noise', 'dropout') else 1):
            changed = v1.perturb(df, kind, level, seed)
            _, ph, qq, _ = runtime.forecast_batch(bundle, changed, masks['test'])
            sc = runtime.score_for_policy(bundle['policy']['name'], ph, qq, margin)
            a = metrics.alerts(actual, sc, threshold, bundle['policy']['threshold'])
            trials.append({'mae': float(np.mean(np.abs(actual-ph))), 'fp': a['fp'], 'fn': a['fn'],
                           'coverage': metrics.interval(actual, qq, margin, threshold)['coverage']})
        results['robustness'].append({'scenario': f'{kind}={level}', 'replicates': len(trials),
                                      **{k: float(np.mean([t[k] for t in trials])) for k in trials[0]}})
    save('results.json', results)
    print('FINAL:', bundle['version'], results['selected'], 'alert=', results['alert'], flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['development', 'final'])
    parser.add_argument('--include-lstm', action='store_true')
    args = parser.parse_args()
    if args.stage == 'development':
        development(args.include_lstm)
    else:
        final()


if __name__ == '__main__':
    main()
