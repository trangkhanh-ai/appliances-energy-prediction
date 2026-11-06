"""Tính lại từ CSV gốc và model tự huấn luyện, không ép số theo báo cáo cũ.

Chỉ nạp out/bundle.joblib do chính pipeline này tạo. Không nạp model tải về.
"""
from pathlib import Path
import hashlib, importlib.metadata, json, platform, sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import appliances_core as c
from ridge_scratch import RidgeScratch

P = Path(__file__).resolve().parent
O = P / 'out'
R = json.loads((O / 'results.json').read_text(encoding='utf-8'))
B = joblib.load(O / 'bundle.joblib')
D, audit = c.load_raw(P / 'data/energydata_complete.csv')
X = c.build_features(D)[B['features']].to_numpy(float)
Y = c.build_labels(D)
tr = c.origin_mask(D.t, *c.SPLITS['train'], X, Y)
te = c.origin_mask(D.t, *c.SPLITS['test'], X, Y)
ca = c.origin_mask(D.t, *c.SPLITS['calibration'], X, Y)
saved = pd.read_csv(O / 'test_predictions.csv', parse_dates=['issue_time'])
origins = np.where(te)[0]
labels = np.array([D.Appliances.iloc[i+1:i+7].to_numpy(float) for i in origins])
np.testing.assert_array_equal(labels, saved[[f'actual_{h}m' for h in range(10,61,10)]].to_numpy())
np.testing.assert_array_equal(pd.DatetimeIndex(D.t[te]), pd.DatetimeIndex(saved.issue_time))
point = c.predict_point(B['point_name'], B['point_model'], X[te])
qs = c.predict_quantiles(B['quantile_models'], X[te])[0]
np.testing.assert_allclose(point, saved[[f'pred_{h}m' for h in range(10,61,10)]], rtol=0, atol=1e-10)
np.testing.assert_allclose(qs, saved[['p10','p50','p90']], rtol=0, atol=1e-10)
lo, up = np.maximum(qs[:,0]-B['margin'], 0), qs[:,2]+B['margin']
truth = labels.sum(1) > B['threshold']; alarm = up > B['threshold']
A = R['alert']['ai_calibrated_upper']
counts = {'tp':int((truth&alarm).sum()), 'fp':int((~truth&alarm).sum()),
          'fn':int((truth&~alarm).sum()), 'tn':int((~truth&~alarm).sum())}
assert all(A[k] == v for k,v in counts.items())
mae = float(mean_absolute_error(labels.sum(1), point.sum(1)))
coverage = float(np.mean((labels.sum(1)>=lo)&(labels.sum(1)<=up)))
np.testing.assert_allclose(mae, R['point']['test'][B['point_name']]['mae_hour'], rtol=0, atol=1e-10)
np.testing.assert_allclose(coverage, R['quantile_test']['calibrated_coverage'], rtol=0, atol=1e-12)
np.testing.assert_allclose(B['threshold'], np.quantile(Y[tr].sum(1), .95))
cal_q = c.predict_quantiles(B['quantile_models'], X[ca])[0]
np.testing.assert_allclose(B['margin'], c.calibration_margin(Y[ca].sum(1), cal_q))
online_error = 0.
for j in np.random.default_rng(0).choice(len(origins),200,replace=False):
    i = origins[j]; one = c.forecast_one(B,D.iloc[i-c.DAY:i+1])
    online_error = max(online_error,float(np.max(np.abs(np.r_[one['point_steps_Wh'],one['p10'],one['p50'],one['p90'],one['lower_cal'],one['upper_cal']]-np.r_[point[j],qs[j],lo[j],up[j]]))))
assert online_error <= 1e-10
scratch = {}
for alpha in (1.,100.):
    own = RidgeScratch(alpha).fit(X[tr],Y[tr]).predict(X[te])
    ref = make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(X[tr],Y[tr]).predict(X[te])
    scratch[str(alpha)] = float(np.max(np.abs(own-ref)))
    assert scratch[str(alpha)] < 1e-8
# Paired circular block resampling: keeps nearby origins together, models frozen.
# These are descriptive uncertainty intervals on this period, not an independent future guarantee.
err = np.abs(labels.sum(1)-point.sum(1))
base_err = np.abs(labels.sum(1)-6*D.Appliances.to_numpy()[te])
rng = np.random.default_rng(42); bootstrap = {}
for size in (6,144):
    draws=[]; gains=[]
    for _ in range(1000):
        starts = rng.integers(0,len(err),int(np.ceil(len(err)/size)))
        idx=((starts[:,None]+np.arange(size))%len(err)).ravel()[:len(err)]
        draws.append(err[idx].mean()); gains.append(100*(1-err[idx].mean()/base_err[idx].mean()))
    bootstrap[str(size)]={'replicates':1000,'mae_Wh_percentile95':np.percentile(draws,[2.5,97.5]).tolist(),
                           'gain_percent_percentile95':np.percentile(gains,[2.5,97.5]).tolist()}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
out={'passed':True,'test_origins':len(origins),'labels_checked':int(labels.size),
     'independent_counts':counts,'mae_hour_Wh':mae,'coverage':coverage,'online_200_max_abs_diff':online_error,
     'ridge_scratch_max_difference':scratch,'paired_block_bootstrap':bootstrap,
     'python':sys.version,'platform':platform.platform(),
     'packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','xgboost','scipy','joblib','matplotlib']},
     'data_sha256':audit['sha256'],'model_sha256':digest(O/'bundle.joblib'),
     'source_sha256':{p.name:digest(p) for p in sorted(P.glob('*.py'))}}
(O/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
