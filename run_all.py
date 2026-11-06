"""run_all.py - Chạy toàn bộ thực nghiệm đồ án (UCI Appliances) và ghi out/results.json, out/figures/*.png,
out/test_predictions.csv, out/bundle.joblib.

Chạy:  python run_all.py            (cần data/energydata_complete.csv; ~2-4 phút trên 2 CPU)
Quy tắc chọn mô hình được cố định TRƯỚC khi nhìn TEST (xem SELECTION_RULE); TEST chỉ dùng để báo cáo.
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import appliances_core as c  # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / "out"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SELECTION_RULE = ("Chọn theo MAE/bước có trọng số số origin trên hai lượt walk-forward (tháng 3; 1-14/4). "
                  "Nếu chênh <1% thì chọn phương án đơn giản hơn. TEST không tham gia chọn.")
BASELINES = ("persistence", "diurnal_median", "seasonal_day")
LEARNED = ("ridge", "hgb_l1", "hgb_l2", "xgb_l1", "rf")
SIMPLE_ORDER = list(BASELINES) + list(LEARNED)          # thứ tự đơn giản -> phức tạp (dùng khi hòa)
RATIOS = (1, 2, 5, 10, 20)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 10, "font.family": "Times New Roman"})

df, audit = c.load_raw(ROOT / "data" / "energydata_complete.csv")
X, Y = c.build_features(df), c.build_labels(df)
YH = Y.sum(1)
R = {"audit": audit, "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
     "selection_rule": SELECTION_RULE, "splits": {k: list(v) for k, v in c.SPLITS.items()}}
try:
    import xgboost
    R["versions"]["xgboost"] = xgboost.__version__
except ImportError:
    pass


def mask(rng, cols):
    return c.origin_mask(df.t, rng[0], rng[1], X[cols].to_numpy(), Y)


def run_fold(name, cols, train_rng, test_rng, Xtab=None):
    Xt = X if Xtab is None else Xtab
    mt, me = c.origin_mask(df.t, *train_rng, Xt[cols].to_numpy(), Y), c.origin_mask(df.t, *test_rng, Xt[cols].to_numpy(), Y)
    if name in BASELINES:
        P = c.baseline_predictions(df, np.where(me)[0], np.where(mt)[0])[name]
    else:
        model = c.fit_point(name, Xt.loc[mt, cols].to_numpy(float), Y[mt])
        P = c.predict_point(name, model, Xt.loc[me, cols].to_numpy(float))
    return c.reg_metrics(Y[me], P), int(me.sum())


def pick(scores, tol=0.01):
    """scores: {tên: MAE có trọng số}. Chọn nhỏ nhất; nếu trong 1% của nhỏ nhất thì lấy phương án đơn giản nhất."""
    best = min(scores.values())
    near = [k for k, v in scores.items() if v <= best * (1 + tol)]
    return min(near, key=lambda k: SIMPLE_ORDER.index(k) if k in SIMPLE_ORDER else list(scores).index(k))


def weighted(folds):
    n = sum(f[1] for f in folds)
    return sum(f[0]["mae_step"] * f[1] for f in folds) / n


# ------------------------------------------------------------------ 1. Chọn tập feature (HGB L1)
fa, fb = c.FOLDS["walk_A_march"], c.FOLDS["walk_B_april"]
R["feature_sets"] = {}
fs_scores = {}
for fs, cols in c.FEATURE_SETS.items():
    folds = [run_fold("hgb_l1", cols, *fa), run_fold("hgb_l1", cols, *fb)]
    R["feature_sets"][fs] = {"n_features": len(cols), "mae_step_A_march": folds[0][0]["mae_step"],
                             "mae_step_B_april": folds[1][0]["mae_step"], "mae_step_weighted": weighted(folds),
                             "mae_hour_A": folds[0][0]["mae_hour"], "mae_hour_B": folds[1][0]["mae_hour"]}
    if fs != "E_D_plus_rv1_control":                       # E chỉ là đối chứng nhiễu, không được chọn
        fs_scores[fs] = weighted(folds)
sizes = {k: len(v) for k, v in c.FEATURE_SETS.items()}
best = min(fs_scores.values())
near = [k for k, v in fs_scores.items() if v <= best * 1.01]
FS = min(near, key=lambda k: sizes[k])
COLS = c.FEATURE_SETS[FS]
R["selected_feature_set"] = FS
print("feature set:", FS, {k: round(v, 3) for k, v in fs_scores.items()}, flush=True)

# ------------------------------------------------------------------ 2. Chọn phương pháp point
R["method_selection"] = {}
sel = {}
for name in BASELINES + LEARNED:
    t0 = time.perf_counter()
    folds = [run_fold(name, COLS, *fa), run_fold(name, COLS, *fb)]
    R["method_selection"][name] = {"A_march": folds[0][0], "B_april": folds[1][0], "mae_step_weighted": weighted(folds),
                                   "seconds": time.perf_counter() - t0}
    sel[name] = weighted(folds)
POINT = pick(sel)
R["selected_point_method"] = POINT
c.save_json(OUT / "selection.json", {"feature_set": FS, "point_method": POINT,
    "features": COLS, "rule": SELECTION_RULE, "folds": c.FOLDS,
    "point_parameters": c.POINT_PARAMS, "feature_scores": fs_scores, "method_scores": sel})
print("point:", POINT, {k: round(v, 3) for k, v in sel.items()}, flush=True)

# ------------------------------------------------------------------ 3. Huấn luyện cuối trên TRAIN, đánh giá val/cal/test
M = {k: mask(c.SPLITS[k], COLS) for k in c.SPLITS}
IDX = {k: np.where(m)[0] for k, m in M.items()}
Xs = {k: X.loc[m, COLS].to_numpy(float) for k, m in M.items()}
Ys = {k: Y[m] for k, m in M.items()}
R["split_rows"] = {k: int(m.sum()) for k, m in M.items()}
models, fit_seconds, preds = {}, {}, {}
for name in LEARNED:
    t0 = time.perf_counter()
    models[name] = c.fit_point(name, Xs["train"], Ys["train"])
    fit_seconds[name] = time.perf_counter() - t0
for part in ("train", "validation", "calibration", "test"):
    base = c.baseline_predictions(df, IDX[part], IDX["train"])
    preds[part] = {**base, **{n: c.predict_point(n, models[n], Xs[part]) for n in LEARNED}}
R["final_fit_seconds"] = fit_seconds
R["point"] = {part: {n: c.reg_metrics(Ys[part], P) for n, P in d.items()} for part, d in preds.items()}

# ------------------------------------------------------------------ 4. Phân vị + calibration + ngưỡng
Ttrain = float(np.quantile(Ys["train"].sum(1), 0.95))
p50_train = float(np.median(Ys["train"].sum(1)))
qmodels = c.fit_quantiles(Xs["train"], Ys["train"].sum(1))
q = {k: c.predict_quantiles(qmodels, Xs[k]) for k in ("calibration", "test", "validation")}
margin = c.calibration_margin(Ys["calibration"].sum(1), q["calibration"][0])
yh = {k: Ys[k].sum(1) for k in Ys}
P_t = preds["test"][POINT]
ps = {k: preds[k][POINT].sum(1) for k in ("calibration", "test")}
buffer = float(np.quantile(np.maximum(yh["calibration"] - ps["calibration"], 0), 0.9))   # đối chứng: point + buffer
qt = q["test"][0]
lo, up = np.maximum(qt[:, 0] - margin, 0), qt[:, 2] + margin
R["threshold_Wh_per_hour"] = Ttrain
R["train_p50_hour_total"] = p50_train
R["calibration_margin_Wh"] = margin
R["point_plus_buffer_Wh"] = buffer
R["quantile_test"] = {
    "n": len(yh["test"]), "raw_coverage": float(np.mean((yh["test"] >= qt[:, 0]) & (yh["test"] <= qt[:, 2]))),
    "calibrated_coverage": float(np.mean((yh["test"] >= lo) & (yh["test"] <= up))),
    "calibrated_mean_width": float(np.mean(up - lo)), "raw_mean_width": float(np.mean(qt[:, 2] - qt[:, 0])),
    "frac_y_le_p10": float(np.mean(yh["test"] <= qt[:, 0])), "frac_y_le_p50": float(np.mean(yh["test"] <= qt[:, 1])),
    "frac_y_le_p90": float(np.mean(yh["test"] <= qt[:, 2])), "crossing_rows": int(q["test"][1].sum()),
    "pinball": {str(a): float(sklearn.metrics.mean_pinball_loss(yh["test"], qt[:, i], alpha=a)) for i, a in enumerate(c.QUANTILES)}}
cal_q = q["calibration"][0]
R["quantile_calibration_set"] = {"raw_coverage": float(np.mean((yh["calibration"] >= cal_q[:, 0]) & (yh["calibration"] <= cal_q[:, 2]))),
                                 "calibrated_coverage": float(np.mean((yh["calibration"] >= np.maximum(cal_q[:, 0] - margin, 0)) &
                                                                      (yh["calibration"] <= cal_q[:, 2] + margin)))}

# ------------------------------------------------------------------ 5. Cảnh báo: AI so với luật không-AI
yt = yh["test"]
truth = yt > Ttrain
A0 = X.loc[M["test"], "app_lag_0"].to_numpy(float)
scores = {"persistence_x6": 6 * A0, "last_hour_sum": X.loc[M["test"], "app_sum_1h"].to_numpy(float),
          "yesterday_same_window": X.loc[M["test"], "app_yday_window_sum"].to_numpy(float),
          "ai_point_sum": P_t.sum(1), "ai_point_plus_buffer": P_t.sum(1) + buffer,
          "ai_raw_p90": qt[:, 2], "ai_calibrated_upper": up}
days_test = len(yt) / c.DAY
R["alert"] = {}
for name, s in scores.items():
    m = c.alert_metrics(yt, s, Ttrain)
    e = c.episode_metrics(truth, s > Ttrain)
    lead = e.pop("lead_minutes")
    R["alert"][name] = {**m, "precision_ci95": c.wilson(m["tp"], m["tp"] + m["fp"]), "recall_ci95": c.wilson(m["tp"], m["tp"] + m["fn"]),
                        "episodes": e, "lead_minutes_sorted": sorted(lead),
                        "alert_origins_per_day": m["alerts"] / days_test,
                        "cost": c.cost_curve(yt, s, Ttrain, RATIOS)}
npos = int(truth.sum())
R["alert"]["no_alert"] = {"fp": 0, "fn": npos, "cost": {str(r): r * npos for r in RATIOS}}
R["alert"]["always_alert"] = {"fp": len(yt) - npos, "fn": 0, "cost": {str(r): len(yt) - npos for r in RATIOS}}
excess = np.maximum(yt - Ttrain, 0)
R["exposure_covered"] = {n: float(excess[s > Ttrain].sum() / excess.sum()) for n, s in scores.items()}
R["test_positive_origins"] = npos
R["test_days"] = days_test
# chọn buffer cộng thêm theo tỷ lệ chi phí trên CALIBRATION, áp dụng cho TEST
up_cal = q["calibration"][0][:, 2] + margin
R["cost_tuned_on_calibration"] = {}
for r in RATIOS:
    grid = np.arange(-600, 801, 20)
    b = min(grid, key=lambda g: (c.cost_curve(yh["calibration"], up_cal + g, Ttrain, [r])[str(r)], abs(g)))
    m = c.alert_metrics(yt, up + b, Ttrain)
    R["cost_tuned_on_calibration"][str(r)] = {"extra_buffer_Wh": float(b), "fp": m["fp"], "fn": m["fn"],
                                              "recall": m["recall"], "precision": m["precision"], "cost": m["fp"] + r * m["fn"]}

# ------------------------------------------------------------------ 6. Phân tích lỗi trên TEST
per_step = [{"minutes": 10 * (i + 1), "mae": float(np.mean(np.abs(Ys["test"][:, i] - P_t[:, i]))),
             "rmse": float(np.sqrt(np.mean((Ys["test"][:, i] - P_t[:, i]) ** 2)))} for i in range(c.HORIZON)]
R["per_step_test"] = per_step
ph = P_t.sum(1)
lvl = {"low": yt <= p50_train, "middle": (yt > p50_train) & (yt <= Ttrain), "high": yt > Ttrain}
R["by_level_test"] = {k: {"n": int(m.sum()), "mae_hour": float(np.mean(np.abs(yt[m] - ph[m]))),
                          "mean_residual_actual_minus_pred": float(np.mean(yt[m] - ph[m])),
                          "coverage": float(np.mean(((yt >= lo) & (yt <= up))[m])),
                          "underpredicted_share": float(np.mean(ph[m] < yt[m]))} for k, m in lvl.items()}
hr = pd.DatetimeIndex(df.t[M["test"]]).hour
R["by_hour_test"] = {int(h): {"n": int((hr == h).sum()), "mae_hour": float(np.mean(np.abs(yt[hr == h] - ph[hr == h])))} for h in range(24)}
d60 = X.loc[M["test"], "app_lag_0"].to_numpy(float) - X.loc[M["test"], "app_lag_6"].to_numpy(float)
dthr = float(np.quantile(np.abs(X.loc[M["train"], "app_lag_0"].to_numpy(float) - X.loc[M["train"], "app_lag_6"].to_numpy(float)), 0.95))
st = {"rising": d60 > dthr, "falling": d60 < -dthr, "stable": np.abs(d60) <= dthr}
R["by_state_test"] = {"threshold_p95_abs_delta_60min": dthr,
                      **{k: {"n": int(m.sum()), "mae_hour": float(np.mean(np.abs(yt[m] - ph[m]))) if m.any() else None,
                             "mean_residual": float(np.mean(yt[m] - ph[m])) if m.any() else None} for k, m in st.items()}}
# top lỗi dự báo thiếu
order = np.argsort(-(yt - ph))[:5]
times_test = pd.DatetimeIndex(df.t[M["test"]])
R["worst_underpredictions"] = [{"issue_time": str(times_test[i]), "actual": float(yt[i]), "point": float(ph[i]), "upper": float(up[i]),
                                "last_reading": float(A0[i])} for i in order]

# ------------------------------------------------------------------ 7. System-level: latency, kích thước, parity online/batch
bundle = {"point_name": POINT, "point_model": models[POINT], "features": COLS, "quantile_models": qmodels,
          "margin": margin, "threshold": Ttrain}
joblib.dump(bundle, OUT / "bundle.joblib")
R["system"] = {"bundle_bytes": (OUT / "bundle.joblib").stat().st_size, "latency_ms": c.measure_latency(bundle, df, n=200),
               "cpu_count": __import__("os").cpu_count()}
rng = np.random.default_rng(0)
chk = rng.choice(len(yt), 200, replace=False)
diffs = []
for j in chk:
    i = int(IDX["test"][j])
    one = c.forecast_one(bundle, df.iloc[i - c.DAY: i + 1])
    diffs.append(abs(one["point_sum_Wh"] - ph[j]) + abs(one["upper_cal"] - up[j]))
R["system"]["online_vs_batch_max_abs_diff"] = float(np.max(diffs))

# ------------------------------------------------------------------ 8. Stress test (feature từ dữ liệu bị nhiễu, nhãn thật giữ nguyên)
scen = [("noise", 0.05), ("noise", 0.10), ("dropout", 0.05), ("dropout", 0.10), ("meter_scale", 0.9), ("meter_scale", 1.1),
        ("temp_shift", 3.0), ("temp_shift", -3.0), ("weather_stuck", 0.0)]
stress = [{"scenario": "clean", "mae_hour": float(np.mean(np.abs(yt - ph))), "coverage": R["quantile_test"]["calibrated_coverage"],
           "fp": R["alert"]["ai_calibrated_upper"]["fp"], "fn": R["alert"]["ai_calibrated_upper"]["fn"],
           "recall": R["alert"]["ai_calibrated_upper"]["recall"], "precision": R["alert"]["ai_calibrated_upper"]["precision"]}]
for kind, level in scen:
    out = []
    for seed in range(5 if kind in ("noise", "dropout") else 1):
        Xp = c.build_features(c.perturb(df, kind, level, seed))
        xp = Xp.loc[M["test"], COLS].to_numpy(float)
        if not np.isfinite(xp).all():
            continue
        pp = c.predict_point(POINT, models[POINT], xp)
        qq = c.predict_quantiles(qmodels, xp)[0]
        l2, u2 = np.maximum(qq[:, 0] - margin, 0), qq[:, 2] + margin
        a = c.alert_metrics(yt, u2, Ttrain)
        out.append({"mae_hour": float(np.mean(np.abs(yt - pp.sum(1)))), "coverage": float(np.mean((yt >= l2) & (yt <= u2))),
                    "fp": a["fp"], "fn": a["fn"], "recall": a["recall"], "precision": a["precision"]})
    stress.append({"scenario": f"{kind}={level}", **{k: float(np.mean([o[k] for o in out])) for k in out[0]}})
R["stress_test"] = stress

# ------------------------------------------------------------------ 9. Permutation importance (validation, mô hình cuối)
base_mae = float(np.mean(np.abs(yh["validation"] - preds["validation"][POINT].sum(1))))
imp = {}
rng = np.random.default_rng(0)
for j, col in enumerate(COLS):
    deltas = []
    for _ in range(5):
        xp = Xs["validation"].copy()
        xp[:, j] = rng.permutation(xp[:, j])
        deltas.append(float(np.mean(np.abs(yh["validation"] - c.predict_point(POINT, models[POINT], xp).sum(1)))) - base_mae)
    imp[col] = float(np.mean(deltas))
R["permutation_importance_validation_mae_hour_increase"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
R["validation_base_mae_hour"] = base_mae

# ------------------------------------------------------------------ 10. Độ nhạy múi giờ/DST (chỉ báo cáo, không dùng để chọn)
t_loc = pd.DatetimeIndex(df.t) + pd.to_timedelta(np.where(df.t >= "2016-03-27 01:00", 1, 0), unit="h")
start = t_loc + c.STEP
hour = np.asarray(start.hour + start.minute / 60)
dow = np.asarray(start.dayofweek)
X2 = X.copy()
X2["hour_sin"], X2["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
X2["dow_sin"], X2["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
X2["is_weekend"] = (dow >= 5).astype(float)
fa2, fb2 = run_fold(POINT, COLS, *fa, Xtab=X2), run_fold(POINT, COLS, *fb, Xtab=X2)
mt2 = c.origin_mask(df.t, *c.SPLITS["train"], X2[COLS].to_numpy(), Y)
mdl2 = c.fit_point(POINT, X2.loc[mt2, COLS].to_numpy(float), Y[mt2])
mte2 = c.origin_mask(df.t, *c.SPLITS["test"], X2[COLS].to_numpy(), Y)
m2 = c.reg_metrics(Y[mte2], c.predict_point(POINT, mdl2, X2.loc[mte2, COLS].to_numpy(float)))
R["dst_sensitivity"] = {"mae_step_A_march": [R["method_selection"][POINT]["A_march"]["mae_step"], fa2[0]["mae_step"]],
                        "mae_step_B_april": [R["method_selection"][POINT]["B_april"]["mae_step"], fb2[0]["mae_step"]],
                        "mae_step_test": [R["point"]["test"][POINT]["mae_step"], m2["mae_step"]],
                        "note": "mỗi cặp = [timestamp gốc, +1h sau 27/3 01:00]; chỉ báo cáo, không dùng để chọn"}
hh = df.assign(h=df.t.dt.hour)
R["dst_profile_peak_hour"] = {"pre_2016-02-01_to_03-26": int(hh[(hh.t >= "2016-02-01") & (hh.t < "2016-03-27")].groupby("h").Appliances.mean().idxmax()),
                              "post_03-28_to_04-30": int(hh[(hh.t >= "2016-03-28") & (hh.t < "2016-05-01")].groupby("h").Appliances.mean().idxmax())}

# ------------------------------------------------------------------ 11. Lưu kết quả + dự đoán
pd.DataFrame({"issue_time": times_test, **{f"actual_{10*(i+1)}m": Ys["test"][:, i] for i in range(6)},
              **{f"pred_{10*(i+1)}m": P_t[:, i] for i in range(6)}, "actual_hour_total": yt, "point_hour_total": ph,
              "p10": qt[:, 0], "p50": qt[:, 1], "p90": qt[:, 2], "lower_cal": lo, "upper_cal": up,
              "alert": up > Ttrain, "truth_high": truth}).to_csv(OUT / "test_predictions.csv", index=False)
c.save_json(OUT / "results.json", R)

# ------------------------------------------------------------------ 12. Hình
A = df.Appliances
tr0 = df[(df.t >= "2016-01-12") & (df.t < "2016-01-19")]
fig, ax = plt.subplots(figsize=(9, 2.6))
ax.plot(tr0.t, tr0.Appliances, lw=.8)
ax.set_ylabel("Appliances (Wh/10 phút)")
ax.set_title("Chuỗi Appliances, tuần đầu của tập train")
fig.tight_layout(); fig.savefig(FIG / "f31_first_week.png"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
axs[0].hist(A, bins=60); axs[0].set_yscale("log"); axs[0].set_xlabel("Appliances (Wh/10 phút)"); axs[0].set_title("Phân bố (trục y log)")
axs[1].hist(YH[np.isfinite(YH)][:len(YH)], bins=60); axs[1].set_yscale("log"); axs[1].set_xlabel("Tổng 6 bước tiếp theo (Wh)")
axs[1].axvline(Ttrain, color="r", ls="--", label=f"p95 train = {Ttrain:.0f}"); axs[1].legend(); axs[1].set_title("Nhãn tổng giờ tới")
fig.tight_layout(); fig.savefig(FIG / "f32_distribution.png"); plt.close(fig)

trn = df[df.t < "2016-04-01"]
hm = trn.assign(h=trn.t.dt.hour, d=trn.t.dt.dayofweek).pivot_table(index="d", columns="h", values="Appliances", aggfunc="mean")
fig, ax = plt.subplots(figsize=(9, 2.8))
im = ax.imshow(hm.values, aspect="auto", cmap="viridis"); ax.set_yticks(range(7)); ax.set_yticklabels(["T2", "T3", "T4", "T5", "T6", "T7", "CN"])
ax.set_xticks(range(24)); ax.set_xlabel("Giờ trong ngày (timestamp gốc)"); ax.grid(False); fig.colorbar(im, label="Wh/10 phút")
ax.set_title("Trung bình Appliances theo giờ × thứ (train)")
fig.tight_layout(); fig.savefig(FIG / "f33_hour_weekday.png"); plt.close(fig)

pre = hh[(hh.t >= "2016-02-01") & (hh.t < "2016-03-27")].groupby("h").Appliances.mean()
post = hh[(hh.t >= "2016-03-28") & (hh.t < "2016-05-01")].groupby("h").Appliances.mean()
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.plot(pre.index, pre.values, marker="o", ms=3, label="1/2-26/3 (trước DST)"); ax.plot(post.index, post.values, marker="o", ms=3, label="28/3-30/4 (sau DST)")
ax.set_xlabel("Giờ (timestamp gốc)"); ax.set_ylabel("Wh/10 phút"); ax.legend(); ax.set_title("Hồ sơ theo giờ trước/sau 27/3: đỉnh tối dịch 18h → 17h")
fig.tight_layout(); fig.savefig(FIG / "f34_dst_profile.png"); plt.close(fig)

top = list(R["permutation_importance_validation_mae_hour_increase"].items())[:15][::-1]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh([k for k, _ in top], [v for _, v in top]); ax.set_xlabel("Tăng MAE tổng giờ khi hoán vị feature (Wh), validation")
ax.set_title(f"Permutation importance ({POINT}, bộ {FS})")
fig.tight_layout(); fig.savefig(FIG / "f35_importance.png"); plt.close(fig)

names = list(BASELINES) + list(LEARNED)
fig, ax = plt.subplots(figsize=(9, 3))
w = .38; xs = np.arange(len(names))
ax.bar(xs - w / 2, [R["point"]["validation"][n]["mae_hour"] for n in names], w, label="validation (1-14/4)")
ax.bar(xs + w / 2, [R["point"]["test"][n]["mae_hour"] for n in names], w, label="test (1-27/5)")
ax.set_xticks(xs); ax.set_xticklabels(names); ax.set_ylabel("MAE tổng giờ (Wh)"); ax.legend(); ax.set_title("So sánh phương pháp")
fig.tight_layout(); fig.savefig(FIG / "f41_model_comparison.png"); plt.close(fig)

n3 = 3 * c.DAY
fig, ax = plt.subplots(figsize=(9, 3))
tt = times_test[:n3]
ax.fill_between(tt, lo[:n3], up[:n3], alpha=.25, label="Dải P10–P90 sau calibration"); ax.plot(tt, qt[:n3, 1], lw=.8, label="P50")
ax.plot(tt, yt[:n3], color="k", lw=.8, label="Thực tế"); ax.axhline(Ttrain, color="r", ls="--", lw=.8, label=f"Ngưỡng T={Ttrain:.0f}")
ax.set_ylabel("Tổng 60 phút tới (Wh)"); ax.legend(ncol=4, fontsize=7); ax.set_title("Ba ngày đầu của TEST (không chọn lọc)")
fig.tight_layout(); fig.savefig(FIG / "f42_test_quantiles.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.2))
for n in ("persistence_x6", "last_hour_sum", "yesterday_same_window", "ai_point_sum", "ai_calibrated_upper", "no_alert"):
    ax.plot(RATIOS, [R["alert"][n]["cost"][str(r)] for r in RATIOS], marker="o", ms=3, label=n)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Tỷ lệ chi phí bỏ sót : báo giả (FN:FP)"); ax.set_ylabel("Chi phí tương đối = FP + r·FN")
ax.legend(fontsize=7); ax.set_title("Chi phí kịch bản theo tỷ lệ FN:FP (TEST)")
fig.tight_layout(); fig.savefig(FIG / "f43_cost_curve.png"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9, 3))
lab = [s["scenario"] for s in stress]
axs[0].barh(lab, [s["mae_hour"] for s in stress]); axs[0].set_title("MAE tổng giờ (Wh)"); axs[0].invert_yaxis()
axs[1].barh(lab, [s["fn"] for s in stress], label="FN"); axs[1].barh(lab, [s["fp"] for s in stress], left=[s["fn"] for s in stress], label="FP")
axs[1].set_title("Bỏ sót (FN) + báo giả (FP)"); axs[1].legend(); axs[1].invert_yaxis(); axs[1].set_yticklabels([])
fig.tight_layout(); fig.savefig(FIG / "f44_stress.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 2.8))
ax.bar(range(24), [R["by_hour_test"][h]["mae_hour"] for h in range(24)]); ax.set_xlabel("Giờ của origin (timestamp gốc)"); ax.set_ylabel("MAE tổng giờ (Wh)")
ax.set_title("Sai số theo giờ trong ngày (TEST)")
fig.tight_layout(); fig.savefig(FIG / "f45_error_by_hour.png"); plt.close(fig)
print("done", flush=True)
