"""Figures from TRAIN exploration and frozen predictions, with no model fitting."""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from src.appliances.config import RELEASE, ROOT, BOUNDARIES, HORIZONS
from src.appliances.data import load_data, write_json


def build_figures():
    raw, _ = load_data()
    train = raw.loc[raw.timestamp < pd.Timestamp(BOUNDARIES[1])].copy()
    out = RELEASE / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120}
    )

    def save(name):
        plt.tight_layout()
        plt.savefig(out / name, dpi=160, bbox_inches="tight")
        plt.close()

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(train.timestamp, train.energy_kWh, lw=0.5, color="#185b80")
    axs[0].set(ylabel="kWh / 10 min", title="TRAIN measurements before 11 April 2016")
    axs[1].hist(train.energy_kWh, bins=60, color="#185b80")
    axs[1].set(xlabel="kWh / 10 min", ylabel="Records", title="TRAIN energy distribution")
    save("01_train_data.png")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    train.groupby(train.timestamp.dt.hour).energy_kWh.mean().plot(
        ax=axs[0], color="#185b80", marker="o"
    )
    axs[0].set(xlabel="Hour", ylabel="Mean kWh / 10 min", title="TRAIN daily pattern")
    train.groupby(train.timestamp.dt.dayofweek).energy_kWh.mean().plot(
        kind="bar", ax=axs[1], color="#407c8d"
    )
    axs[1].set(
        xlabel="Weekday (Monday = 0)", ylabel="Mean kWh / 10 min", title="TRAIN weekday pattern"
    )
    save("02_calendar.png")
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ["#1a5e82", "#438a8c", "#b59143", "#9d516a"]
    for i, name in enumerate(("TRAIN", "Validation", "Calibration", "TEST")):
        ax.barh(
            0,
            pd.Timestamp(BOUNDARIES[i + 1]) - pd.Timestamp(BOUNDARIES[i]),
            left=pd.Timestamp(BOUNDARIES[i]),
            color=colors[i],
            label=name,
        )
    ax.xaxis_date()
    ax.set(yticks=[], title="Chronological split with purged future labels")
    ax.legend(ncol=4)
    save("03_split.png")
    comparison = pd.read_csv(RELEASE / "comparison.csv")
    summary = json.loads((RELEASE / "summary.json").read_text(encoding="utf-8"))
    c = comparison[comparison.split.eq("test")].sort_values("mae_hour_kWh")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        c.model,
        c.mae_hour_kWh,
        color=["#d17b36" if m == summary["winner"] else "#2b6b8d" for m in c.model],
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="MAE next-hour total (kWh)",
        title="TEST comparison; selection was locked on OOF and validation",
    )
    save("04_models.png")
    pred = pd.read_csv(
        RELEASE / "predictions/test.csv", parse_dates=["timestamp"], float_precision="round_trip"
    )
    sample = pred.iloc[:288]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(
        sample.timestamp,
        sample.lower_hour_kWh,
        sample.upper_hour_kWh,
        color="#afd0dd",
        alpha=0.6,
        label="Calibrated interval",
    )
    ax.plot(
        sample.timestamp,
        sample.actual_hour_kWh,
        color="#142b40",
        lw=1,
        label="Measured next-hour total",
    )
    ax.plot(sample.timestamp, sample.p50_hour_kWh, color="#c77732", lw=1, label="P50 median")
    ax.set(ylabel="kWh / next hour", title="First two TEST days, overlapping forecast origins")
    ax.legend(fontsize=8)
    save("05_intervals.png")
    metrics = json.loads((RELEASE / "metrics.json").read_text(encoding="utf-8"))
    per = metrics["test"]["point"]["horizons"]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot([v["minutes"] for v in per], [v["mae_kWh"] for v in per], marker="o", color="#185b80")
    ax.set(
        xlabel="Forecast horizon (minutes)",
        ylabel="MAE (kWh / 10 min)",
        title="TEST error by horizon",
    )
    save("06_horizons.png")
    actual = pred.actual_hour_kWh.to_numpy()
    upper = pred.upper_hour_kWh.to_numpy()
    positive = actual > summary["threshold_kWh"]
    precision, recall, _ = precision_recall_curve(positive, upper)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(recall, precision, color="#185b80")
    axs[0].axhline(positive.mean(), ls="--", color="#b26735", label="Positive prevalence")
    axs[0].set(xlabel="Recall", ylabel="Precision", title="TEST precision-recall")
    axs[0].legend()
    a = metrics["test"]["alert"]
    matrix = np.array([[a["tn"], a["fp"]], [a["fn"], a["tp"]]])
    axs[1].imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        axs[1].text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            color="white" if value > matrix.max() / 2 else "black",
        )
    axs[1].set(
        xticks=[0, 1],
        xticklabels=["No warning", "Warning"],
        yticks=[0, 1],
        yticklabels=["Normal", "High"],
        title="Forecast-origin counts",
    )
    save("07_alerts.png")
    pred["hour"] = pred.timestamp.dt.hour
    pred["absolute_error_kWh"] = abs(pred.predicted_hour_kWh - pred.actual_hour_kWh)
    threshold = summary["threshold_kWh"]
    groups = []
    for name, mask in [
        ("All", np.ones(len(pred), dtype=bool)),
        ("Normal", ~positive),
        ("High", positive),
    ]:
        frame = pred.loc[mask]
        groups.append(
            {
                "group": name,
                "rows": len(frame),
                "mae_hour_kWh": float(frame.absolute_error_kWh.mean()),
                "coverage": float(
                    (
                        (frame.actual_hour_kWh >= frame.lower_hour_kWh)
                        & (frame.actual_hour_kWh <= frame.upper_hour_kWh)
                    ).mean()
                ),
            }
        )
    pd.DataFrame(groups).to_csv(RELEASE / "conditional_errors.csv", index=False)
    pred.groupby("hour").absolute_error_kWh.agg(["count", "mean"]).to_csv(
        RELEASE / "errors_by_hour.csv"
    )
    pred.sort_values("absolute_error_kWh", ascending=False).head(12).to_csv(
        RELEASE / "largest_errors.csv", index=False
    )
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar([g["group"] for g in groups], [g["mae_hour_kWh"] for g in groups], color="#185b80")
    axs[0].set(ylabel="MAE total (kWh)", title="Conditional TEST error")
    axs[1].bar([g["group"] for g in groups], [100 * g["coverage"] for g in groups], color="#438a8c")
    axs[1].axhline(80, color="#ad6e32", ls="--")
    axs[1].set(ylabel="Coverage (%)", ylim=(0, 100), title="Conditional interval coverage")
    save("08_high_load.png")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    boxes = [
        ("CSV\nM-Bus / ZigBee", 0.01),
        ("Validate\nCausal features", 0.19),
        ("6-step AI\nQuantiles", 0.37),
        ("Dashboard\nHuman review", 0.55),
        ("SQLite logs\nMonitoring", 0.76),
    ]
    for text, x in boxes:
        ax.text(
            x,
            0.5,
            text,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=.6", facecolor="#e5eff5", edgecolor="#39728b"),
            va="center",
            fontsize=10,
        )
    for x in [0.155, 0.335, 0.515, 0.725]:
        ax.annotate(
            "",
            xy=(x + 0.028, 0.5),
            xytext=(x, 0.5),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->"),
        )
    ax.set_title("Data -> AI -> Decision -> Feedback", pad=20)
    save("09_architecture.png")
    variables = [c for c in train.columns if c.startswith("T") or c.startswith("RH_")]
    correlations = (
        train[variables + ["energy_kWh"]].corr()["energy_kWh"].drop("energy_kWh").sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(correlations.index, correlations, color="#34758b")
    ax.set(
        xlabel="Pearson correlation on TRAIN",
        title="Sensor association; correlation does not establish causation",
    )
    save("10_sensors.png")
    write_json(
        RELEASE / "train_statistics.json",
        {
            "energy_kWh": train.energy_kWh.describe(percentiles=[0.05, 0.5, 0.95, 0.99]).to_dict(),
            "sensor_correlations": correlations.to_dict(),
            "raw_train_rows": len(train),
        },
    )
    print("Saved figures and conditional error tables")


if __name__ == "__main__":
    build_figures()
