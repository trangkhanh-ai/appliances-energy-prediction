"""Figures from TRAIN exploration and frozen predictions, with no model fitting."""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from src.appliances.config import RELEASE, BOUNDARIES
from src.appliances.data import load_data, write_json


def draw_architecture():
    """Draw the implemented CSV-to-decision flow and the monitoring feedback."""
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set(xlim=(0, 10), ylim=(-0.1, 6.1))
    ax.axis("off")
    boxes = [
        (0.25, 4.55, "1  Dữ liệu nguồn", "CSV UCI Appliances\nSố đo M-Bus / ZigBee"),
        (3.55, 4.55, "2  Kiểm tra và xử lý", "SHA256, lưới 10 phút\nĐổi Wh sang kWh"),
        (
            6.85,
            4.55,
            "3  Tạo đặc trưng",
            "Lịch sử điện và cảm biến\n38 đặc trưng đã quan sát",
        ),
        (
            6.85,
            2.65,
            "4  Mô hình đã huấn luyện",
            "6 XGBoost dự báo điểm\n3 HGB dự báo phân vị",
        ),
        (
            3.55,
            2.65,
            "5  Dự báo và cảnh báo",
            "10–60 phút, tổng giờ\nKhoảng dự báo, ngưỡng TRAIN",
        ),
        (
            0.25,
            2.65,
            "6  Người sử dụng",
            "Xem biểu đồ và cảnh báo\nCân nhắc kiểm tra sử dụng điện",
        ),
        (
            0.25,
            0.75,
            "7  Phản hồi",
            "Đã xem / kiểm tra / bỏ qua\nGhi lý do cho quyết định",
        ),
        (
            3.55,
            0.75,
            "8  Nhật ký SQLite",
            "Lưu dự báo và phản hồi\nBổ sung số đo khi đủ 60 phút",
        ),
        (
            6.85,
            0.75,
            "9  Giám sát",
            "Theo dõi MAE và coverage\nĐề nghị đánh giá lại mô hình",
        ),
    ]
    width, height = 2.9, 1.12
    for x, y, title, detail in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.04",
                facecolor="#f0f5f8",
                edgecolor="#38677c",
            )
        )
        ax.text(
            x + width / 2,
            y + 0.83,
            title,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
        )
        ax.text(x + width / 2, y + 0.36, detail, ha="center", va="center", fontsize=11)
    for left, right in zip(boxes[:-1], boxes[1:]):
        x1, y1, *_ = left
        x2, y2, *_ = right
        if x1 == x2:
            start, end = (x1 + width / 2, y1 - 0.03), (
                x2 + width / 2,
                y2 + height + 0.03,
            )
        elif x2 > x1:
            start, end = (x1 + width + 0.04, y1 + height / 2), (
                x2 - 0.04,
                y2 + height / 2,
            )
        else:
            start, end = (x1 - 0.04, y1 + height / 2), (
                x2 + width + 0.04,
                y2 + height / 2,
            )
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=15, color="#244b60", lw=1.5
            )
        )
    # Predictions are logged immediately, even before a user submits feedback.
    ax.add_patch(
        FancyArrowPatch(
            (5, 2.61),
            (5, 1.91),
            arrowstyle="-|>",
            mutation_scale=15,
            color="#244b60",
            lw=1.5,
        )
    )
    ax.annotate(
        "Xem xét phiên bản mới khi chất lượng giảm; cần đánh giá và phê duyệt riêng",
        xy=(5, 0.21),
        ha="center",
        fontsize=11,
        color="#244b60",
    )
    return fig


def build_figures():
    raw, _ = load_data()
    train = raw.loc[raw.timestamp < pd.Timestamp(BOUNDARIES[1])].copy()
    out = RELEASE / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )

    def save(name):
        plt.tight_layout()
        plt.savefig(out / name, dpi=160, bbox_inches="tight")
        plt.close()

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(train.timestamp, train.energy_kWh, lw=0.5, color="#185b80")
    axs[0].set(ylabel="kWh / 10 min", title="Điện năng TRAIN trước ngày 11/04/2016")
    axs[1].hist(train.energy_kWh, bins=60, color="#185b80")
    axs[1].set(
        xlabel="kWh / 10 min",
        ylabel="Số bản ghi",
        title="Phân phối điện năng trên TRAIN",
    )
    save("01_train_data.png")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    train.groupby(train.timestamp.dt.hour).energy_kWh.mean().plot(
        ax=axs[0], color="#185b80", marker="o"
    )
    axs[0].set(
        xlabel="Giờ trong ngày",
        ylabel="Trung bình kWh / 10 phút",
        title="Mức tiêu thụ theo giờ trên TRAIN",
    )
    train.groupby(train.timestamp.dt.dayofweek).energy_kWh.mean().plot(
        kind="bar", ax=axs[1], color="#407c8d"
    )
    axs[1].set(
        xlabel="Thứ trong tuần (Thứ Hai = 0)",
        ylabel="Trung bình kWh / 10 phút",
        title="Mức tiêu thụ theo thứ trên TRAIN",
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
    ax.set(yticks=[], title="Chia tập theo thời gian, loại nhãn vượt ranh giới")
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
        xlabel="MAE tổng điện năng giờ tới (kWh)",
        title="So sánh TEST sau khi chọn bằng OOF và validation",
    )
    save("04_models.png")
    pred = pd.read_csv(
        RELEASE / "predictions/test.csv",
        parse_dates=["timestamp"],
        float_precision="round_trip",
    )
    sample = pred.iloc[:288]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(
        sample.timestamp,
        sample.lower_hour_kWh,
        sample.upper_hour_kWh,
        color="#afd0dd",
        alpha=0.6,
        label="Khoảng dự báo đã hiệu chỉnh",
    )
    ax.plot(
        sample.timestamp,
        sample.actual_hour_kWh,
        color="#142b40",
        lw=1,
        label="Tổng giờ đo thực tế",
    )
    ax.plot(
        sample.timestamp,
        sample.p50_hour_kWh,
        color="#c77732",
        lw=1,
        label="Trung vị P50",
    )
    ax.set(ylabel="kWh / giờ tới", title="Hai ngày TEST đầu; cửa sổ nhãn chồng lấp")
    ax.legend(fontsize=8)
    save("05_intervals.png")
    metrics = json.loads((RELEASE / "metrics.json").read_text(encoding="utf-8"))
    per = metrics["test"]["point"]["horizons"]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(
        [v["minutes"] for v in per],
        [v["mae_kWh"] for v in per],
        marker="o",
        color="#185b80",
    )
    ax.set(
        xlabel="Khoảng dự báo (phút)",
        ylabel="MAE (kWh / 10 phút)",
        title="Sai số TEST theo khoảng dự báo",
    )
    save("06_horizons.png")
    actual = pred.actual_hour_kWh.to_numpy()
    upper = pred.upper_hour_kWh.to_numpy()
    positive = actual > summary["threshold_kWh"]
    precision, recall, _ = precision_recall_curve(positive, upper)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(recall, precision, color="#185b80")
    axs[0].axhline(
        positive.mean(), ls="--", color="#b26735", label="Tỷ lệ mức tiêu thụ cao"
    )
    axs[0].set(xlabel="Recall", ylabel="Precision", title="Precision–recall trên TEST")
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
        xticklabels=["Không cảnh báo", "Cảnh báo"],
        yticks=[0, 1],
        yticklabels=["Bình thường", "Cao"],
        title="Số thời điểm phát dự báo",
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
    axs[0].bar(
        [g["group"] for g in groups],
        [g["mae_hour_kWh"] for g in groups],
        color="#185b80",
    )
    axs[0].set(ylabel="MAE tổng giờ (kWh)", title="Sai số theo nhóm trên TEST")
    axs[1].bar(
        [g["group"] for g in groups],
        [100 * g["coverage"] for g in groups],
        color="#438a8c",
    )
    axs[1].axhline(80, color="#ad6e32", ls="--")
    axs[1].set(
        ylabel="Tỷ lệ bao phủ (%)", ylim=(0, 100), title="Tỷ lệ bao phủ theo nhóm"
    )
    save("08_high_load.png")
    draw_architecture()
    save("09_architecture.png")
    variables = [c for c in train.columns if c.startswith("T") or c.startswith("RH_")]
    correlations = (
        train[variables + ["energy_kWh"]]
        .corr()["energy_kWh"]
        .drop("energy_kWh")
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(correlations.index, correlations, color="#34758b")
    ax.set(
        xlabel="Tương quan Pearson trên TRAIN",
        title="Liên hệ cảm biến; tương quan chưa xác nhận nhân quả",
    )
    save("10_sensors.png")
    write_json(
        RELEASE / "train_statistics.json",
        {
            "energy_kWh": train.energy_kWh.describe(
                percentiles=[0.05, 0.5, 0.95, 0.99]
            ).to_dict(),
            "sensor_correlations": correlations.to_dict(),
            "raw_train_rows": len(train),
        },
    )
    print("Saved figures and conditional error tables")


if __name__ == "__main__":
    build_figures()
