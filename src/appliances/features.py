"""The same feature function serves training, verification and the live demo."""

import numpy as np
import pandas as pd
from .config import (
    FREQUENCY,
    LAGS,
    SENSORS,
    HORIZONS,
    HISTORY_ROWS,
    BOUNDARIES,
    SPLIT_NAMES,
)


def make_features(history):
    times = pd.to_datetime(history.timestamp, errors="raise")
    energy = history.energy_kWh.astype(float)
    if (
        len(times) == 0
        or times.isna().any()
        or not times.diff().dropna().eq(FREQUENCY).all()
        or not times.eq(times.dt.floor("10min")).all()
    ):
        raise ValueError("History must be consecutive and chronological")
    if not np.isfinite(energy).all() or (energy < 0).any():
        raise ValueError("Energy must be finite and nonnegative")
    x = pd.DataFrame(index=history.index)
    for lag in LAGS:
        x[f"energy_lag_{lag}"] = energy.shift(lag)
    for name, window, operation in [
        ("sum_1h", 6, "sum"),
        ("mean_4h", 24, "mean"),
        ("std_4h", 24, "std"),
        ("mean_24h", 144, "mean"),
        ("max_24h", 144, "max"),
    ]:
        values = np.full(len(energy), np.nan)
        if len(energy) >= window:
            windows = np.lib.stride_tricks.sliding_window_view(
                energy.to_numpy(), window
            )
            values[window - 1 :] = getattr(windows, operation)(axis=1)
        x[name] = values
    x["change_10m"] = energy.diff(1)
    x["change_1h"] = energy.diff(6)
    start = times + FREQUENCY
    hour = start.dt.hour + start.dt.minute / 60
    weekday = start.dt.dayofweek
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    x["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    x["is_weekend"] = (weekday >= 5).astype(int)
    for column in SENSORS:
        if column in history:
            value = history[column].astype(float)
            if column == "lights":
                value = value / 1000.0
            x[f"{column}_lag_1"] = value.shift(1)
    return x


def feature_columns(frame, group):
    if group not in ("G0", "G1"):
        raise ValueError("Feature group must be G0 or G1")
    return [
        c
        for c in frame.columns
        if group == "G1" or not c.endswith("_lag_1") or c == "energy_lag_1"
    ]


def supervised(raw):
    x = make_features(raw)
    labels = pd.DataFrame(
        {
            f"actual_{m}m_kWh": raw.energy_kWh.shift(-i)
            for i, m in enumerate(HORIZONS, 1)
        }
    )
    rows = pd.concat(
        [raw[["timestamp"]].reset_index(drop=True), x.reset_index(drop=True), labels],
        axis=1,
    )
    rows["forecast_end"] = rows.timestamp + pd.Timedelta(minutes=60)
    rows["actual_hour_kWh"] = labels.sum(axis=1, min_count=6)
    ready = x.notna().all(axis=1) & labels.notna().all(axis=1)
    rows["ready"] = ready
    return rows, x.columns.tolist()


def split_rows(rows):
    parts = {}
    counts = {}
    for name, start, end in zip(SPLIT_NAMES, BOUNDARIES[:-1], BOUNDARIES[1:]):
        inside = rows.timestamp.between(
            pd.Timestamp(start), pd.Timestamp(end), inclusive="left"
        )
        safe = rows.forecast_end < pd.Timestamp(end)
        part = rows.loc[inside & safe & rows.ready].copy().reset_index(drop=True)
        if part.empty:
            raise ValueError(f"Empty split {name}")
        parts[name] = part
        counts[name] = len(part)
    return parts, counts


def features_at_issue(history, columns):
    if len(history) < HISTORY_ROWS:
        raise ValueError(f"Need {HISTORY_ROWS} consecutive measurements")
    observed = history.tail(HISTORY_ROWS).reset_index(drop=True)
    row = make_features(observed).tail(1)
    if not set(columns).issubset(row.columns):
        raise ValueError("Missing required sensors")
    row = row[columns]
    if not np.isfinite(row.to_numpy()).all():
        raise ValueError("Invalid feature values")
    return row
