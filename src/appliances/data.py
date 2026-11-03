"""Read measured data, preserve its bytes, and fail visibly on invalid input."""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .config import RAW_FILE, RAW_COLUMNS, RAW_SHA256, SOURCE_URL, FREQUENCY


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def download_data(path=RAW_FILE):
    path = Path(path)
    if path.exists():
        if sha256(path) != RAW_SHA256:
            raise ValueError("Existing raw CSV has a different SHA256")
        return path
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    name = next(
        name for name in archive.namelist() if name.endswith("energydata_complete.csv")
    )
    content = archive.read(name)
    if hashlib.sha256(content).hexdigest() != RAW_SHA256:
        raise ValueError("UCI download differs from the audited source")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def validate_raw(frame):
    if tuple(frame.columns) != RAW_COLUMNS or frame.empty:
        raise ValueError("Unexpected Appliances columns or empty CSV")
    times = pd.to_datetime(frame.date, format="%Y-%m-%d %H:%M:%S", errors="raise")
    numeric = frame.drop(columns="date").apply(pd.to_numeric, errors="raise")
    if times.isna().any() or times.duplicated().any() or frame.isna().any().any():
        raise ValueError("Missing or duplicate source values")
    if (
        not times.diff().dropna().eq(FREQUENCY).all()
        or not times.eq(times.dt.floor("10min")).all()
    ):
        raise ValueError("Source must be an ordered continuous 10-minute grid")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Non-finite source values")
    if (numeric[["Appliances", "lights"]] < 0).any().any():
        raise ValueError("Negative energy measurement")
    humidity = [c for c in numeric if c.startswith("RH_")]
    if not numeric[humidity].apply(lambda c: c.between(0, 100)).all().all():
        raise ValueError("Humidity outside [0,100] percent")
    clean = numeric.copy()
    clean.insert(0, "timestamp", times)
    clean["energy_kWh"] = clean.Appliances / 1000.0
    return clean


def load_data(path=RAW_FILE):
    if sha256(path) != RAW_SHA256:
        raise ValueError("Raw checksum mismatch")
    original = pd.read_csv(path)
    clean = validate_raw(original)
    if len(clean) != 19735:
        raise ValueError("Unexpected official row count")
    audit = {
        "source_url": SOURCE_URL,
        "sha256": sha256(path),
        "rows": len(clean),
        "columns": len(original.columns),
        "start": str(clean.timestamp.min()),
        "end": str(clean.timestamp.max()),
        "missing_cells": int(original.isna().sum().sum()),
        "duplicates": int(clean.timestamp.duplicated().sum()),
        "frequency_minutes": 10,
        "zero_energy_rows": int(clean.energy_kWh.eq(0).sum()),
        "negative_energy_rows": 0,
        "unit_conversion": "energy_kWh = Appliances_Wh / 1000",
        "timestamp_assumption": "Declared timestamp orders completed measurements; timezone/interval-end semantics not independently certified",
        "weather_excluded": "Airport series interpolated upstream; real-time availability not established",
    }
    return clean, audit
