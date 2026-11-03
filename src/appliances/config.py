"""One place for the time, unit and experiment contracts."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data/raw/energydata_complete.csv"
SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
)
RAW_SHA256 = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
FREQUENCY = pd.Timedelta(minutes=10)
HORIZONS = (10, 20, 30, 40, 50, 60)
HISTORY_ROWS = 1009
LAGS = (0, 1, 2, 3, 6, 144, 1008)
SENSORS = tuple(name for i in range(1, 10) for name in (f"T{i}", f"RH_{i}")) + (
    "lights",
)
WEATHER = ("T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint")
RAW_COLUMNS = (
    ("date", "Appliances", "lights")
    + tuple(name for i in range(1, 10) for name in (f"T{i}", f"RH_{i}"))
    + WEATHER
    + ("rv1", "rv2")
)
BOUNDARIES = (
    "2016-01-11 17:00",
    "2016-04-11",
    "2016-04-25",
    "2016-05-09",
    "2016-05-27 18:10",
)
SPLIT_NAMES = ("train", "validation", "calibration", "test")
FOLDS = (
    ("2016-02-15", "2016-03-01"),
    ("2016-03-01", "2016-03-15"),
    ("2016-03-15", "2016-04-11"),
)
QUANTILES = (0.1, 0.5, 0.9)
RELEASE = ROOT / "results/appliances_20261102"
MODEL_DIR = ROOT / "artifacts/appliances_20261102"
