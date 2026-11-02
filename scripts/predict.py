"""Predict using only rows at or before the chosen issue time."""

import argparse, json
import pandas as pd
from src.appliances.data import load_data
from src.appliances.inference import Forecaster

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--issue-time", default="2016-05-09T00:00")
    a = p.parse_args()
    raw, _ = load_data()
    t = pd.Timestamp(a.issue_time)
    history = raw.loc[raw.timestamp <= t]
    if history.empty or history.timestamp.iloc[-1] != t:
        raise ValueError("Issue time must match an observed timestamp")
    print(json.dumps(Forecaster().predict(history), ensure_ascii=False, indent=2))
