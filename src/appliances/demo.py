"""Local replay Data -> AI -> Human decision, stored in SQLite."""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from .config import MODEL_DIR, BOUNDARIES
from .data import load_data
from .inference import Forecaster


class Replay:
    def __init__(self, database, model_dir=MODEL_DIR):
        self.raw, _ = load_data()
        self.model = Forecaster(model_dir)
        self.reference = json.loads(
            (Path(model_dir) / "monitor_reference.json").read_text(encoding="utf-8")
        )
        self.first = int(
            self.raw.index[self.raw.timestamp >= pd.Timestamp(BOUNDARIES[3])][0]
        )
        self.last = len(self.raw) - 7
        self.position = self.first
        self.session = uuid.uuid4().hex
        self.lock = threading.RLock()
        Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(database), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """CREATE TABLE IF NOT EXISTS forecasts(
            session TEXT, issue_time TEXT, forecast_end TEXT, model_sha256 TEXT,
            predicted REAL, lower REAL, upper REAL, actual REAL, payload TEXT,
            PRIMARY KEY(session,issue_time));
            CREATE TABLE IF NOT EXISTS decisions(
            token TEXT PRIMARY KEY, session TEXT, issue_time TEXT, operator TEXT, action TEXT, note TEXT);"""
        )
        self.latest = None
        self.timings = []

    def _forecast(self):
        history = self.raw.iloc[: self.position + 1]
        payload = self.model.predict(history)
        self.db.execute(
            "INSERT OR IGNORE INTO forecasts VALUES(?,?,?,?,?,?,?,?,?)",
            (
                self.session,
                payload["issue_time"],
                payload["forecast_end"],
                payload["model_sha256"],
                payload["predicted_hour_kWh"],
                payload["lower_hour_kWh"],
                payload["upper_hour_kWh"],
                None,
                json.dumps(payload),
            ),
        )
        now = history.timestamp.iloc[-1]
        pending = self.db.execute(
            "SELECT issue_time,forecast_end FROM forecasts WHERE session=? AND actual IS NULL AND forecast_end<=?",
            (self.session, now.isoformat()),
        ).fetchall()
        observed = history.set_index("timestamp").energy_kWh
        for issue, end in pending:
            values = observed.loc[
                (observed.index > pd.Timestamp(issue))
                & (observed.index <= pd.Timestamp(end))
            ]
            if len(values) == 6:
                self.db.execute(
                    "UPDATE forecasts SET actual=? WHERE session=? AND issue_time=?",
                    (float(values.sum()), self.session, issue),
                )
        self.db.commit()
        self.latest = payload

    def state(self):
        if self.latest is None:
            self._forecast()
        recent = self.db.execute(
            "SELECT issue_time,predicted,lower,upper,actual FROM forecasts WHERE session=? AND actual IS NOT NULL ORDER BY issue_time",
            (self.session,),
        ).fetchall()
        monitoring = {
            "status": "insufficient_history",
            "matured_forecasts": len(recent),
            "required_for_review": 288,
        }
        if recent:
            values = np.asarray([[r[1], r[2], r[3], r[4]] for r in recent], dtype=float)
            error = abs(values[:, 0] - values[:, 3])
            coverage = (values[:, 3] >= values[:, 1]) & (values[:, 3] <= values[:, 2])
            monitoring.update(
                mae_hour_kWh=float(error.mean()), coverage=float(coverage.mean())
            )
            if len(recent) >= 288:
                errors = [float(error[-288:-144].mean()), float(error[-144:].mean())]
                cov = [float(coverage[-288:-144].mean()), float(coverage[-144:].mean())]
                bad_error = all(
                    v > 1.5 * self.reference["calibration_mae_kWh"] for v in errors
                )
                bad_cov = all(
                    v < max(0.0, self.reference["calibration_coverage"] - 0.1)
                    for v in cov
                )
                monitoring.update(
                    status="review" if bad_error or bad_cov else "within_reference",
                    daily_mae=errors,
                    daily_coverage=cov,
                )
        history = self.raw.iloc[max(0, self.position - 143) : self.position + 1]
        decisions = self.db.execute(
            "SELECT issue_time,operator,action,note FROM decisions WHERE session=? ORDER BY rowid DESC LIMIT 10",
            (self.session,),
        ).fetchall()
        return {
            "as_of": str(self.raw.timestamp.iloc[self.position]),
            "forecast": self.latest,
            "history": [
                {"time": str(r.timestamp), "energy_kWh": float(r.energy_kWh)}
                for r in history.itertuples()
            ],
            "monitoring": monitoring,
            "decisions": [
                dict(zip(("issue_time", "operator", "action", "note"), r))
                for r in decisions
            ],
            "timing_p95_ms": (
                float(np.quantile(self.timings, 0.95)) if self.timings else None
            ),
            "at_end": self.position >= self.last,
            "mode": "Replay of real 2016 measurements, one step = 10 minutes",
        }

    def dispatch(self, action="state", decision=None):
        with self.lock:
            started = time.perf_counter()
            if action == "step":
                if self.position < self.last:
                    self.position += 1
                self._forecast()
            elif action == "reset":
                self.position = self.first
                self.session = uuid.uuid4().hex
                self.latest = None
                self._forecast()
            elif action == "decision":
                if self.latest is None:
                    self._forecast()
                d = decision or {}
                if d.get("action") not in ("acknowledge", "inspect", "dismiss"):
                    raise ValueError("Unknown decision action")
                if (
                    not isinstance(d.get("operator"), str)
                    or not d["operator"].strip()
                    or len(d["operator"]) > 100
                ):
                    raise ValueError(
                        "Operator name is required, maximum 100 characters"
                    )
                if (
                    not isinstance(d.get("token"), str)
                    or not d["token"]
                    or len(d["token"]) > 100
                ):
                    raise ValueError("A request token is required")
                note = d.get("note", "")
                if not isinstance(note, str) or len(note) > 1000:
                    raise ValueError("Note too long")
                if d.get("issue_time") != self.latest["issue_time"]:
                    raise ValueError("Stale forecast; review current prediction")
                row = (
                    d["token"],
                    self.session,
                    d["issue_time"],
                    d["operator"].strip(),
                    d["action"],
                    note,
                )
                old = self.db.execute(
                    "SELECT * FROM decisions WHERE token=?", (d["token"],)
                ).fetchone()
                if old is not None and tuple(old) != row:
                    raise ValueError(
                        "Request token already used for a different decision"
                    )
                self.db.execute(
                    "INSERT OR IGNORE INTO decisions VALUES(?,?,?,?,?,?)", row
                )
                self.db.commit()
            elif action != "state":
                raise ValueError("Unknown replay action")
            result = self.state()
            elapsed = (time.perf_counter() - started) * 1000
            self.timings.append(elapsed)
            result["dispatch_ms"] = elapsed
            return result
