import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.appliances.config import RELEASE, HORIZONS
from src.appliances.data import load_data, validate_raw
from src.appliances.features import (
    supervised,
    split_rows,
    make_features,
    features_at_issue,
    feature_columns,
)
from src.appliances.inference import Forecaster
from src.appliances.demo import Replay


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.audit = load_data()
        cls.rows, cls.columns = supervised(cls.raw)
        cls.parts, cls.counts = split_rows(cls.rows)
        cls.model = Forecaster()

    def test_source_and_units(self):
        self.assertEqual(len(self.raw), 19735)
        np.testing.assert_array_equal(self.raw.energy_kWh, self.raw.Appliances / 1000)

    def test_split_purge(self):
        for left, right in zip(list(self.parts.values())[:-1], list(self.parts.values())[1:]):
            self.assertLess(left.forecast_end.max(), right.timestamp.min())
        self.assertEqual(sum(self.counts.values()), 19735 - 1008 - 6 - 18)

    def test_six_measured_labels(self):
        row = self.rows.iloc[2000]
        np.testing.assert_allclose(
            row[[f"actual_{m}m_kWh" for m in HORIZONS]].to_numpy(dtype=float),
            self.raw.energy_kWh.iloc[2001:2007].to_numpy(),
        )

    def test_future_change_cannot_change_inputs(self):
        altered = self.raw.copy()
        altered.loc[2001, "energy_kWh"] += 1
        np.testing.assert_array_equal(
            make_features(self.raw).iloc[2000], make_features(altered).iloc[2000]
        )

    def test_groups_exclude_weather_random(self):
        g0 = feature_columns(self.rows[self.columns], "G0")
        self.assertEqual(len(g0), 19)
        self.assertEqual(len(self.columns), 38)
        self.assertFalse(any("rv" in c or "T_out" in c for c in self.columns))
        self.assertIn("T1_lag_1", self.columns)
        self.assertNotIn("T1_lag_1", g0)

    def test_batch_matches_online(self):
        history = self.raw.iloc[:2001]
        np.testing.assert_allclose(
            features_at_issue(history, self.model.bundle["features"]),
            make_features(self.raw).iloc[[2000]][self.model.bundle["features"]],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_insufficient_history_rejected(self):
        with self.assertRaises(ValueError):
            self.model.predict(self.raw.iloc[:1008])

    def test_gap_and_duplicate_rejected(self):
        for frame in [
            self.raw.iloc[:2001].drop(index=1500),
            pd.concat([self.raw.iloc[:1500], self.raw.iloc[1499:2000]]),
        ]:
            with self.assertRaises(ValueError):
                self.model.predict(frame)

    def test_bad_energy_rejected(self):
        for value in [np.nan, np.inf, -1]:
            history = self.raw.iloc[:2001].copy()
            history.loc[2000, "energy_kWh"] = value
            with self.assertRaises(ValueError):
                self.model.predict(history)

    def test_no_actual_at_issue(self):
        result = self.model.predict(self.raw.iloc[:2001])
        self.assertIsNone(result["actual_hour_kWh"])
        self.assertEqual(len(result["horizons"]), 6)
        self.assertLessEqual(result["lower_hour_kWh"], result["upper_hour_kWh"])

    def test_calibration_interval_csv_round_trip(self):
        from src.appliances.metrics import interval_metrics

        frame = pd.read_csv(RELEASE / "predictions/calibration.csv", float_precision="round_trip")
        labels = frame[[f"actual_{m}m_kWh" for m in HORIZONS]].to_numpy()
        quantiles = frame[["p10_hour_kWh", "p50_hour_kWh", "p90_hour_kWh"]].to_numpy()
        metric = interval_metrics(
            labels.sum(axis=1),
            quantiles,
            self.model.bundle["margin"],
            self.model.bundle["threshold"],
        )
        stored = json.loads((RELEASE / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metric["coverage"], stored["calibration"]["interval"]["coverage"])

    def test_replay_labels_mature_only_after_six_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            demo = Replay(folder + "/session.sqlite3")
            try:
                first = demo.dispatch()
                for _ in range(5):
                    demo.dispatch("step")
                self.assertEqual(demo.state()["monitoring"]["matured_forecasts"], 0)
                state = demo.dispatch("step")
                self.assertEqual(state["monitoring"]["matured_forecasts"], 1)
                self.assertTrue(
                    all(
                        pd.Timestamp(p["time"]) <= pd.Timestamp(state["as_of"])
                        for p in state["history"]
                    )
                )
                d = {
                    "token": "test-once",
                    "issue_time": state["forecast"]["issue_time"],
                    "operator": "Test",
                    "action": "inspect",
                    "note": "automated functional test",
                }
                demo.dispatch("decision", d)
                demo.dispatch("decision", d)
                self.assertEqual(len(demo.state()["decisions"]), 1)
                changed = {**d, "note": "different"}
                with self.assertRaises(ValueError):
                    demo.dispatch("decision", changed)
                demo.dispatch("reset")
                self.assertEqual(demo.state()["monitoring"]["matured_forecasts"], 0)
                self.assertEqual(demo.state()["decisions"], [])
            finally:
                demo.db.close()


if __name__ == "__main__":
    unittest.main()
