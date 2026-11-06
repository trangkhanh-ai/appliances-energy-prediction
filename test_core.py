"""Kiểm thử tối thiểu cho các rủi ro lớn nhất: rò rỉ tương lai, split/purge, nhất quán batch-online.
Chạy:  python -m unittest test_core -v     (đặt cạnh appliances_core.py và data/energydata_complete.csv)
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import appliances_core as c

CSV = Path(__file__).parent / "data" / "energydata_complete.csv"
DF, AUDIT = c.load_raw(CSV)


class LeakageAndSplit(unittest.TestCase):
    def test_features_ignore_future(self):
        """Sửa mọi số đo sau thời điểm i -> feature tại i không đổi."""
        i = 5000
        base = c.build_features(DF).iloc[i]
        d2 = DF.copy()
        d2.loc[i + 1:, ["Appliances", "lights"] + c.WEATHER + c.HOUSE] *= 7.0
        pd.testing.assert_series_equal(base, c.build_features(d2).iloc[i])

    def test_labels_are_future_values(self):
        Y = c.build_labels(DF)
        i = 1234
        np.testing.assert_array_equal(Y[i], DF.Appliances.to_numpy()[i + 1:i + 7])

    def test_splits_disjoint_and_purged(self):
        X, Y = c.build_features(DF), c.build_labels(DF)
        t = pd.DatetimeIndex(DF.t)
        seen = np.zeros(len(DF), bool)
        for name, (s, e) in c.SPLITS.items():
            m = c.origin_mask(DF.t, s, e, X[c.FEATURE_SETS["D_plus_house"]].to_numpy(), Y)
            self.assertFalse((seen & m).any(), name)
            self.assertTrue((t[m] + 6 * c.STEP < pd.Timestamp(e)).all(), name)   # nhãn không vượt biên
            seen |= m

    def test_online_equals_batch(self):
        """forecast_one trên 145 hàng cuối cho cùng feature với tính batch toàn chuỗi."""
        X = c.build_features(DF)[c.FEATURE_SETS["C_plus_weather"]]
        i = 8000
        tail = c.build_features(DF.iloc[i - c.DAY: i + 1].reset_index(drop=True)).iloc[-1][c.FEATURE_SETS["C_plus_weather"]]
        np.testing.assert_allclose(tail.to_numpy(float), X.iloc[i].to_numpy(float), rtol=0, atol=1e-9)

    def test_data_contract(self):
        self.assertEqual((AUDIT["rows"], AUDIT["missing_cells"], AUDIT["grid_gaps"]), (19735, 0, 0))

    def test_ridge_scratch_matches_sklearn(self):
        """RidgeScratch (NumPy) khớp StandardScaler+Ridge của scikit-learn."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from ridge_scratch import RidgeScratch
        rng = np.random.default_rng(0)
        X, Y = rng.normal(size=(300, 5)), rng.normal(size=(300, 3))
        X[:, 2] = 1.0
        for a in (1.0, 100.0):
            ref = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X, Y).predict(X)
            np.testing.assert_allclose(RidgeScratch(a).fit(X, Y).predict(X), ref, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
