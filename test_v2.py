"""Kiểm các rủi ro mới: đặc trưng, trọng số, purge và ngưỡng cảnh báo."""
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import appliances_core as v1
from v2 import core, metrics


class V2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, _ = v1.load_raw(Path(__file__).parent/'data/energydata_complete.csv')

    def test_rise_features_ignore_future(self):
        index = 5000
        changed = self.df.copy()
        changed.loc[index+1:, 'Appliances'] *= 9
        pd.testing.assert_series_equal(core.build_features(self.df).iloc[index], core.build_features(changed).iloc[index])

    def test_rise_features_finite_after_warmup(self):
        x = core.build_features(self.df)
        self.assertTrue(np.isfinite(x[core.FEATURE_SETS['F']].iloc[144:].to_numpy()).all())
        self.assertEqual(len(core.FEATURE_SETS['F']), 22)
        np.testing.assert_array_equal(x[v1.AR_FEATURES], v1.build_features(self.df)[v1.AR_FEATURES])

    def test_default_fit_delegates_to_v1(self):
        with patch.object(v1, 'fit_point', return_value='unchanged') as fit:
            model, _ = core.fit_point('hgb_l1', np.ones((20, 2)), np.ones((20, 6)))
        self.assertEqual(model, 'unchanged')
        fit.assert_called_once()

    def test_weights_come_only_from_training_labels(self):
        y = np.arange(120).reshape(20, 6)
        weights, threshold = core.training_weights(y, 4)
        self.assertEqual(threshold, np.quantile(y.sum(1), .95))
        np.testing.assert_array_equal(weights, np.where(y.sum(1) > threshold, 4, 1))
        for bad in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                core.training_weights(y, bad)

    def test_fold_thresholds_do_not_use_validation(self):
        x, y = core.build_features(self.df), v1.build_labels(self.df)
        for name, (train, validation) in v1.FOLDS.items():
            mask = v1.origin_mask(self.df.t, *train, x[core.FEATURE_SETS['A']].to_numpy(), y)
            _, threshold = core.training_weights(y[mask])
            self.assertEqual(threshold, 1780.0 if 'march' in name else 1720.0)
            self.assertTrue((pd.DatetimeIndex(self.df.t[mask]) + pd.Timedelta('60min') < pd.Timestamp(validation[0])).all())

    def test_policy_threshold_selected_from_calibration_only(self):
        y = np.array([0, 0, 0, 0, 10, 10, 10])
        score = np.arange(7, dtype=float)
        threshold = metrics.threshold_for_fpr(y, score, 5, .1)
        self.assertLessEqual(np.mean(score[y <= 5] > threshold), .1)
        threshold = metrics.threshold_for_recall(y, score, 5, .7)
        self.assertGreaterEqual(np.mean(score[y > 5] > threshold), .7)

    def test_calibration_threshold_ties_and_empty_classes(self):
        threshold = metrics.threshold_for_recall([0, 10, 10], [1, 2, 2], 5, .7)
        self.assertEqual(np.sum(np.array([2., 2.]) > threshold), 2)
        self.assertEqual(metrics.alerts([0, 10, 10], [1, 2, 2], 5, threshold)['recall'], 1.)
        with self.assertRaises(ValueError):
            metrics.threshold_for_recall([0, 0], [1, 2], 5)

    def test_offline_hour_target_matches_six_real_labels(self):
        i = 1234
        y = v1.build_labels(self.df)
        self.assertEqual(y[i].sum(), self.df.Appliances.iloc[i+1:i+7].sum())

    def test_rise_features_online_equal_batch(self):
        i = 6000
        batch = core.build_features(self.df).loc[i, core.FEATURE_SETS['F']].to_numpy(float)
        online = core.build_features(self.df.iloc[i-144:i+1].reset_index(drop=True)).iloc[-1][core.FEATURE_SETS['F']].to_numpy(float)
        np.testing.assert_allclose(batch, online, atol=1e-9, rtol=0)

    def test_lstm_input_sequence_ignores_future(self):
        from v2.lstm import inputs
        i = 5000
        changed = self.df.copy()
        changed.loc[i+1:, 'Appliances'] *= 8
        before, calendar = inputs(self.df)
        after, calendar_after = inputs(changed)
        np.testing.assert_array_equal(before[i], after[i])
        np.testing.assert_array_equal(calendar[i], calendar_after[i])

    def test_direct_forecast_is_not_replaced_by_step_sum(self):
        from v2 import runtime
        bundle = {'legacy': {'margin': 0.}, 'version': 'v2', 'config': {'target': 'direct'},
                  'policy': {'name': 'point', 'threshold': 8}}
        fake = (np.ones((1, 6)), np.array([10.]), np.array([[1., 5., 7.]]), np.array([False]))
        with patch.object(v1, 'forecast_one', return_value={}), patch.object(runtime, 'forecast_batch', return_value=fake):
            rec = runtime.forecast_one(bundle, self.df.iloc[:145])
        self.assertEqual(rec['point_sum_Wh'], 6)
        self.assertEqual(rec['hour_prediction_Wh'], 10)
        self.assertTrue(rec['alert'])

    def test_false_alarm_and_miss_counts_use_score_threshold(self):
        result = metrics.alerts([0, 10, 10, 0], [4., 8., 3., 9.], 5, 7)
        self.assertEqual([result[k] for k in ('tp', 'fp', 'fn', 'tn')], [1, 1, 1, 1])

    def test_high_group_metrics_do_not_hide_underprediction(self):
        result = metrics.regression([1., 20., 30.], [1., 10., 20.], 10)
        self.assertEqual(result['high_n'], 2)
        self.assertEqual(result['high_bias'], -10)
        self.assertEqual(result['high_underprediction_rate'], 1)

    def test_serving_rejects_invalid_history_before_prediction(self):
        from v2 import serve
        for history in (self.df.iloc[:144], self.df.iloc[:145].iloc[::-1]):
            with self.assertRaises(ValueError):
                serve.forecast_one({}, history)


if __name__ == '__main__':
    unittest.main()
