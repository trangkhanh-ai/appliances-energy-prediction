"""Kiểm tra các đầu vào phải bị từ chối trước khi dự báo."""
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import appliances_core as c

DF, _ = c.load_raw(Path(__file__).parent / 'data/energydata_complete.csv')

class InputContract(unittest.TestCase):
    def reject(self, history):
        with self.assertRaises(ValueError):
            c.forecast_one({'features': c.AR_FEATURES}, history)

    def test_short_history(self): self.reject(DF.iloc[:144])
    def test_duplicate_time(self):
        d = DF.iloc[:145].copy(); d.loc[144, 't'] = d.t.iloc[143]; self.reject(d)
    def test_gap(self):
        d = DF.iloc[:145].copy(); d.loc[144, 't'] += c.STEP; self.reject(d)
    def test_reverse_time(self): self.reject(DF.iloc[:145].iloc[::-1])
    def test_negative_energy(self):
        d = DF.iloc[:145].copy(); d.loc[0, 'Appliances'] = -1; self.reject(d)
    def test_missing_energy(self):
        d = DF.iloc[:145].astype({'Appliances':float}).copy(); d.loc[0, 'Appliances'] = np.nan; self.reject(d)
    def test_infinite_energy(self):
        d = DF.iloc[:145].astype({'Appliances':float}).copy(); d.loc[0, 'Appliances'] = np.inf; self.reject(d)
    def test_partial_window(self):
        self.assertTrue(np.isnan(c._win(np.arange(3), 6, 'sum')).all())
    def test_no_forecast_dashboard(self):
        import demo_replay as demo
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'dashboard.html'
            demo.render_html(p, [], {'rec':None}, None)
            self.assertIn('KHÔNG PHÁT DỰ BÁO', p.read_text(encoding='utf-8'))

if __name__ == '__main__': unittest.main()
