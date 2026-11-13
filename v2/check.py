"""Kiểm tests, v1, model v2, notebook và demo mà không train lại."""
import argparse
import csv
import json
import os
import re
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def main(include_lstm=False):
    environment = os.environ.copy()
    environment.pop('PYTHONOPTIMIZE', None)
    environment.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    commands = [['-m', 'unittest', 'discover', '-v'], ['tools/check_project.py'], ['-m', 'v2.verify']]
    if include_lstm:
        commands[-1].append('--include-lstm')
    unit_tests = None
    for command in commands:
        completed = subprocess.run([sys.executable, *command], cwd=ROOT, env=environment, capture_output=True, text=True, encoding='utf-8', timeout=240)
        if completed.returncode:
            raise RuntimeError(' '.join(command)+'\n'+completed.stdout[-3000:]+completed.stderr[-3000:])
        if command[:3] == ['-m', 'unittest', 'discover']:
            unit_tests = int(re.search(r'Ran (\d+) tests', completed.stderr).group(1))
        print('PASS:', ' '.join(command), flush=True)
    notebooks = sorted((ROOT/'notebooks').glob('*.ipynb'))
    assert len(notebooks) == 3
    for file in notebooks:
        notebook = json.loads(file.read_text(encoding='utf-8'))
        code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        assert code and all(cell.get('execution_count') is not None for cell in code), file.name
        assert not any(output.get('output_type') == 'error' for cell in code for output in cell.get('outputs', [])), file.name
    with TemporaryDirectory(prefix='appliances-v2-demo-') as temporary:
        completed = subprocess.run([sys.executable, '-m', 'v2.demo', '--out', temporary], cwd=ROOT, env=environment, capture_output=True, text=True, encoding='utf-8', timeout=180)
        if completed.returncode:
            raise RuntimeError(completed.stdout+completed.stderr)
        connection = sqlite3.connect(Path(temporary)/'monitor.sqlite3')
        rows = connection.execute('SELECT * FROM forecasts ORDER BY issue_time').fetchall()
        assert connection.execute('SELECT count(*) FROM decisions').fetchone()[0] == 0
        connection.close()
        saved = {row['issue_time']: row for row in csv.DictReader((ROOT/'out_v2/test_predictions.csv').open(encoding='utf-8'))}
        assert len(rows) == 14 and sum(row[-1] is not None for row in rows) == 8
        for i, row in enumerate(rows):
            target = saved[row[0]]
            assert abs(row[3]-float(target['selected_hour_prediction'])) < 1e-9
            if i < 8:
                assert abs(row[-1]-float(target['actual_hour_total'])) < 1e-9
            else:
                assert row[-1] is None
    result = {'passed': True, 'include_lstm': include_lstm, 'unit_tests': unit_tests, 'notebooks_executed': len(notebooks),
              'demo_forecasts': 14, 'actuals_available': 8, 'fabricated_acknowledgements': 0,
              'scope': 'No fitting or tuning; original v1 preserved'}
    (ROOT/'out_v2/checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-lstm', action='store_true')
    main(parser.parse_args().include_lstm)
