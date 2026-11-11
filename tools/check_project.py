"""Kiểm bản đã chốt trong thư mục tạm, không ghi đè model/kết quả gốc.

Chạy tại thư mục repo: python tools/check_project.py
Lưu log riêng nếu cần: python tools/check_project.py --out runs/check-20261111
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "appliances_core.py", "ridge_scratch.py", "run_all.py", "verify_results.py",
    "demo_replay.py", "monitor_check.py", "draw_architecture.py",
    "test_core.py", "test_contract.py",
)


def check_frozen_files(root):
    """Kiểm mã băm trước khi tiến trình kiểm chứng nạp model joblib."""
    manifest = json.loads((root / "out/verification.json").read_text(encoding="utf-8"))
    if manifest.get("passed") is not True:
        raise ValueError("Phiên kết quả gốc chưa có verification passed=true")
    if set(manifest["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("Danh sách mã nguồn khác phiên đã chốt; cần kiểm chứng phiên mới")
    expected = {
        **manifest["source_sha256"],
        "data/energydata_complete.csv": manifest["data_sha256"],
        "out/bundle.joblib": manifest["model_sha256"],
    }
    for name, digest in expected.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"SHA-256 không khớp: {name}")
    return manifest


def run_step(folder, name, arguments, logs, environment):
    """Dùng đúng Python hiện tại; lỗi ở bước nào thì dừng tại bước đó."""
    result = subprocess.run(
        [sys.executable, *arguments], cwd=folder, env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    output = result.stdout + result.stderr
    if logs is not None:
        (logs / f"{name}.log").write_text(output, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{name} thất bại (exit {result.returncode}):\n{output[-4000:]}")
    print(f"OK: {name}")
    return output


def check_demo(folder):
    """Đối chiếu demo với CSV đã chốt và kiểm thời điểm được phép có actual."""
    with (folder / "out/test_predictions.csv").open(encoding="utf-8", newline="") as stream:
        saved = {row["issue_time"]: row for row in csv.DictReader(stream)}
    with closing(sqlite3.connect(folder / "replay/monitor.sqlite3")) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM forecasts ORDER BY issue_time").fetchall()
        decisions = connection.execute("SELECT count(*) FROM decisions").fetchone()[0]
    if len(rows) != 14 or decisions != 0:
        raise ValueError("Demo phải có 14 dự báo và không tự tạo xác nhận người dùng")
    last_time = datetime.fromisoformat(rows[-1]["issue_time"])
    fields = {
        "point_sum_Wh": "point_hour_total", "p10": "p10", "p50": "p50", "p90": "p90",
        "lower_cal": "lower_cal", "upper_cal": "upper_cal",
    }
    actual_count = 0
    for row in rows:
        reference = saved[row["issue_time"]]
        for database_field, csv_field in fields.items():
            if not math.isclose(row[database_field], float(reference[csv_field]), rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Demo khác dự báo đã chốt tại {row['issue_time']}: {database_field}")
        expected_status = "ALERT" if row["upper_cal"] > row["threshold"] else "NORMAL"
        if row["status"] != expected_status:
            raise ValueError("Trạng thái cảnh báo không khớp cận trên và ngưỡng")
        available = datetime.fromisoformat(row["issue_time"]) + timedelta(minutes=60) <= last_time
        if available:
            actual = row["actual_hour_total"]
            if actual is None or not math.isclose(actual, float(reference["actual_hour_total"]), rel_tol=0, abs_tol=1e-9):
                raise ValueError("Actual đã đến phải khớp dữ liệu kiểm chứng")
            actual_count += 1
        elif row["actual_hour_total"] is not None:
            raise ValueError("Demo lộ actual tương lai trước đủ 60 phút")
    html = (folder / "replay/dashboard.html").read_text(encoding="utf-8")
    if "Chưa có phản hồi CLI" not in html:
        raise ValueError("Giao diện phải phản ánh đúng phiên chưa có xác nhận người dùng")
    return {"forecasts": len(rows), "actuals_available": actual_count, "decisions": decisions}


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Thư mục mới để lưu log và summary.json")
    args = parser.parse_args()
    logs = args.out.resolve() if args.out is not None else None
    try:
        if logs is not None:
            logs.mkdir(parents=True, exist_ok=False)
        original = check_frozen_files(ROOT)
        print("OK: mã băm của dữ liệu, model và 9 tệp mã nguồn")
        environment = os.environ.copy()
        environment.pop("PYTHONOPTIMIZE", None)
        environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                           OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        with TemporaryDirectory(prefix="appliances-check-") as temporary:
            folder = Path(temporary)
            files = [*SOURCE_FILES, "data/energydata_complete.csv", "out/bundle.joblib",
                     "out/results.json", "out/test_predictions.csv"]
            for name in files:
                destination = folder / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
            tests = run_step(folder, "unit-tests", ["-m", "unittest", "discover", "-v"], logs, environment)
            run_step(folder, "verify-results", ["verify_results.py"], logs, environment)
            run_step(folder, "demo-replay", ["demo_replay.py", "--start", "2016-05-26 15:30",
                     "--steps", "14", "--out", str(folder / "replay")], logs, environment)
            demo = check_demo(folder)
            verification = json.loads((folder / "out/verification.json").read_text(encoding="utf-8"))
        # Check again: the frozen release must still have its original hashes.
        if check_frozen_files(ROOT) != original:
            raise ValueError("Manifest gốc đã thay đổi trong lúc kiểm tra")
        test_count = re.search(r"Ran (\d+) tests?", tests)
        if test_count is None or verification.get("passed") is not True:
            raise ValueError("Thiếu kết quả kiểm thử hoặc kiểm chứng")
        summary = {
            "passed": True, "unit_tests": int(test_count.group(1)),
            "labels_checked": verification["labels_checked"],
            "online_200_max_abs_diff": verification["online_200_max_abs_diff"],
            "mae_hour_Wh": verification["mae_hour_Wh"], "coverage": verification["coverage"],
            "independent_counts": verification["independent_counts"], "demo": demo,
            "scope": "Kiểm model đã lưu trong thư mục tạm; không huấn luyện lại",
        }
        if logs is not None:
            (logs / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"KIỂM TRA KHÔNG ĐẠT: {error}\n")


if __name__ == "__main__":
    main()
