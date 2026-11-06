"""demo_replay.py - Mô phỏng Data -> AI -> Decision trên máy cá nhân bằng cách phát lại dữ liệu lịch sử.

Tại mỗi mốc t chỉ đưa vào mô hình các hàng có thời gian <= t (không đọc tương lai). Actual của khung 60 phút
chỉ được gắn vào bản ghi sau khi đủ 6 phép đo tương lai "đã đến". Không gửi lệnh tới thiết bị nào.

Chạy:  python demo_replay.py --start "2016-05-26 15:30" --steps 12 --operator "Ten nguoi van hanh" --ack viewed
Cần out/bundle.joblib (tạo bởi run_all.py). Kết quả: out/demo/monitor.sqlite3, out/demo/dashboard.html
"""
import argparse
import html
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd

import appliances_core as c

ROOT = Path(__file__).parent
ACTIONS = {
    "ALERT": "Kiểm tra các tải đang chạy; cân nhắc hoãn tải linh hoạt trong 60 phút tới; ghi nhận xác nhận.",
    "NORMAL": "Chưa phát cảnh báo; tiếp tục theo dõi số đo và kiểm tra khi có dấu hiệu bất thường.",
    "NO_FORECAST": "Không phát dự báo: lịch sử chưa đủ 145 phép đo liên tiếp. Kiểm tra nguồn dữ liệu.",
}


def classify(rec):
    return "ALERT" if rec["alert"] else "NORMAL"


def init_db(path):
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS forecasts(issue_time TEXT PRIMARY KEY, last_reading_Wh REAL, point_sum_Wh REAL, p10 REAL, p50 REAL,
        p90 REAL, lower_cal REAL, upper_cal REAL, threshold REAL, status TEXT, actual_hour_total REAL);
    CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, issue_time TEXT, operator TEXT, action TEXT, note TEXT,
        logged_at TEXT);""")
    con.execute("PRAGMA journal_mode=WAL")
    return con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-05-26 15:30")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--operator", default="")
    ap.add_argument("--ack", choices=["viewed", "inspected", "ignored"], default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "demo")
    a = ap.parse_args()
    if a.steps < 1:
        ap.error("--steps phải lớn hơn 0")
    a.out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(ROOT / "out" / "bundle.joblib")
    df, _ = c.load_raw(ROOT / "data" / "energydata_complete.csv")      # nguồn dữ liệu (replay)
    db = a.out / "monitor.sqlite3"
    if db.exists():
        db.unlink()
    con = init_db(db)
    t0 = pd.Timestamp(a.start)
    matches = df.index[df.t == t0]
    if not len(matches) or int(matches[0]) + a.steps > len(df):
        con.close()
        ap.error("Mốc bắt đầu hoặc số bước vượt phạm vi dữ liệu")
    i0 = int(matches[0])
    rows = []
    print(f"{'thời điểm':<17}{'đo gần nhất':>12}{'điểm Σ6':>9}{'P50':>7}{'cận trên':>9}{'ngưỡng':>8}  trạng thái")
    for k in range(a.steps):
        i = i0 + k
        history = df.iloc[: i + 1]                                    # CHỈ dữ liệu <= t
        if len(history) < c.DAY + 1:
            status, rec = "NO_FORECAST", None
        else:
            rec = c.forecast_one(bundle, history.tail(c.DAY + 1))
            status = classify(rec)
        row = {"issue_time": str(df.t.iloc[i]), "last": float(df.Appliances.iloc[i]), "status": status, "rec": rec}
        # actual của khung 60 phút chỉ có sau khi 6 phép đo tương lai đã đến, tức là tại thời điểm phát lại hiện tại i_now
        rows.append((i, row))
        if rec:
            print(f"{row['issue_time']:<17}{row['last']:>12.0f}{rec['point_sum_Wh']:>9.0f}{rec['p50']:>7.0f}{rec['upper_cal']:>9.0f}"
                  f"{rec['threshold']:>8.0f}  {status}")
            con.execute("INSERT OR REPLACE INTO forecasts VALUES(?,?,?,?,?,?,?,?,?,?,NULL)",
                        (row["issue_time"], row["last"], rec["point_sum_Wh"], rec["p10"], rec["p50"], rec["p90"], rec["lower_cal"],
                         rec["upper_cal"], rec["threshold"], status))
            if status == "ALERT" and a.ack:
                con.execute("INSERT INTO decisions(issue_time,operator,action,note,logged_at) VALUES(?,?,?,?,datetime('now'))",
                            (row["issue_time"], a.operator or "(tự khai, chưa xác thực)", a.ack, "Phản hồi CLI mô phỏng, không phải nghiệm thu hay can thiệp thực"))
    # gắn actual cho các bản ghi đã đủ 6 phép đo tại thời điểm phát lại cuối cùng
    i_now = i0 + a.steps - 1
    for i, row in rows:
        if row["rec"] and i + c.HORIZON <= i_now:
            actual = float(df.Appliances.iloc[i + 1: i + 1 + c.HORIZON].sum())
            row["actual"] = actual
            con.execute("UPDATE forecasts SET actual_hour_total=? WHERE issue_time=?", (actual, row["issue_time"]))
    con.commit()
    last = rows[-1][1]
    render_html(a.out / "dashboard.html", rows, last, con)
    con.close()
    print(f"\nĐã ghi {db.name} và dashboard.html trong {a.out}")


def render_html(path, rows, last, con):
    rec = last["rec"]
    if rec is None:
        path.write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Chưa đủ dữ liệu</title>'
                        '<h1>KHÔNG PHÁT DỰ BÁO</h1><p>Cần ít nhất 145 số đo liên tiếp. Kiểm tra dữ liệu trước khi sử dụng.</p></html>', encoding="utf-8")
        return
    T = rec["threshold"]
    ups = [r["rec"]["upper_cal"] for _, r in rows if r["rec"]]
    p50s = [r["rec"]["p50"] for _, r in rows if r["rec"]]
    ymax = max(ups + [T]) * 1.1
    W, H = 560, 150

    def pts(v):
        return " ".join(f"{20 + k * (W - 40) / max(len(v) - 1, 1):.1f},{H - 15 - x / ymax * (H - 30):.1f}" for k, x in enumerate(v))
    ty = H - 15 - T / ymax * (H - 30)
    svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Cận trên và P50 theo thời gian">'
           f'<line x1="20" x2="{W-20}" y1="{ty:.1f}" y2="{ty:.1f}" stroke="#b91c1c" stroke-dasharray="5 4"/>'
           f'<text x="24" y="{ty-4:.1f}" font-size="11" fill="#b91c1c">Ngưỡng T = {T:.0f} Wh</text>'
           f'<polyline fill="none" stroke="#0f766e" stroke-width="2" points="{pts(p50s)}"/>'
           f'<polyline fill="none" stroke="#1d4ed8" stroke-width="2" points="{pts(ups)}"/>'
           f'<text x="24" y="14" font-size="11" fill="#1d4ed8">Cận trên sau calibration</text>'
           f'<text x="190" y="14" font-size="11" fill="#0f766e">P50</text></svg>')
    status = last["status"]
    color = {"ALERT": "#b91c1c", "NORMAL": "#15803d", "NO_FORECAST": "#6b7280"}[status]
    label = {"ALERT": "CẢNH BÁO", "NORMAL": "BÌNH THƯỜNG", "NO_FORECAST": "KHÔNG PHÁT DỰ BÁO"}[status]
    trs = []
    for _, r in rows:
        x = r["rec"]
        if not x:
            continue
        act = f"{r['actual']:.0f}" if "actual" in r else "chưa có"
        mark = "⚠" if r["status"] == "ALERT" else ""
        trs.append(f"<tr><td>{r['issue_time'][5:16]}</td><td>{r['last']:.0f}</td><td>{x['point_sum_Wh']:.0f}</td><td>{x['p50']:.0f}</td>"
                   f"<td>{x['upper_cal']:.0f}</td><td>{mark} {r['status']}</td><td>{act}</td></tr>")
    steps = "".join(f"<li>t+{10 * (k + 1)} phút: <b>{v:.0f}</b> Wh</li>" for k, v in enumerate(rec["point_steps_Wh"]))
    dec = con.execute("SELECT issue_time, operator, action FROM decisions ORDER BY id DESC LIMIT 3").fetchall()
    dec_html = "".join(f"<li>{html.escape(a)} · {html.escape(b)} · <b>{html.escape(c_)}</b></li>" for a, b, c_ in dec) or "<li>Chưa có phản hồi CLI; chưa có nghiệm thu người dùng</li>"
    doc = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demo cảnh báo tiêu thụ điện 60 phút</title><style>
:root{{--bg:#f8fafc;--fg:#0f172a;--card:#fff;--line:#e2e8f0}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f172a;--fg:#e2e8f0;--card:#1e293b;--line:#334155}}}}
body{{margin:0;padding:16px;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
.grid{{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}}
h1{{font-size:18px;margin:0 0 10px}} h2{{font-size:13px;margin:0 0 6px;text-transform:uppercase;letter-spacing:.04em;opacity:.7}}
.status{{font-size:26px;font-weight:700;color:{color}}} table{{border-collapse:collapse;width:100%;font-size:12px}}
td,th{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right}} td:first-child,th:first-child{{text-align:left}}
ul{{margin:4px 0 0 18px;padding:0}} small{{opacity:.7}}</style></head><body>
<h1>Cảnh báo tiêu thụ điện thiết bị trong 60 phút tới <small>· phát lại lịch sử, trang tĩnh, không điều khiển thiết bị</small></h1>
<div class="grid">
<div class="card"><h2>Trạng thái hiện tại</h2><div class="status">{label}</div>
<div>Thời điểm phát: <b>{last['issue_time']}</b><br>Đo gần nhất: <b>{last['last']:.0f}</b> Wh/10 phút</div></div>
<div class="card"><h2>Rủi ro dự báo (tổng 60 phút, Wh)</h2>P10 / P50 / P90: <b>{rec['p10']:.0f} / {rec['p50']:.0f} / {rec['p90']:.0f}</b><br>
Dải sau calibration: <b>{rec['lower_cal']:.0f} – {rec['upper_cal']:.0f}</b><br>Điểm Σ6: <b>{rec['point_sum_Wh']:.0f}</b> · Ngưỡng T: <b>{T:.0f}</b></div>
<div class="card"><h2>Khuyến nghị</h2>{html.escape(ACTIONS[status])}<br><small>Đây là gợi ý; con người quyết định.</small>
<h2 style="margin-top:8px">Phản hồi CLI mô phỏng gần đây</h2><ul>{dec_html}</ul></div>
<div class="card"><h2>Dự báo từng bước</h2><ul>{steps}</ul></div></div>
<div class="card" style="margin-top:12px"><h2>Diễn biến cận trên so với ngưỡng</h2>{svg}</div>
<div class="card" style="margin-top:12px"><h2>Nhật ký các mốc đã phát lại</h2><table><tr><th>Thời điểm</th><th>Đo gần nhất</th><th>Điểm Σ6</th><th>P50</th><th>Cận trên</th><th>Trạng thái</th><th>Actual 60 phút</th></tr>{''.join(trs)}</table>
<small>Actual chỉ hiện sau khi đủ 6 phép đo tương lai đã đến tại thời điểm replay hiện tại.</small></div></body></html>"""
    path.write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
