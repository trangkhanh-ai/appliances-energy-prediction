"""Phát lại Data -> AI -> Decision; SQLite chỉ nhận actual khi đủ 60 phút."""
import argparse
import html
import json
import sqlite3
from pathlib import Path
import joblib
import appliances_core as v1
from v2.serve import forecast_one

ROOT = Path(__file__).resolve().parents[1]


def replay(start='2016-05-26 15:30', steps=14, out=None, ack=None, operator=''):
    if ack and not operator.strip():
        raise ValueError('Phản hồi CLI cần tên tự khai; không phải xác thực hoặc nghiệm thu thực')
    out = Path(out) if out else ROOT/'out_v2/demo'
    out.mkdir(parents=True, exist_ok=True)
    database = out/'monitor.sqlite3'
    if database.exists():
        raise ValueError('Chọn thư mục demo mới để giữ nhật ký cũ')
    bundle = joblib.load(ROOT/'out_v2/bundle.joblib')
    df, _ = v1.load_raw(ROOT/'data/energydata_complete.csv')
    match = df.index[df.t == start]
    if not len(match) or steps < 1 or int(match[0])+steps > len(df):
        raise ValueError('Mốc phát lại hoặc số bước không hợp lệ')
    first, rows = int(match[0]), []
    connection = sqlite3.connect(database)
    try:
        connection.execute('CREATE TABLE forecasts(issue_time TEXT PRIMARY KEY, last_reading_wh REAL, step_sum_wh REAL, hour_prediction_wh REAL, upper_wh REAL, energy_threshold REAL, score_threshold REAL, status TEXT, actual_hour_total REAL)')
        connection.execute('CREATE TABLE decisions(issue_time TEXT, operator TEXT, action TEXT, note TEXT)')
        for index in range(first, first+steps):
            rec = forecast_one(bundle, df.iloc[:index+1])
            rec['last'] = float(df.Appliances.iloc[index])
            rec['status'] = 'ALERT' if rec['alert'] else 'NORMAL'
            rows.append((index, rec))
            connection.execute('INSERT INTO forecasts VALUES(?,?,?,?,?,?,?,?,NULL)',
                (rec['issue_time'], rec['last'], rec['point_sum_Wh'], rec['hour_prediction_Wh'],
                 rec['upper_cal'], rec['threshold'], rec['alert_score_threshold'], rec['status']))
            if ack and rec['alert']:
                connection.execute('INSERT INTO decisions VALUES(?,?,?,?)',
                    (rec['issue_time'], operator, ack, 'CLI mô phỏng; không chứng minh can thiệp hoặc nghiệm thu người dùng'))
            # Chỉ cập nhật các dự báo đã đủ nhãn tại đồng hồ replay hiện tại.
            for origin, old in rows:
                if origin+6 <= index:
                    connection.execute('UPDATE forecasts SET actual_hour_total=? WHERE issue_time=?',
                                       (float(df.Appliances.iloc[origin+1:origin+7].sum()), old['issue_time']))
        connection.commit()
        records = connection.execute('SELECT * FROM forecasts ORDER BY issue_time').fetchall()
        decisions = connection.execute('SELECT count(*) FROM decisions').fetchone()[0]
    finally:
        connection.close()
    last = rows[-1][1]
    content = []
    for row in records:
        content.append('<tr>'+''.join(f'<td>{html.escape(str(value) if value is not None else "Chưa đủ 60 phút")}</td>' for value in row)+'</tr>')
    recommendation = 'Kiểm tra số đo và tải đang chạy; người sử dụng xác nhận trước khi cân nhắc thay đổi tải.' if last['alert'] else 'Tiếp tục theo dõi; chưa phát cảnh báo không có nghĩa giờ tới chắc chắn ở mức thấp.'
    page = f'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Dự báo điện năng thiết bị</title>
<style>body{{font-family:Arial,sans-serif;max-width:1250px;margin:35px auto;padding:0 20px;color:#203047}}h1{{font-size:26px}}.summary{{display:flex;gap:24px;flex-wrap:wrap;background:#f1f6fa;padding:20px}}strong{{font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d2dbe5;padding:9px;text-align:left}}th{{background:#eaf1f7}}.alert{{color:#b52920}}p{{line-height:1.6}}</style>
<h1>Dự báo điện năng thiết bị trong 60 phút tới</h1>
<p>Phát lại lịch sử UCI Appliances · phiên {html.escape(bundle['version'])} · {html.escape(last['issue_time'])}. Không kết nối hoặc điều khiển thiết bị thật.</p>
<div class="summary"><div>Trạng thái<br><strong class="alert">{last['status']}</strong></div><div>Dự báo tổng giờ<br><strong>{last['hour_prediction_Wh']:.1f} Wh</strong></div><div>Tổng sáu dự báo 10 phút<br><strong>{last['point_sum_Wh']:.1f} Wh</strong></div><div>Dải đã hiệu chỉnh<br><strong>{last['lower_cal']:.1f}–{last['upper_cal']:.1f} Wh</strong></div></div>
<p>Ngưỡng điện năng nhóm cao: {last['threshold']:.0f} Wh. Chính sách: {html.escape(bundle['policy']['name'])}; ngưỡng điểm cảnh báo: {last['alert_score_threshold']:.2f}. Hai ngưỡng có thể khác nhau vì ngưỡng chính sách được chọn trên dữ liệu hiệu chỉnh.</p>
<p>{recommendation} Phản hồi CLI tự khai: {decisions} bản ghi. Phản hồi mô phỏng không phải nghiệm thu hoặc bằng chứng tiết kiệm thực tế.</p>
<p>Đầu ra tổng giờ dùng cách <b>{html.escape(last['hour_target'])}</b>. Nếu học tổng giờ trực tiếp, giá trị có thể khác tổng sáu dự báo; các đầu ra được hiển thị riêng.</p>
<h2>Nhật ký phiên phát lại</h2><table><tr><th>Thời điểm</th><th>Số đo gần nhất</th><th>Tổng 6 bước</th><th>Dự báo giờ</th><th>Cận trên</th><th>T nhóm cao</th><th>Ngưỡng score</th><th>Trạng thái</th><th>Thực tế đã đến</th></tr>{''.join(content)}</table></html>'''
    (out/'dashboard.html').write_text(page, encoding='utf-8')
    summary = {'version': bundle['version'], 'forecasts': len(records), 'actuals_available': sum(row[-1] is not None for row in records),
               'fabricated_user_acknowledgements': 0, 'cli_decisions': decisions, 'last_issue_time': last['issue_time']}
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2016-05-26 15:30')
    parser.add_argument('--steps', type=int, default=14)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--ack', choices=['viewed', 'inspected', 'ignored'])
    parser.add_argument('--operator', default='')
    args = parser.parse_args()
    print(json.dumps(replay(args.start, args.steps, args.out, args.ack, args.operator), ensure_ascii=False, indent=2))
