"""Tạo hình từ JSON/CSV đã chạy, không huấn luyện hoặc chọn lại mô hình."""
import json
import sqlite3
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'out_v2'


def main():
    development = json.loads((OUT/'development.json').read_text(encoding='utf-8'))
    results = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(OUT/'test_predictions.csv', parse_dates=['issue_time'])
    figures = OUT/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11, 'figure.dpi': 150})
    rows = development['ranking']
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {'baseline': '#566775', 'rise_weight': '#2077a7', 'direct': '#1e8b65', 'tuning': '#c38924', 'deep_learning': '#8b56a2'}
    for stage, color in colors.items():
        subset = [r for r in rows if r['config']['stage'] == stage]
        ax.scatter([r['mean_mae'] for r in subset], [100*r['mean_relative_high_mae'] for r in subset], label=stage, c=color, alpha=.8)
    ax.axhline(95, c='gray', ls='--', lw=1)
    ax.set(xlabel='MAE tổng giờ trung bình hai fold (Wh)', ylabel='MAE nhóm cao so với v1 (%)',
           title='Chọn trên dữ liệu phát triển; chưa dùng TEST')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures/'f_v2_development.png'); plt.close(fig)

    labels = ['v1', 'best_multi', 'best_direct', 'lstm']
    labels = [label for label in labels if label in results['finalists']]
    readable = {'v1': 'Mốc v1', 'best_multi': 'Sáu bước', 'best_direct': 'Tổng giờ trực tiếp', 'lstm': 'LSTM nhỏ'}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric, title in zip(axes, ['mae', 'high_mae'], ['Toàn bộ TEST', 'Nhóm tiêu thụ cao TEST']):
        ax.bar([readable[x] for x in labels], [results['finalists'][label]['metrics'][metric] for label in labels], color=['#63758b', '#2077a7', '#1e8b65', '#8b56a2'][:len(labels)])
        ax.set(title=title, ylabel='MAE tổng giờ (Wh)')
        ax.tick_params(axis='x', rotation=15)
    fig.suptitle('Các ứng viên đã khóa trước đợt đánh giá TEST hồi cứu')
    fig.tight_layout(); fig.savefig(figures/'f_v2_models.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    policies = ['raw_p90', 'upper', 'point']
    readable_policy = ['P90 thô', 'P90 + margin', 'Dự báo giờ']
    chosen = [results['policy_comparison'][name]['fixed_fpr'] for name in policies]
    axes[0].bar(readable_policy, [100*x['recall'] for x in chosen], color='#2077a7')
    axes[0].set(ylabel='Recall (%)', ylim=(0, 100), title='Phát hiện thời điểm cao')
    axes[1].bar(readable_policy, [x['fp'] for x in chosen], color='#ca793d')
    axes[1].set(ylabel='Số báo nhầm', title='Báo nhầm trên TEST')
    fig.suptitle('Ngưỡng chọn trên CAL với FPR mục tiêu 10%; FPR TEST có thể khác')
    fig.tight_layout(); fig.savefig(figures/'f_v2_policies.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.7))
    subset = predictions.iloc[:288]
    ax.plot(subset.issue_time, subset.actual_hour_total, c='#243444', label='Thực tế tổng giờ')
    ax.plot(subset.issue_time, subset.selected_hour_prediction, c='#1e8b65', label='Dự báo được chọn')
    ax.fill_between(subset.issue_time, subset.lower_cal, subset.upper_cal, color='#2077a7', alpha=.15, label='Dải hiệu chỉnh')
    ax.axhline(results['threshold'], color='#bb5042', ls='--', label='T nhóm cao')
    ax.set(ylabel='Wh', title='Hai ngày đầu TEST, giữ nguyên thứ tự thời gian')
    ax.legend(fontsize=8); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(figures/'f_v2_forecast.png'); plt.close(fig)

    neural = [r for r in development['rows'] if r['config']['method'] == 'lstm']
    if neural:
        fig, axes = plt.subplots(1, len(neural), figsize=(9, 3.6), squeeze=False)
        for ax, row in zip(axes[0], neural):
            history = row['training']['history']
            ax.plot([h['epoch'] for h in history], [h['train_mae_step_Wh'] for h in history], label='Inner TRAIN')
            ax.plot([h['epoch'] for h in history], [h['validation_mae_step_Wh'] for h in history], label='Inner validation')
            ax.axvline(row['training']['epochs'], c='gray', ls='--', label='Epoch chọn')
            ax.set(title=row['fold'], xlabel='Epoch', ylabel='MAE/bước (Wh)'); ax.legend(fontsize=8)
        fig.suptitle('LSTM: chọn epoch trong TRAIN của từng fold, không dùng TEST')
        fig.tight_layout(); fig.savefig(figures/'f_v2_lstm.png'); plt.close(fig)

    choice = results['selection']
    point_name = choice['selected_point']['name']
    blocks = [
        ('1  Nguồn dữ liệu', 'CSV UCI Appliances: số đo điện năng thiết bị mỗi 10 phút'),
        ('2  Xử lý dữ liệu', 'Kiểm lưới thời gian; đặc trưng quá khứ và lịch; từ chối đầu vào lỗi'),
        ('3  Mô hình AI', f"Phiên {choice['production_version']}: {point_name}; bộ phân vị riêng"),
        ('4  Kết quả dự báo', 'Sáu bước; dự báo tổng giờ; P10/P50/P90 và dải hiệu chỉnh'),
        ('5  Hỗ trợ quyết định', f"Chính sách {choice['selected_policy']}; ngưỡng hiệu chỉnh từ dữ liệu phát triển/CAL"),
        ('6  Người sử dụng', 'Xem dự báo, xác minh số đo và cân nhắc kiểm tra tải'),
        ('7  Hành động đề xuất', 'Con người quyết định; chưa điều khiển thiết bị hoặc đo tiết kiệm'),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    for i, (title, detail) in enumerate(blocks):
        y = .96-i*.12
        ax.add_patch(plt.Rectangle((.025, y-.09), .95, .095, facecolor='#f0f5f9', edgecolor='#325c7a'))
        ax.text(.5, y-.023, title, ha='center', va='center', weight='bold', fontsize=12)
        ax.text(.5, y-.063, detail, ha='center', va='center', fontsize=10.5)
        if i < 6:
            ax.annotate('', xy=(.5, y-.116), xytext=(.5, y-.092), arrowprops={'arrowstyle': '->', 'color': '#325c7a'})
    ax.text(.5, .04, 'SQLite lưu dự báo và actual khi đủ 60 phút; giám sát sai số và độ phủ\nDữ liệu một ngôi nhà; chưa kiểm chứng ở nhà hoặc giai đoạn mới', ha='center', va='center', fontsize=10)
    fig.tight_layout(); fig.savefig(figures/'f_v2_architecture.png'); plt.close(fig)
    database = OUT/'demo/monitor.sqlite3'
    if database.exists():
        connection = sqlite3.connect(database)
        demo = pd.read_sql_query('SELECT * FROM forecasts ORDER BY issue_time', connection)
        connection.close()
        last = demo.iloc[-1]
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1.5]})
        axes[0].axis('off')
        text = (f"Phiên {results['version']} · {last.issue_time} · {last.status}\n"
                f"Đo gần nhất: {last.last_reading_wh:.0f} Wh/10 phút\n"
                f"Dự báo tổng giờ: {last.hour_prediction_wh:.1f} Wh · Cận trên: {last.upper_wh:.1f} Wh\n"
                f"Ngưỡng nhóm cao: {last.energy_threshold:.0f} Wh\n"
                "Người sử dụng xác minh tải trước khi quyết định; chưa có phản hồi nghiệm thu")
        axes[0].text(.02, .95, text, transform=axes[0].transAxes, va='top', fontsize=12, linespacing=1.5)
        times = pd.to_datetime(demo.issue_time)
        axes[1].plot(times, demo.hour_prediction_wh, label='Dự báo tổng giờ', color='#1e8b65')
        axes[1].plot(times, demo.upper_wh, label='Cận trên hiệu chỉnh', color='#2077a7')
        axes[1].axhline(last.energy_threshold, ls='--', color='#bb5042', label='Ngưỡng nhóm cao')
        axes[1].set(ylabel='Wh'); axes[1].legend(fontsize=9)
        fig.suptitle('Minh họa đầu ra từ nhật ký SQLite của phiên phát lại v2')
        fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(figures/'f_v2_demo.png'); plt.close(fig)
    print('Created figures:', len(list(figures.glob('*.png'))))


if __name__ == '__main__':
    main()
