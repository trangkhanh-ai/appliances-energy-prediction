"""Tạo và thực thi ba notebook đọc kết quả đã chốt, không tự train lại TEST."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd()
if not (ROOT / 'appliances_core.py').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import appliances_core as v1
from v2 import core
'''


def main(execute=True):
    if sys.platform == 'win32':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import nbformat
    from nbclient import NotebookClient
    cells = nbformat.v4
    notebooks = {
        '01_du_lieu_va_tien_xu_ly.ipynb': [
            cells.new_markdown_cell('# Dữ liệu và tiền xử lý\nUCI Appliances ghi điện năng thiết bị của một ngôi nhà. Mục tiêu là Wh trong sáu khoảng 10 phút tiếp theo. Không suy ra công suất đỉnh hoặc tiết kiệm thực tế từ các số đo này.'),
            cells.new_code_cell(SETUP),
            cells.new_code_cell("df, audit = v1.load_raw(ROOT/'data/energydata_complete.csv')\ndisplay(pd.DataFrame([audit]))\ndisplay(df.head())"),
            cells.new_markdown_cell('## Vì sao giữ đỉnh và chia theo thời gian\nCSV gốc không có ô thiếu hoặc khoảng trống trên lưới 10 phút. Giữ các đỉnh hợp lệ để đánh giá nhóm cao. Scaler chỉ học TRAIN; không xáo trộn chuỗi hoặc dùng SMOTE cho hồi quy này.'),
            cells.new_code_cell("X, Y = core.build_features(df), v1.build_labels(df)\ndisplay(X[core.FEATURE_SETS['F']].iloc[144:149])\ndisplay(pd.DataFrame({'ten': ['A', 'F'], 'so_dac_trung': [18, 22]}))"),
            cells.new_code_cell("i = 5000\nchanged = df.copy()\nchanged.loc[i+1:, 'Appliances'] *= 9\nnp.testing.assert_array_equal(X.iloc[i].to_numpy(), core.build_features(changed).iloc[i].to_numpy())\nprint('PASS: sửa tương lai không làm đổi đặc trưng tại t')\ndisplay(Image(filename=str(ROOT/'out/figures/f32_distribution.png')))"),
        ],
        '02_thuc_nghiem_va_chon_mo_hinh.ipynb': [
            cells.new_markdown_cell('# Thực nghiệm và lựa chọn mô hình\nNotebook trình bày kết quả đã chạy trên hai fold phát triển. VAL trùng fold B. Tuning không dùng TEST; kết quả TEST cũ vẫn là đánh giá hồi cứu.'),
            cells.new_code_cell(SETUP),
            cells.new_code_cell("dev = json.loads((ROOT/'out_v2/development.json').read_text(encoding='utf-8'))\ndisplay(pd.DataFrame(dev['folds']).T)"),
            cells.new_code_cell("rows = [{'cau_hinh': r['config']['name'], 'fold': r['fold'], **r['metrics']} for r in dev['rows']]\ndisplay(pd.DataFrame(rows))"),
            cells.new_markdown_cell('## Quy tắc chốt\nChấp nhận phần dự báo mới khi MAE nhóm cao giảm ít nhất 5% và MAE toàn bộ tăng không quá 3% trên cả hai fold. Chính sách cảnh báo được hiệu chỉnh trên nửa đầu mỗi giai đoạn, đánh giá ở nửa sau có purge. Giữ chính sách v1 là phương án dự phòng, không phải bằng chứng đã cải thiện cảnh báo.'),
            cells.new_code_cell("display(pd.DataFrame([{'cau_hinh': r['config']['name'], 'dat_tieu_chi': r['point_gate_passed'], 'mae_trung_binh': r['mean_mae'], 'ty_le_mae_nhom_cao': r['mean_relative_high_mae']} for r in dev['ranking']]))\ndisplay(dev['selection']['selected_point'])\ndisplay(Image(filename=str(ROOT/'out_v2/figures/f_v2_development.png')))"),
        ],
        '03_ket_qua_va_demo.ipynb': [
            cells.new_markdown_cell('# Kết quả, cảnh báo và giới hạn\nĐọc các file đã chốt để đối chiếu với Word. Notebook này không gọi huấn luyện hoặc tìm cấu hình theo TEST.'),
            cells.new_code_cell(SETUP),
            cells.new_code_cell("r = json.loads((ROOT/'out_v2/results.json').read_text(encoding='utf-8'))\nv = json.loads((ROOT/'out_v2/verification.json').read_text(encoding='utf-8'))\nprint('Phiên được chọn:', r['version'])\ndisplay(pd.DataFrame([{'phuong_an': k, **item['metrics']} for k, item in r['finalists'].items()]))\ndisplay(v['counts'])"),
            cells.new_code_cell("pred = pd.read_csv(ROOT/'out_v2/test_predictions.csv')\nactual = pred[[f'actual_{i}m' for i in range(10, 61, 10)]].sum(axis=1)\nnp.testing.assert_allclose(actual, pred.actual_hour_total)\nmae = np.abs(actual-pred.selected_hour_prediction).mean()\nnp.testing.assert_allclose(mae, r['selected']['mae'])\nprint('MAE được tính lại từ CSV:', mae)\ndisplay(pred[['issue_time','actual_hour_total','point_hour_total','selected_hour_prediction','p90','upper_cal','alert']].head())"),
            cells.new_code_cell("display(Image(filename=str(ROOT/'out_v2/figures/f_v2_models.png')))\ndisplay(Image(filename=str(ROOT/'out_v2/figures/f_v2_policies.png')))"),
            cells.new_markdown_cell('## Cần diễn giải đúng\nGiảm MAE là giảm sai số dự báo. Chưa có số đo tiết kiệm điện hoặc tiền điện, tải điều khiển được hay nghiệm thu người dùng. Các cửa sổ tổng giờ chồng lấn; lead của cụm dự báo không đồng nghĩa báo trước thời điểm bật thiết bị. Chưa có dữ liệu ngoài để xác nhận tổng quát.'),
            cells.new_code_cell("display(pd.DataFrame(r['robustness']))\ndisplay(r['bootstrap'])\ndisplay(r['system'])"),
        ],
    }
    directory = ROOT/'notebooks'
    directory.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='appliances-kernel-') as temporary:
        kernel = Path(temporary)/'kernels/appliances-local'
        kernel.mkdir(parents=True)
        (kernel/'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'Appliances local', 'language': 'python'}), encoding='utf-8')
        previous = os.environ.get('JUPYTER_PATH')
        os.environ['JUPYTER_PATH'] = temporary
        os.environ['PYTHONUTF8'] = '1'
        try:
            for name, content in notebooks.items():
                nb = cells.new_notebook(cells=content, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
                if execute:
                    NotebookClient(nb, timeout=180, kernel_name='appliances-local', resources={'metadata': {'path': str(ROOT)}}).execute()
                nbformat.write(nb, directory/name)
                print(name, 'executed' if execute else 'created')
        finally:
            if previous is None:
                os.environ.pop('JUPYTER_PATH', None)
            else:
                os.environ['JUPYTER_PATH'] = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-execute', action='store_true')
    main(not parser.parse_args().no_execute)
