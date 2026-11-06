"""Kiến trúc hiện có và phần hành động mới chỉ là đề xuất."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'Times New Roman','font.size':12})
blocks=[
    ('1  Nguồn dữ liệu','CSV UCI Appliances, số đo lịch sử mỗi 10 phút'),
    ('2  Xử lý dữ liệu','Kiểm tra dữ liệu; lag, rolling và biến lịch'),
    ('3  Mô hình AI','6 HGB-L1 theo chân trời và 3 HGB phân vị'),
    ('4  Kết quả dự báo','6 bước, tổng 60 phút, P10/P50/P90 và biên CAL'),
    ('5  Hỗ trợ quyết định','So cận trên với T; cảnh báo và gợi ý kiểm tra tải'),
    ('6  Người sử dụng','Đọc trang tĩnh; phản hồi CLI nếu được nhập'),
    ('7  Hành động đề xuất','Con người xác minh và cân nhắc tải linh hoạt')]
fig,ax=plt.subplots(figsize=(6.1,6.5),dpi=200)
ax.set(xlim=(0,6),ylim=(0,8.3));ax.axis('off')
for i,(title,detail) in enumerate(blocks):
    y=7.45-i*1.0
    ax.add_patch(FancyBboxPatch((.12,y),5.72,.70,boxstyle='round,pad=.03',fc='#f2f6fa',ec='#35536a'))
    ax.text(3,y+.47,title,ha='center',va='center',weight='bold',fontsize=12)
    ax.text(3,y+.19,detail,ha='center',va='center',fontsize=10.8)
    if i<6:ax.annotate('',xy=(3,y-.25),xytext=(3,y-.04),arrowprops={'arrowstyle':'->','color':'#35536a'})
ax.text(3,.65,'SQLite lưu dự báo và actual khi đủ 60 phút',ha='center',fontsize=11)
ax.text(3,.29,'Giám sát batch MAE, độ phủ, PSI → con người rà soát',ha='center',fontsize=11)
ax.text(3,-.02,'Chưa nối công tơ thật; chưa điều khiển hoặc đo tiết kiệm',ha='center',fontsize=10)
path=Path(__file__).parent/'out/figures/f51_architecture.png'
path.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(path,bbox_inches='tight');plt.close(fig)
print(path)
