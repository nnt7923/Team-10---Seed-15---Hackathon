"""Render the flow diagrams used in README.md -> docs/images/*.png  (matplotlib only)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
SURFACE, TEXT, TEXT2, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
KIND = {  # fill, stroke
    "src": ("#eef4fc", "#2a78d6"),
    "proc": ("#ffffff", "#52514e"),
    "gpu": ("#fdf0ea", "#eb6834"),
    "out": ("#e9f7f1", "#1baf7a"),
    "dec": ("#fff7e6", "#eda100"),
    "note": ("#f4f3f0", "#c9c8c2"),
}


class Diagram:
    def __init__(self, w, h, title, subtitle=None):
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=150, facecolor=SURFACE)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.axis("off")
        self.n = {}
        self.ax.text(30, 38, title, fontsize=15, fontweight="bold", color=TEXT, va="center")
        if subtitle:
            self.ax.text(30, 66, subtitle, fontsize=9.5, color=TEXT2, va="center")

    def box(self, key, x, y, w, h, title, body=None, kind="proc", fs=9.5):
        fill, stroke = KIND[kind]
        shape = "round,pad=0,rounding_size=10"
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=shape, fc=fill, ec=stroke,
                                         lw=1.6 if kind != "note" else 1.0,
                                         ls="--" if kind == "gpu" else "-"))
        if body:
            self.ax.text(x + 12, y + 18, title, fontsize=fs + 0.5, fontweight="bold", color=TEXT, va="center")
            self.ax.text(x + 12, y + 34, body, fontsize=fs - 1.2, color=TEXT2, va="top", linespacing=1.45)
        else:
            self.ax.text(x + w / 2, y + h / 2, title, fontsize=fs, color=TEXT, ha="center", va="center",
                         fontweight="bold" if kind != "note" else "normal", linespacing=1.4)
        self.n[key] = (x, y, w, h)

    def _port(self, key, side):
        x, y, w, h = self.n[key]
        return {"t": (x + w / 2, y), "b": (x + w / 2, y + h), "l": (x, y + h / 2), "r": (x + w, y + h / 2)}[side]

    def arrow(self, a, b, pa="b", pb="t", label=None, rad=0.0, dashed=False, lpos=0.5, loff=(6, 0)):
        p, q = self._port(a, pa), self._port(b, pb)
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=7,head_width=4", color=LINE, lw=1.4,
                                          connectionstyle=f"arc3,rad={rad}", ls="--" if dashed else "-",
                                          shrinkA=2, shrinkB=2))
        if label:
            mx, my = p[0] + (q[0] - p[0]) * lpos + loff[0], p[1] + (q[1] - p[1]) * lpos + loff[1]
            self.ax.text(mx, my, label, fontsize=8, color=TEXT2, va="center",
                         bbox=dict(fc=SURFACE, ec="none", pad=1.5))

    def legend(self, x, y, items):
        for i, (kind, label) in enumerate(items):
            fill, stroke = KIND[kind]
            self.ax.add_patch(FancyBboxPatch((x + i * 175, y), 22, 14, boxstyle="round,pad=0,rounding_size=3",
                                             fc=fill, ec=stroke, lw=1.4, ls="--" if kind == "gpu" else "-"))
            self.ax.text(x + i * 175 + 30, y + 7, label, fontsize=8.5, color=TEXT2, va="center")

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(OUT / name, facecolor=SURFACE)
        plt.close(self.fig)
        print(OUT / name)


def overview():
    d = Diagram(1500, 980, "Luồng xây dựng bộ dữ liệu cân bằng",
                "Mỗi bước ghi checkpoint .jsonl trong work/ → chạy lại được từ bất kỳ bước nào "
                "(scripts/run_pipeline.py --stages ...)")
    srcs = [("A · Việt Nam", "VN tự gán nhãn (val/test) · VN-ViSeg\nRoboflow VN · Zalo signs"),
            ("B · Châu Á", "IDD (Ấn Độ)\nxe máy, autorickshaw"),
            ("C · Bộ lớn", "Mapillary v2 · Cityscapes\nBDD100K · COCO street"),
            ("D · Thời tiết xấu", "ACDC\nfog · night · rain · snow"),
            ("E · Tự thu", "Dashcam VN\nchưa có nhãn")]
    for i, (t, b) in enumerate(srcs):
        d.box(f"s{i}", 30 + i * 292, 95, 270, 92, t, b, kind="src")
    d.box("conv", 330, 235, 840, 70, "1 · convert", "8 định dạng → record JSONL chung  ·  ánh xạ nhãn gốc → token (glob)  ·  "
          "Cityscapes/Mapillary: vẽ lại theo thứ tự che khuất", kind="proc")
    for i in range(5):
        d.arrow(f"s{i}", "conv", "b", "t", rad=0)
    d.box("res", 330, 345, 840, 70, "2 · resolve", "ghép người lái + xe → 5/7 (người dắt xe = 8)  ·  tách mask chồng lấn (vật gần giữ pixel)  ·  "
          "bỏ xe đẩy, nhóm đám đông  ·  bỏ vật cạnh ngắn < 1.25% cạnh dài ảnh", kind="proc")
    d.arrow("conv", "res")
    d.box("sam", 30, 460, 400, 82, "3 · sam  (GPU, tuỳ chọn)", "nguồn chỉ có box (Zalo, Roboflow)\n→ SAM 2.1 / SAM 3 tạo mask từ box", kind="gpu")
    d.box("comp", 1070, 460, 400, 82, "4 · complete  (GPU, tuỳ chọn)", "nguồn thiếu class (COCO không có biển,\nBDD không có đèn) → teacher bổ sung", kind="gpu")
    d.box("clean", 480, 460, 540, 82, "5 · clean", "pHash + độ sáng · gắn ngày/đêm/mưa/sương\nloại ảnh gần trùng (LSH) · trần khung hình/video", kind="proc")
    d.arrow("res", "clean")
    d.arrow("res", "sam", "l", "t", rad=0.25, dashed=True)
    d.arrow("res", "comp", "r", "t", rad=-0.25, dashed=True)
    d.arrow("sam", "clean", "r", "l", dashed=True)
    d.arrow("comp", "clean", "l", "r", dashed=True)
    d.box("split", 330, 585, 380, 92, "6a · split theo group", "val / test CHỈ từ ảnh VN nhãn đầy đủ\ncùng video → cùng tập (chống rò rỉ)", kind="proc")
    d.box("bal", 790, 585, 380, 92, "6b · balance", "chọn ảnh greedy theo thâm hụt class\n→ repeat-factor sampling → quota VN ≥ 35%", kind="dec")
    d.arrow("clean", "split", "b", "t")
    d.arrow("split", "bal", "r", "l", label="train pool", lpos=0.0, loff=(8, -16))
    d.box("rep", 330, 725, 380, 82, "7 · report", "thống kê trước / sau cân bằng\n5 biểu đồ + stats/REPORT.md", kind="out")
    d.box("exp", 790, 725, 380, 82, "8 · export", "du_lieu.zip chuẩn BTC · train.txt có lặp\nmetadata/ + datasets_report.md cho slide", kind="out")
    d.arrow("bal", "rep", "b", "t", rad=0.0)
    d.arrow("bal", "exp", "b", "t")
    d.box("train", 560, 850, 380, 70, "yolo26n-seg · imgsz 640", kind="out", fs=11)
    d.arrow("exp", "train", "b", "t")
    d.legend(30, 930, [("src", "nguồn dữ liệu"), ("proc", "xử lý CPU"), ("gpu", "cần GPU / tuỳ chọn"),
                       ("dec", "cân bằng"), ("out", "đầu ra")])
    d.save("pipeline_overview.png")


def rider_merge():
    d = Diagram(1300, 900, "Ghép người lái với xe (class 4/5 và 6/7)",
                "Bộ dữ liệu công khai tách rider và xe thành 2 instance; đề bài cần 'xe máy có người' là một đối tượng")
    d.box("in", 470, 95, 360, 60, "Mỗi người: @rider / @rider_moto /\n@rider_bike / @person", kind="src")
    d.box("cand", 400, 195, 500, 78, "Xét từng xe 2 bánh trong ảnh",
          "rider_moto chỉ xét xe máy · rider_bike chỉ xét xe đạp\n@rider, @person xét cả hai", kind="proc")
    d.arrow("in", "cand")
    d.box("geo", 400, 315, 500, 92, "Đo vùng tiếp giáp + hình học",
          "nới mask người 8% → % diện tích xe bị chạm (≥ 0.10 rider, ≥ 0.25 person)\n"
          "tâm người trên xe, đầu cao hơn xe · @person: chân cao hơn đáy xe ≥ 8%\n"
          "(người dắt xe đứng dưới đất → Pedestrian, xe → Motorcycle)", kind="dec")
    d.arrow("cand", "geo")
    d.box("match", 120, 470, 430, 92, "Có xe khớp nhất",
          "gán người vào xe đó (1 xe có thể 2–3 người)\n→ xe máy ⇒ 5 motorcyclist · xe đạp ⇒ 7 cyclist", kind="proc")
    d.box("nomatch", 750, 470, 430, 92, "Không khớp xe nào",
          "@person → 8 pedestrian (kể cả người dắt xe)\n@rider_moto/bike → giữ mask người làm 5/7\n@rider chung chung → bỏ (ghi log)", kind="proc")
    d.arrow("geo", "match", "b", "t", label="có", rad=0.1)
    d.arrow("geo", "nomatch", "b", "t", label="không", rad=-0.1)
    d.box("mode", 120, 610, 430, 120, "Mask đầu ra theo rider_merge.mode",
          "union   : xe ∪ tất cả người lái  (mặc định)\nvehicle : chỉ phần xe\nrider   : chỉ phần người\n"
          "→ nhiều mảnh: nối thành 1 polygon (Ultralytics)", kind="dec")
    d.arrow("match", "mode")
    d.box("rest", 750, 610, 430, 70, "Xe không có người → 4 motorcycle / 6 bicycle", kind="proc")
    d.box("warn", 120, 775, 1060, 70, "✓ Guideline BTC: 'một instance duy nhất phủ cả xe và mọi người trên xe' → mode = union.\n"
          "Sau khi ghép, bước tách chồng lấn bảo đảm mask không đè lên đối tượng khác", kind="note")
    d.save("rider_merge.png")


def balancing():
    d = Diagram(1300, 1000, "Thuật toán cân bằng",
                "Bước A chọn ảnh để phân bố instance gần đều; bước B lặp ảnh class hiếm; bước C giữ đúng miền Việt Nam")
    d.box("pool", 430, 95, 440, 70, "Train pool đủ điều kiện",
          "đã có mask · nhãn đầy đủ (hoặc đã qua teacher) · không thuộc val/test", kind="src")
    d.box("seed", 430, 205, 440, 66, "Khởi tạo: giữ TẤT CẢ ảnh VN nhãn đầy đủ", "cur_c = số instance hiện có của từng class", kind="proc")
    d.arrow("pool", "seed")
    d.box("score", 300, 315, 700, 120, "A · Chấm điểm mọi ảnh còn lại (vector hoá numpy)",
          "deficit_c = max(T − cur_c, 0)      T = target_instances_per_class (như nhau cho 11 class)\n"
          "gain_i    = Σ_c min(n_ic, deficit_c) · deficit_c / T     ← class càng thiếu, instance càng đáng giá\n"
          "excess_i  = Σ_c n_ic · clip(cur_c / T − 1.5, 0, 1)       ← chỉ phạt class đã vượt 1.5×T, có trần\n"
          "score_i   = (gain_i − λ·excess_i) × (1 + bonus đêm/mưa/sương)", kind="dec", fs=9.5)
    d.arrow("seed", "score")
    d.box("pick", 300, 475, 700, 70, "Lấy top-B ảnh có score > 0, bỏ qua nguồn đã chạm trần max_images",
          "cập nhật cur_c sau mỗi ảnh", kind="proc")
    d.arrow("score", "pick")
    d.box("stop", 1050, 465, 220, 92, "Dừng khi", "mọi class đủ T\nhoặc hết ngân sách\nhoặc hết ảnh có lợi", kind="note", fs=9)
    d.arrow("pick", "score", "l", "l", rad=-0.6, label="lặp", loff=(-60, 0))
    d.arrow("pick", "stop", "r", "l", dashed=True)
    d.box("bg", 300, 585, 700, 56, "+ 2% ảnh nền (không có đối tượng) để giảm báo nhầm", kind="proc")
    d.arrow("pick", "bg")
    d.box("rfs", 300, 680, 700, 100, "B · Repeat-factor sampling (LVIS)",
          "f_c = tỉ lệ ảnh đã chọn có class c      r_c = max(1, √(t / f_c))\n"
          "r_i = max_{c ∈ ảnh i} r_c  → làm tròn ngẫu nhiên, tối đa rfs_max_repeat\n"
          "thực hiện bằng cách lặp dòng trong train.txt (chạy được với Ultralytics gốc)", kind="dec", fs=9.5)
    d.arrow("bg", "rfs")
    d.box("dom", 300, 820, 700, 66, "C · Quota miền: nếu ảnh VN < 35% luồng train → nhân số lần lặp ảnh VN",
          "đo lại max/min, entropy, Gini → stats/REPORT.md", kind="out")
    d.arrow("rfs", "dom")
    d.save("balancing_algorithm.png")


def training_plan():
    d = Diagram(1400, 900, "Kế hoạch 3 vòng: teacher → hoàn thiện nhãn → train yolo26n-seg",
                "Thuê GPU cho teacher lớn; model nộp bài vẫn là yolo26n-seg @ 640 theo luật")
    d.box("p1", 40, 100, 400, 150, "Vòng 1 · Teacher",
          "dữ liệu: các nguồn nhãn ĐẦY ĐỦ\n(Mapillary, VN tự gán, IDD/ACDC nếu có)\nmodel: yolo26x-seg, imgsz 1024\n"
          "GPU thuê: vài giờ (hạn chót ~24h)", kind="gpu")
    d.box("p2", 500, 100, 400, 150, "Vòng 2 · Hoàn thiện nhãn",
          "SAM: box → mask (Zalo, Roboflow)\nteacher: thêm class còn thiếu (COCO, BDD)\n"
          "teacher + SAM: gán nhãn dashcam VN\nngười duyệt mẫu needs_review (CVAT)", kind="gpu")
    d.box("p3", 960, 100, 400, 150, "Vòng 3 · Bộ dữ liệu cân bằng",
          "run_pipeline: clean → balance →\nreport → export\n→ dataset_out/ + metadata + stats", kind="out")
    d.arrow("p1", "p2", "r", "l", label="best.pt", loff=(-22, -12))
    d.arrow("p2", "p3", "r", "l", label="02_resolved", loff=(-36, -12))
    d.box("train", 960, 320, 400, 120, "Train yolo26n-seg @ 640",
          "từ weights pretrain COCO · 200–300 epoch\ncopy_paste 0.3–0.5 · mosaic · close_mosaic 15\n"
          "batch lớn trên GPU thuê", kind="out")
    d.arrow("p3", "train")
    d.box("eval", 960, 500, 400, 92, "Đánh giá trên val/test VN",
          "val VN tự gán + public leaderboard (20 lượt)\nmAP50-95 mask từng class · nhầm 4↔5, 6↔7, 5↔8", kind="dec")
    d.arrow("train", "eval")
    d.box("abl", 500, 500, 400, 92, "Ablation từng nguồn",
          "baseline = Mapillary + COCO → thêm từng nguồn\nnguồn làm tụt điểm: giảm max_images / bỏ", kind="proc")
    d.arrow("eval", "abl", "l", "r")
    d.box("al", 40, 500, 400, 92, "Thu thêm dữ liệu có mục tiêu",
          "class yếu nhất → tìm/quay thêm cảnh đó\nảnh model sai → ưu tiên gán nhãn tay", kind="src")
    d.arrow("abl", "al", "l", "r")
    d.arrow("al", "p2", "t", "b", rad=0.0, label="vòng lặp active learning", loff=(10, 0))
    d.box("note", 40, 660, 1320, 150, "Nguyên tắc",
          "1. BTC không cấp dữ liệu train → val/test = ảnh VN đội tự gán theo guideline BTC (thước đo chính).\n"
          "2. Không trộn ảnh nhãn thiếu class vào train nếu chưa qua teacher (tạo false negative).\n"
          "3. Mọi nhãn tự động giữ origin + conf → lọc và duyệt được; thống kê tách nhãn người / máy.\n"
          "4. Đổi một yếu tố mỗi lần thí nghiệm; ghi lại config + stats.json của từng phiên bản dữ liệu.", kind="note", fs=10)
    d.save("training_plan.png")


if __name__ == "__main__":
    overview()
    rider_merge()
    balancing()
    training_plan()
