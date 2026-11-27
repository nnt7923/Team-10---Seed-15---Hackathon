# Team-10---Seed-15---Hackathon
# VN Traffic Balanced — bộ dữ liệu instance segmentation giao thông Việt Nam cân bằng 11 class

Pipeline gom nhiều bộ dữ liệu công khai (và dữ liệu tự gán nhãn), chuẩn hoá về **11 lớp của cuộc thi
AIInAction RoboTaxi** theo đúng **guideline gán nhãn của BTC**, ghép **người lái + xe** thành class 5/7,
tách mask chồng lấn, loại ảnh trùng, **cân bằng class**, rồi xuất **`du_lieu.zip` đúng cấu trúc BTC** cho
`yolo26n-seg @ imgsz 640`. Mỗi ảnh có **metadata nguồn gốc đầy đủ**. Mỗi lần chạy sinh **thống kê
trước / sau cân bằng** và **bảng bộ dữ liệu dùng thẳng cho slide báo cáo**.

> Cuộc thi là *data-centric*: model và kích thước ảnh cố định, thứ duy nhất ta cải thiện là dữ liệu.
> Điểm = 100 × mask mAP@[.5:.95], **trung bình đều 11 lớp**, nên class hiếm (3, 7, 10) nặng ngang class 5.

![Tổng quan pipeline](docs/images/pipeline_overview.png)

---

## Mục lục
0. [Tình hình cuộc thi (đọc trước)](#0-tình-hình-cuộc-thi-đọc-trước)
1. [11 lớp & quy ước BTC](#1-11-lớp--quy-ước-btc)
2. [Kế hoạch 24 giờ](#2-kế-hoạch-24-giờ)
3. [Nguồn dữ liệu](#3-nguồn-dữ-liệu)
4. [Ánh xạ class & ghép người lái](#4-ánh-xạ-class--ghép-người-lái)
5. [Làm sạch](#5-làm-sạch)
6. [Thuật toán cân bằng](#6-thuật-toán-cân-bằng)
7. [Đầu ra & metadata](#7-đầu-ra--metadata)
8. [Thống kê trước / sau cân bằng](#8-thống-kê-trước--sau-cân-bằng)
9. [Cài đặt & chạy](#9-cài-đặt--chạy)
10. [Huấn luyện & nộp bài](#10-huấn-luyện--nộp-bài)
11. [Cấu trúc thư mục](#11-cấu-trúc-thư-mục)
12. [Việc phải kiểm tra & hạn chế](#12-việc-phải-kiểm-tra--hạn-chế)
13. [Tài liệu tham khảo](#13-tài-liệu-tham-khảo)

---

## 0. Tình hình cuộc thi (đọc trước)

Theo trang [competition.antco.ai](https://competition.antco.ai/) (đọc ngày 27/11/2026):

| Điểm | Hệ quả cho pipeline |
|---|---|
| **BTC không cấp dữ liệu train**, chỉ có 10 ảnh mẫu *tổng hợp* | Nguồn chuẩn vàng là **`vn_manual_labeled`**: 200–300+ ảnh VN thật do đội tự gán theo guideline (SAM 3 vẽ nháp, người sửa, người thứ hai duyệt). Đây là **nguồn duy nhất cho val/test** |
| Đóng nộp **10:30 ngày 28/11**, công bố private 11:00, thuyết trình 11:30 | Ưu tiên nguồn tải được ngay. Cityscapes và IDD phải chờ duyệt tài khoản nên chỉ dùng nếu đã có sẵn |
| 20 lượt nộp, mỗi lần một bài đang chấm; chọn tối đa 2 bài cho private | Nộp sớm để có mốc; dùng public leaderboard làm thước đo thứ hai |
| **Thứ hạng = 50% điểm private + 50% thuyết trình** | Slide phải có quy trình dữ liệu và **mọi bộ công khai đã dùng: tên, link, license, số ảnh đã dùng, cách ánh xạ 11 lớp**. Bảng này được sinh tự động tại `metadata/datasets_report.md` |
| Seed của đội: **15** (`TEAM_ID` trong notebook) | Pipeline dùng `seed: 15` |
| Nộp `model.onnx` (fp32, imgsz 640) trong zip ≤ 42 MB, xuất bằng notebook mẫu | Bước 1 của notebook phải dùng `data.yaml` của ta (mục 10) |

## 1. 11 lớp & quy ước BTC

| ID | Class | Định nghĩa BTC (tóm tắt) | Ảnh hưởng tới pipeline |
|---:|---|---|---|
| 0 | car | Sedan, SUV, minivan, **bán tải**, taxi, **xe 9–15 chỗ**, xe cứu thương dạng van | COCO/BDD gán bán tải là *truck* → nhiễu (ghi trong báo cáo) |
| 1 | bus | Xe buýt, xe khách **từ 16 chỗ** | |
| 2 | truck | Xe tải, xe ben, xe bồn, cứu hoả, **chỉ phần đầu kéo** | |
| 3 | other_vehicle | Ba gác, xích lô, xe lam, xe công trình, máy cày, **rơ-moóc / sơ-mi rơ-moóc / container**, mọi phương tiện không thuộc lớp khác | thêm `train`/`on-rails` → 3; trailer → 3 |
| 4 | motorcycle | Xe máy, xe tay ga, xe điện 2 bánh kiểu xe máy, **không người** (đỗ hoặc đang dắt) | |
| 5 | motorcyclist | **Một instance duy nhất phủ cả xe và mọi người trên xe** | `rider_merge.mode: union` |
| 6 | bicycle | Xe đạp, xe đạp điện kiểu xe đạp, không người | |
| 7 | cyclist | Một instance phủ cả xe đạp và người | union |
| 8 | pedestrian | Mọi người không ngồi trên phương tiện, **kể cả người dắt xe** | `@person` chỉ thành rider khi chân cao hơn đáy xe |
| 9 | sign | Biển báo, biển chỉ dẫn, biển công trường; **chỉ mặt biển, không cột**; biển quảng cáo **không gán** | Mapillary `traffic-sign--*` (bỏ mặt sau) |
| 10 | traffic_light | **Cả hộp đèn** (xe và người đi bộ); đèn xe, đèn trang trí không gán | |

Quy ước chung của BTC mà pipeline thực thi:

* **Mask chỉ phủ phần nhìn thấy, không chồng nhau**: vật ở trước giữ pixel → bước `deoverlap` (mục 5).
* **Không gán**: vật sau kính, người ngồi trong ô tô, hình trên áp phích, động vật, **xe đẩy tay**, **vật có cạnh
  ngắn < ~1.25% cạnh dài ảnh** → `min_short_side_frac: 0.0125`; Mapillary `wheeled-slow` → `@ignore`.
* **Đầu kéo = Truck, rơ-moóc = Other vehicle, hai instance riêng.** Bộ công khai thường gộp cả xe container
  vào *truck* → nhiễu đã biết.

Cấu hình: [configs/classes.yaml](configs/classes.yaml).

## 2. Kế hoạch 24 giờ

![Kế hoạch 3 vòng](docs/images/training_plan.png)

| Mốc (giờ VN) | Việc | Ai / ở đâu | Đầu ra |
|---|---|---|---|
| 27/11 chiều | Viết `labeling_guideline` của đội từ guideline BTC; quay/thu ảnh VN (dashcam, điện thoại, đêm, mưa); tải **Mapillary v2, BDD100K, COCO**, Roboflow/Kaggle VN | 2 người thu + 1 người tải | `raw/` |
| 27/11 tối | Gán **val/test VN 200–300 ảnh** (SAM 3 nháp → sửa → duyệt chéo). Song song: `convert,resolve` các nguồn công khai, rồi train **baseline** (Mapillary + COCO) và **nộp lần 1** | 2 người gán, 1 người chạy pipeline + GPU | mốc điểm public |
| 27/11 đêm | Train teacher `yolo26x-seg` vài giờ trên GPU thuê (nguồn nhãn đầy đủ); chạy `sam,complete` cho nguồn thiếu nhãn và ảnh dashcam VN; duyệt nhanh ảnh `needs_review` | GPU thuê | `02_resolved.jsonl` đủ nhãn |
| 28/11 sáng sớm | `clean,balance,report,export` → train `yolo26n-seg` đầy đủ → nộp; ablation 1–2 cấu hình (ví dụ bỏ/giảm một nguồn) | GPU | 2 bài tốt nhất |
| trước 10:30 | **Chọn 2 bài cho private**; chốt slide từ `stats/REPORT.md`, `metadata/datasets_report.md`, sơ đồ trong `docs/images/` | | slide |

Nếu không kịp train teacher, đặt `partial_label_policy: require_completion` (mặc định) để **không** đưa ảnh thiếu
nhãn vào train. Thà ít ảnh hơn còn hơn dạy model rằng biển báo là nền.

## 3. Nguồn dữ liệu

Registry đầy đủ (link, license, trích dẫn, cách tải, định dạng, ánh xạ nhãn): [configs/sources.yaml](configs/sources.yaml).

| Tier | id | Bộ dữ liệu | Định dạng | Quy mô (công bố) | License | Vai trò | Tải trong 24h? |
|---|---|---|---|---|---|---|---|
| A | `vn_manual_labeled` | Ảnh VN đội tự gán theo guideline BTC | YOLO-seg | 200–300+ | tự sở hữu | **chuẩn vàng, val/test** | ✅ tự làm |
| A | `vn_traffic_viseg` | [VN-Traffic-Instance-Segmentation](https://github.com/Tuanvinh1307/VN-Traffic-Instance-Segmentation-YOLO11) | YOLO-seg | ~9.5k + 1.2k test | xem repo | ảnh VN có polygon | ✅ |
| A | `roboflow_vn_traffic` | [Roboflow vn_traffic](https://universe.roboflow.com/projectxla/vn_traffic) | YOLO detect | ~2.9k | theo trang | ảnh VN; mask bằng SAM | ✅ API |
| A | `zalo_traffic_sign_2020` | [Zalo AI 2020 Traffic Sign](https://datasetninja.com/zalo-traffic-sign) | COCO bbox | ~4.5k | không rõ | **biển báo VN** | ✅ |
| E | `vn_dashcam_self` | Dashcam VN tự thu | ảnh thô | tuỳ | tự sở hữu | xích lô, ba gác, chở hàng, đêm, mưa | ✅ |
| B | `idd_segmentation` | [India Driving Dataset](https://inai.iiit.ac.in/domains/idd.html) | polygons JSON | ~20k | phi thương mại | đường đông kiểu châu Á, autorickshaw → 3 | ⏱ chờ duyệt |
| C | `mapillary_vistas_v2` | [Mapillary Vistas v2](https://blog.mapillary.com/update/2021/01/18/vistas-2-dataset.html) | polygons JSON | 25k | CC BY-NC-SA 4.0 | **nguồn tốt nhất cho biển báo & đèn** | ✅ cần đăng nhập |
| C | `cityscapes` | [Cityscapes](https://www.cityscapes-dataset.com/) | polygons JSON | 5k fine | phi thương mại | nhãn tinh | ⏱ chờ duyệt |
| C | `bdd100k_ins_seg` | [BDD100K ins-seg](https://github.com/SysCV/bdd100k-models/blob/main/ins_seg) | Scalabel JSON | 10k | xem LICENSE | đêm/mưa; thiếu biển & đèn | ✅ đăng ký |
| C | `coco2017_street` | [COCO 2017](https://cocodataset.org/) (lọc cảnh phố) | COCO | ~6k sau lọc | CC BY 4.0 (nhãn) | đa dạng; thiếu biển báo | ✅ script |
| D | `acdc` | [ACDC](https://arxiv.org/abs/2104.13395) | COCO-panoptic | ~4k ảnh xấu trời | phi thương mại | sương / đêm / mưa / tuyết | ⏱ đăng ký |

### Ma trận phủ class (✓ có nhãn · ◐ phải sinh mask/bổ sung · ✗ thiếu → teacher)

| Nguồn | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VN tự gán | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| VN-ViSeg (cần kiểm tra) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Roboflow VN | ◐ | ◐ | ◐ | ✗ | ◐ | ◐ | ◐ | ◐ | ◐ | ✗ | ✗ |
| Zalo signs | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | ✗ |
| IDD | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | ✓ | ✓* | ✓ | ✓ | ✓ |
| Mapillary v2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | ✓ | ✓* | ✓ | ✓ | ✓ |
| Cityscapes | ✓ | ✓ | ✓ | ◐ | ✓ | ✓* | ✓ | ✓* | ✓ | ✓ | ✓ |
| BDD100K | ✓ | ✓ | ✓ | ◐ | ✓ | ✓* | ✓ | ✓* | ✓ | ✗ | ✗ |
| COCO street | ✓ | ✓ | ✓ | ✗ | ✓ | ✓* | ✓ | ✓* | ✓ | ✗ | ✓ |
| ACDC | ✓ | ✓ | ✓ | ◐ | ✓ | ✓* | ✓ | ✓* | ✓ | ✓ | ✓ |

\* = sinh ra bằng bước ghép người lái + xe (mục 4). Ô ✗ khai báo trong `labeled_classes`; ảnh của nguồn đó
**không vào train** cho đến khi chạy bước `complete` (`partial_label_policy`).

## 4. Ánh xạ class & ghép người lái

**Ánh xạ** khai báo theo từng nguồn trong `label_map` (khớp chính xác trước, rồi đến glob `*`):

```yaml
"object--traffic-sign--*back*": "@ignore"   # mặt sau biển báo
"object--traffic-sign--*": sign              # biển mặt trước, chỉ dẫn, tạm thời
"object--vehicle--wheeled-slow": "@ignore"   # xe đẩy tay, xe lăn: BTC không gán
"object--vehicle--on-rails": other_vehicle   # mọi phương tiện không thuộc lớp khác
"human--rider--motorcyclist": "@rider_moto"  # người lái -> ghép với xe máy
person: "@person"                            # COCO: chưa biết đi bộ hay đang lái
"*group": "@ignore"                          # nhóm đám đông / nhóm xe
```

Cityscapes / IDD / Mapillary lưu polygon theo thứ tự vẽ (sau đè trước); converter vẽ lại để lấy **phần nhìn thấy**.
Bộ panoptic (ACDC) gộp mọi biển báo vào một segment → tách theo **thành phần liên thông**.

![Ghép người lái](docs/images/rider_merge.png)

Ví dụ (dữ liệu demo tổng hợp): người lái + xe máy thành **một** polygon class 5; xe không người giữ class 4.

![QA ghép rider](docs/images/demo_qa_rider_merge.jpg)

## 5. Làm sạch

| Bước | Cách làm | Tham số |
|---|---|---|
| **Tách chồng lấn** (`deoverlap`) | Quy ước BTC: mask không chồng nhau, vật phía trước giữ pixel. Với nguồn không có thứ tự độ sâu (COCO, BDD, mask SAM, mask teacher, mask ghép rider), vẽ lại theo **xa → gần**: vật có đáy thấp hơn trong ảnh là vật gần camera hơn; biển/đèn vẽ cuối. Instance không chồng với gì giữ nguyên polygon gốc | tự động |
| **Vật quá nhỏ** | bỏ instance có **cạnh ngắn bbox < 1.25% cạnh dài ảnh** (≈ 8 px ở 640), đúng ngưỡng BTC không gán | `min_short_side_frac` |
| Gắn điều kiện | ưu tiên thuộc tính nguồn (BDD, thư mục ACDC); nếu không có thì đoán ngày/đêm theo độ sáng | `conditions.night_luma_thr` |
| Loại ảnh gần trùng | pHash 64-bit + LSH 4 band + union-find; giữ ảnh của nguồn `priority` cao hơn | `dedupe.hamming_thr` |
| Trần theo video | tối đa N khung hình mỗi `group`, lấy mẫu đều | `dedupe.max_per_group` |
| Chia tập | theo **group** (cùng video → cùng tập). **Val/test chỉ từ `vn_manual_labeled`** (nguồn `use_for_val`, nhãn đủ 11 lớp), không cân bằng | `split.*` |

Ví dụ tách chồng lấn (demo): ô tô gần giữ phần chồng, ô tô xa chỉ còn phần nhìn thấy.

![QA tách chồng lấn](docs/images/demo_qa_deoverlap.jpg)

## 6. Thuật toán cân bằng

![Thuật toán cân bằng](docs/images/balancing_algorithm.png)

1. **Chọn ảnh theo thâm hụt (greedy)**: mỗi instance của class *c* có giá trị `deficit_c / T`, tức phần target
   còn thiếu. Vì vậy class hiếm áp đảo điểm số. Class đã vượt `1.5 × T` bị phạt nhẹ mỗi instance, **có trần**,
   nên ảnh chỉ toàn ô tô thừa bị loại, nhưng ảnh có 1 xích lô + vài ô tô vẫn được chọn.
   `condition_bonus` ưu tiên ảnh đêm/mưa/sương; `max_images` từng nguồn chặn một bộ lấn át các bộ khác.
2. **Repeat-factor sampling** (LVIS): ảnh chứa class xuất hiện ở ít ảnh được lặp `r_i = max_c √(t/f_c)` lần,
   thực hiện bằng cách lặp dòng trong `train.txt`.
3. **Quota miền**: nếu ảnh VN < `min_domain_share` luồng train thì tăng số lần lặp ảnh VN.
4. Lúc train: `copy_paste` dán instance class hiếm sang ảnh khác. Đây là lớp cân bằng cuối, xử lý phần lệch do
   class hiếm luôn đi kèm ô tô trong cùng ảnh.

**Mô phỏng trên pool lớn, lệch giống dữ liệu thật** (`scripts/simulate_balance.py`, 45.8k ảnh, tỉ lệ instance
mỗi ảnh là *giả định*, không phải số đo):

| | Ảnh chọn | Max/min | other_vehicle giữ lại | cyclist giữ lại | bicycle giữ lại |
|---|---:|---:|---:|---:|---:|
| Pool | 45,800 | 22.9 | 10,661 | 10,886 | 15,005 |
| Công thức v1 (phạt không trần) | 20,718 | 11.4 | 8,607 (81%) | 8,021 (74%) | 10,937 (73%) |
| **Công thức hiện tại** | 23,082 | 10.0 | **10,099 (95%)** | **9,810 (90%)** | **12,334 (82%)** |

Công thức hiện tại giữ được gần như toàn bộ instance class hiếm. Max/min không xuống thấp hơn nhiều vì xe máy
và ô tô luôn xuất hiện cùng class hiếm trong một ảnh. Phần lệch đó dành cho `copy_paste` lúc train.
Đổi lại công thức v1 để so sánh: `balance.scoring: legacy`.

## 7. Đầu ra & metadata

```
dataset_out/
├── data.yaml                      # đường dẫn tuyệt đối, dùng trên máy local
├── train.txt  val.txt  test.txt   # train.txt đã lặp RFS
├── images/{train,val,test}/<source>__<id>.jpg     # cạnh dài ≤ 1280
├── labels/{train,val,test}/<source>__<id>.txt     # YOLO-seg, 1 polygon / instance, không chồng nhau
├── stats/  REPORT.md  stats.json  *.png
└── metadata/
    ├── datasets_report.md              # ★ bảng cho slide: tên, link, license, số ảnh dùng, ánh xạ 11 lớp
    ├── images.jsonl                    # 1 dòng / ảnh: nguồn gốc, license, điều kiện, số lần lặp, nguồn nhãn
    ├── sources.json                    # registry + số ảnh/instance từng nguồn ở mọi giai đoạn
    ├── class_mapping_observed.json     # nhãn gốc → lớp đích → số instance thực tế
    ├── unselected_train_pool_uids.txt  # ảnh hợp lệ nhưng không được chọn
    ├── build_info.json                 # thời điểm build, config, log từng bước
    └── stats.json
du_lieu.zip                         # ★ cấu trúc BTC: images/, labels/, data.yaml (không có `path`), train/val/test.txt, metadata/, stats/
```

Một dòng trong `metadata/images.jsonl`:

```json
{
  "file": "images/train/mapillary_vistas_v2__abc123.jpg", "split": "train", "repeat": 2, "rfs_factor": 1.73,
  "source": "mapillary_vistas_v2", "source_name": "Mapillary Vistas v2.0", "tier": "C",
  "license": "CC BY-NC-SA 4.0 (Research edition)", "homepage": "https://www.mapillary.com/dataset/vistas",
  "orig_image": "D:/.../raw/mapillary/training/images/abc123.jpg", "group": "mapillary_vistas_v2/abc123",
  "domain": "global", "timeofday": "night", "weather": "rain",
  "orig_size": [4032, 3024], "export_size": [1280, 960],
  "counts": {"car": 6, "motorcyclist": 3, "sign": 4, "traffic_light": 1},
  "label_origin": {"human": 11, "merged": 3}, "teacher_completed": false, "needs_review": false,
  "phash": "c3d1...", "dup_cluster_size": 2, "dup_of": ["cityscapes/..."]
}
```

`label_origin`: `human` (người gán), `merged` (ghép rider), `rider_only`, `sam_box` (SAM từ box), `teacher` (kèm `conf`).

## 8. Thống kê trước / sau cân bằng

Bước `report` ghi `dataset_out/stats/REPORT.md` cho các giai đoạn **thô → pool sạch → pool train hợp lệ → đã chọn →
train hiệu dụng (sau RFS) → val → test**. Báo cáo gồm số ảnh, số instance, max/min, entropy, Gini; instance và tỉ lệ
theo class; bảng theo nguồn; tỉ lệ miền / ngày-đêm / thời tiết; 5 biểu đồ; log từng bước (rider được ghép, mask bị
cắt do chồng lấn, vật nhỏ bị bỏ, ảnh trùng, ảnh thiếu nhãn bị loại…).

> ⚠️ **Chưa có thống kê của dữ liệu thật.** Số liệu và biểu đồ dưới đây là từ **dữ liệu demo tổng hợp**
> (`scripts/make_demo_data.py`), chỉ để minh hoạ báo cáo và chứng minh pipeline chạy đúng.

| Giai đoạn (demo) | Ảnh | Instance | Max/min | Entropy ↑ | Gini ↓ | Ảnh VN |
|---|---:|---:|---:|---:|---:|---:|
| Pool sạch (trước cân bằng) | 660 | 4,905 | 24.04 | 0.891 | 0.395 | 24% |
| Train hiệu dụng (sau cân bằng) | 606 | 4,707 | 10.33 | 0.912 | 0.362 | 38% |

120 ảnh mô phỏng COCO (thiếu nhãn biển báo) bị loại đúng thiết kế vì chưa chạy teacher. Bước `resolve` đã cắt
855 mask bị che, bỏ 36 vật bị che hoàn toàn và 44 vật dưới ngưỡng 1.25%.

![Demo: tỉ lệ class](docs/images/demo_class_share_before_after.png)
![Demo: tỉ lệ nguồn](docs/images/demo_source_share_before_after.png)

## 9. Cài đặt & chạy

```powershell
pip install -r requirements.txt
```

**Chạy thử ngay, không cần tải dữ liệu:**

```powershell
python scripts/make_demo_data.py
```

```powershell
python scripts/run_pipeline.py --config configs/pipeline_demo.yaml --synthetic
```

```powershell
python scripts/simulate_balance.py
```

**Với dữ liệu thật:**

1. Xem danh sách nguồn và cách tải:
   ```powershell
   python scripts/download.py --list
   ```
2. Tải vào `raw/`. COCO và Roboflow có script; các nguồn khác in hướng dẫn:
   ```powershell
   python scripts/download.py coco2017_street
   ```
3. Cắt video VN thành khung hình (mỗi video một thư mục = một group chống rò rỉ):
   ```powershell
   python scripts/extract_frames.py --videos D:/dashcam --out raw/vn_dashcam_self/images --fps 1
   ```
4. Ảnh VN đã gán nhãn tay (CVAT/Label Studio xuất YOLO-seg) đặt vào `raw/vn_manual_labeled/{images,labels}/<lô>/`.
5. Trong [configs/sources.yaml](configs/sources.yaml): `enabled: true` cho nguồn đã có, **đối chiếu `label_map` với tên nhãn thật**.
6. Chuyển đổi và xem log ghép rider / tách chồng lấn:
   ```powershell
   python scripts/run_pipeline.py --config configs/pipeline.yaml --stages convert,resolve
   ```
7. (GPU) Đặt `sam_model` và `teacher_model` trong [configs/pipeline.yaml](configs/pipeline.yaml):
   ```powershell
   python scripts/run_pipeline.py --config configs/pipeline.yaml --stages sam,complete
   ```
8. Làm sạch, cân bằng, báo cáo, xuất `du_lieu.zip`:
   ```powershell
   python scripts/run_pipeline.py --config configs/pipeline.yaml --stages clean,balance,report,export
   ```
9. QA bằng mắt (ví dụ class 5 của Mapillary):
   ```powershell
   python scripts/visualize_labels.py --dataset dataset_out --source mapillary_vistas_v2 --cls 5 --n 12
   ```

Chỉ thử tham số cân bằng khác thì chạy lại `--stages balance,report,export`.

Vẽ lại sơ đồ:

```powershell
python scripts/draw_diagrams.py
```

## 10. Huấn luyện & nộp bài

1. Mở notebook mẫu của BTC, điền **`TEAM_ID = 15`**.
2. **Bước 1 (nạp dữ liệu): sửa để dùng `data.yaml` trong `du_lieu.zip`**, không chia train/val ngẫu nhiên lại.
   Chia ngẫu nhiên sẽ (a) đưa ảnh nước ngoài vào val, (b) làm mất lặp RFS, (c) để khung hình cùng video rơi vào
   cả train lẫn val.
3. Bước 2 (train), gợi ý trên GPU thuê:
   ```powershell
   yolo segment train model=yolo26n-seg.pt data=du_lieu/data.yaml imgsz=640 epochs=150 batch=64 copy_paste=0.4 mosaic=1.0 close_mosaic=15 cos_lr=True patience=40 seed=15
   ```
4. Bước 3 (bắt buộc): xuất ONNX bằng ô của notebook (fp32, imgsz 640), kiểm tra, đóng gói `submission.zip`.
5. Chọn checkpoint theo **mask mAP50-95 trên val VN**, đối chiếu với public leaderboard.

Đã kiểm tra: giải nén `du_lieu.zip` vào thư mục khác, Ultralytics đọc được `data.yaml` (train 606 dòng /
4,707 instance, val 48 ảnh trên bộ demo). Đã train thử 1 epoch trên bộ demo.

## 11. Cấu trúc thư mục

```
vn_traffic_balanced/
├── README.md
├── requirements.txt
├── configs/
│   ├── classes.yaml          # 11 lớp, ghép rider, ngưỡng 1.25%
│   ├── sources.yaml          # REGISTRY: nguồn gốc, license, cách đọc, ánh xạ nhãn
│   ├── pipeline.yaml         # tham số chạy thật (seed 15)
│   ├── sources_demo.yaml
│   └── pipeline_demo.yaml
├── vtb/
│   ├── common.py             # schema record, I/O, hình học, COCO RLE
│   ├── converters.py         # yolo_seg | yolo_det | coco | coco_bbox | coco_panoptic | polygons_json | bdd_scalabel | images_only
│   ├── resolve.py            # ghép rider, tách chồng lấn, lọc 1.25%
│   ├── labeling.py           # SAM box→mask, teacher bù class thiếu (GPU)
│   ├── quality.py            # pHash, ngày/đêm/thời tiết, loại trùng, chia tập theo group
│   ├── balance.py            # chọn ảnh greedy + RFS + quota miền
│   ├── stats.py              # thống kê, biểu đồ, REPORT.md
│   └── export.py             # YOLO-seg, metadata, datasets_report.md, du_lieu.zip
├── scripts/
│   ├── run_pipeline.py       # chạy toàn bộ / từng bước
│   ├── simulate_balance.py   # so sánh công thức cân bằng trên pool lớn mô phỏng
│   ├── download.py · extract_frames.py · visualize_labels.py
│   ├── make_demo_data.py     # dữ liệu tổng hợp để kiểm thử
│   └── draw_diagrams.py      # sinh docs/images/*.png
└── docs/images/
```

## 12. Việc phải kiểm tra & hạn chế

* **Tên nhãn gốc** trong `sources.yaml` lấy theo tài liệu công khai. Nhãn không khớp sẽ **bị bỏ âm thầm**; kiểm tra
  `metadata/class_mapping_observed.json` sau lần chạy đầu.
* **Đường dẫn thư mục** của BDD/ACDC/IDD có thể khác giữa các phiên bản, cần chỉnh `paths`.
* **Nhiễu nhãn so với guideline BTC** (ghi vào slide): bán tải và xe 9–15 chỗ trong COCO/BDD bị gán *truck*/*bus*;
  xe container nguyên chiếc bị gán *truck*, trong khi BTC tách đầu kéo và rơ-moóc.
* **Thứ tự độ sâu** ở `deoverlap` là heuristic (đáy thấp hơn = gần hơn); sai với vật treo cao hoặc trên cầu vượt.
* Quy tắc `@person` → rider dựa vào hình học; người ngồi trên xe đang dừng chống chân có thể bị coi là người dắt xe.
* **License**: phần lớn bộ ngoài là phi thương mại. Dùng được cho hackathon nội bộ, không phát hành lại.
  License của Zalo và VN-ViSeg chưa rõ.
* **Class 3** hiếm ở mọi bộ công khai; nguồn tốt nhất là ảnh VN tự thu (xích lô, ba gác, xe lam, rơ-moóc).
* Bước `sam` và `complete` viết theo API Ultralytics nhưng **chưa chạy thử** (cần weights). Các bước còn lại
  đã chạy end-to-end trên dữ liệu demo.

## 13. Tài liệu tham khảo

* Trang cuộc thi và guideline gán nhãn: [competition.antco.ai](https://competition.antco.ai/)
* Neuhold et al., *The Mapillary Vistas Dataset*, ICCV 2017 · [Vistas 2.0](https://blog.mapillary.com/update/2021/01/18/vistas-2-dataset.html)
* Varma et al., *IDD*, WACV 2019 · [inai.iiit.ac.in](https://inai.iiit.ac.in/domains/idd.html)
* Cordts et al., *The Cityscapes Dataset*, CVPR 2016
* Yu et al., *BDD100K*, CVPR 2020 · [arXiv 1805.04687](https://arxiv.org/abs/1805.04687)
* Lin et al., *Microsoft COCO*, ECCV 2014
* Sakaridis et al., *ACDC*, ICCV 2021 / T-PAMI 2025 · [arXiv 2104.13395](https://arxiv.org/abs/2104.13395)
* Gupta et al., *LVIS* (repeat-factor sampling), CVPR 2019
* Ghiasi et al., *Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation*, CVPR 2021
* [Ultralytics SAM 3](https://docs.ultralytics.com/models/sam-3) · [VN-Traffic-Instance-Segmentation-YOLO11](https://github.com/Tuanvinh1307/VN-Traffic-Instance-Segmentation-YOLO11) · [Zalo Traffic Sign 2020](https://datasetninja.com/zalo-traffic-sign)
