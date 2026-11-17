# IT701100 — Mô hình đồ thị có xác suất: Causal Bayesian Networks trên UCI In-Vehicle Coupon

Bài tập lớn HK1 2026–2027. Câu hỏi nhân quả: **đặt hạn coupon 2 giờ thay vì 1 ngày làm thay đổi xác suất tài xế chấp nhận coupon bao nhiêu?**
Ba mô hình được so sánh theo đúng khung của đề bài: B0 (BN rời rạc, cấu trúc học bằng Hill Climbing có ràng buộc tầng), H1 (thay CPD của Y bằng MLP), H2 (cấu trúc học bằng NOTEARS-MLP tự cài đặt).

## Sản phẩm nộp

| File | Nội dung |
|---|---|
| `notebooks/03_coupon_b0_h1_h2.ipynb` | **Notebook chính** (đã chạy, có output): mục lục đánh số 1–17, lý thuyết đi kèm code, mỗi mục kết quả có đoạn "Đọc kết quả" ngay sau output, Table A/B/C, DoWhy, ablation, kết luận |
| `scripts/nb_coupon_src.py` | Mã nguồn của notebook ở dạng `.py` có đánh dấu cell (`# %%`), dùng để sinh lại notebook |
| `scripts/build_notebook.py` | Sinh `.ipynb` từ file nguồn |
| `artifacts/coupon_results/` | Output đã lưu (81 file, xem bảng dưới) |
| `research-plan-coupon.md` | Kế hoạch thực nghiệm (estimand, tầng biến, lịch chạy) |
| `docs/dataset-scouting/` | Hồ sơ khảo sát dữ liệu |

### Các file output chính trong `artifacts/coupon_results/`

| Nhóm | File |
|---|---|
| Cấu hình, split | `config.json`, `splits.json`, `states.json`, `structures.csv` |
| Cấu trúc | `b0_candidates_seed*.csv`, `b0_edges_seed*.json`, `h2_edges_seed*.json`, `h2_adjacency_seed*.csv`, `h2_thresholds_seed*.csv`, `h2_l1_scale_selection.csv`, `dag_b0_seed0.png`, `dag_h2_seed0.png` |
| Tham số | `cpd_y_B0_seed*.csv`, `cpd_y_H1_seed*.csv`, `cpd_y_H2_seed*.csv`, `h1_seed*.pt` |
| Table A/B/C | `table_A_runs.csv`, `table_A_reference.csv`, `table_B_runs.csv`, `table_C.csv`, `runs_raw.csv` |
| DoWhy | `identification_b0_seed{0–4}.txt`, `identification_h2_seed{0–4}.txt`, `backdoor_check.csv`, `dowhy_refuters.csv`, `bn_refuters.csv` |
| Hiệu ứng không đồng nhất | `cate_by_coupon.csv`, `cate_ci_by_coupon.csv`, `cate_ci_seed0.png`, `cell_effects_b0_seed0.csv` |
| Ablation, độ nhạy | `ablation_A2.csv`, `ablation_A5.csv`, `ablations_A1_A3_H1H2.csv`, `ablation_monotone_logit.csv`, `ablation_monotone_mlp.csv` |
| Tổng hợp | `ate_all_estimators.csv`, `ate_forest.png` |

## Chạy lại

Yêu cầu Python **3.13** (dowhy 0.14 chưa chạy trên 3.14; pgmpy 1.1 cần ≥ 3.10).

```bash
python3.13 -m venv .venv313
```

```bash
.venv313/bin/pip install -r requirements.txt
```

```bash
.venv313/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/03_coupon_b0_h1_h2.ipynb
```

Dữ liệu thô `data/raw/coupon/in-vehicle-coupon-recommendation.csv` được tải tự động từ UCI nếu chưa có (sha256 được kiểm tra trong notebook).
Trên Google Colab: chỉ cần mở file `notebooks/03_coupon_b0_h1_h2.ipynb` (File → Upload notebook); cell đầu tiên tự `pip install`, dữ liệu tải từ UCI và output ghi vào `/content/pgm_coupon/`. Không cần tải thêm file nào. Lệnh cài của cell đó đã được kiểm tra ngày 2026-11-09 trong venv sạch: giải ra đúng bộ phiên bản đã kiểm chứng (pgmpy 1.1.2, dowhy 0.14, torch 2.15, pandas 3.1) và chạy thử không lỗi.

Thời gian chạy toàn bộ (5 seed, bootstrap 200 lần cho 3 mô hình, refuter DoWhy 100 lần): khoảng 30 phút trên CPU Apple M-series 8 lõi; phần chậm nhất là bootstrap của H1 (huấn luyện lại MLP 200 lần mỗi seed) và 3 lần NOTEARS mỗi seed; số phút thay đổi theo tải máy.

Kiểm tra tái lập (2026-11-09): làm đúng ba bước trên trong một thư mục sạch (chỉ có `notebooks/`, `scripts/`, `requirements.txt`; dữ liệu tự tải từ UCI) cho toàn bộ notebook (bản 119 cell lúc đó) chạy không lỗi trong 29 phút, và split, cấu trúc DAG, Table A/B, CI bootstrap giống hệt bản nộp; các cell thêm hoặc sửa sau đó (bản 126 cell hiện tại) chỉ dùng lại các hàm đã có và được đối chiếu với output của bản sạch sau mỗi lần chạy lại (chênh lệch 0 ở mọi chỉ số chung).

Để sửa notebook: sửa `scripts/nb_coupon_src.py` rồi chạy

```bash
.venv313/bin/python scripts/build_notebook.py scripts/nb_coupon_src.py notebooks/03_coupon_b0_h1_h2.ipynb
```

## Bản rút gọn, dễ đọc (notebook 04)

Cùng bài toán, cùng dữ liệu, cùng cách chia và cùng seed với notebook 03, nhưng **chỉ gồm phần đề bài bắt buộc** (B0, H1, H2, DoWhy, Bảng A/B/C) và được viết cho người mới: mỗi mục theo khuôn *Mục tiêu → Ý tưởng → code có chú thích → Cách đọc kết quả*, có bảng thuật ngữ và một mục tính tay bằng pandas trước khi dùng mô hình.

| File | Nội dung |
|---|---|
| `notebooks/04_coupon_simple.ipynb` | Notebook đã chạy, có output: mục lục đánh số 1–15, 70 cell |
| `scripts/nb_coupon_simple_src.py` | Mã nguồn của notebook 04 (sửa file này, không sửa `.ipynb`) |
| `artifacts/coupon_simple/` | Output của notebook 04 (80 file): `config.json`, `splits.json`, `runs_raw.csv`, `table_A*.csv`, `table_B*.csv`, `table_C*.csv`, `summary.csv`, cạnh DAG, bảng xác suất của Y, checkpoint H1, ma trận H2, văn bản nhận diện của DoWhy, 6 hình |

Chạy lại (khoảng 6 phút trên CPU Apple M-series 8 lõi, đo ngày 2026-11-10):

```bash
.venv313/bin/python scripts/build_notebook.py scripts/nb_coupon_simple_src.py notebooks/04_coupon_simple.ipynb
```

```bash
.venv313/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/04_coupon_simple.ipynb
```

Chạy thử nhanh khoảng 1 phút (2 seed, rất ít lần lặp, số liệu chỉ để kiểm tra code): thêm `COUPON_QUICK=1` trước lệnh `nbconvert`. Chế độ này ghi đè `artifacts/coupon_simple/`, nên sau đó phải chạy lại bản đầy đủ.

Kết quả chính của notebook 04 (trung bình 5 seed): ATE = −0,164 (B0), −0,163 (H1), −0,156 (H2), cả 15 khoảng tin cậy 95% nằm dưới 0; AUROC 0,739 / 0,742 / 0,736. Đối chiếu với notebook 03 (so `runs_raw.csv` của hai bản, ngày 2026-11-10):

- Split của cả 5 seed giống hệt.
- B0 và H1: AUROC, log loss và ATE **trùng** notebook 03 ở mọi seed. Khoảng tin cậy khác ở chữ số thứ ba vì bootstrap 100 lần thay vì 200.
- H2: trùng ở seed 0–3. Ở seed 4, NOTEARS viết lại cho độ mạnh cạnh hơi khác quanh ngưỡng, nên Y chỉ còn hai cha ngoài T (`coupon`, `habit_match`) thay vì bốn: AUROC 0,729 so với 0,745, ATE −0,155 so với −0,153. Vì vậy trung bình của H2 lệch nhẹ (AUROC 0,736 so với 0,740; ATE −0,156 ở cả hai bản).

Nộp notebook nào thì báo cáo và slide phải lấy số từ notebook đó.

Khác notebook 03: không có ablation (A1–A5), không có CATE theo loại quán, không có lưới hệ số phạt L1 của H2 (dùng cố định 0,1, lấy từ kết quả của notebook 03), không có các biến thể đơn điệu. Notebook 04 chưa được chạy thử trên Google Colab và chưa được kiểm tra tái lập trong môi trường sạch.

## Bản không dùng `habit_match` (notebook 05)

Bản sao của notebook 04, chỉ đổi cách đưa thói quen vào mô hình: bỏ biến ghép `habit_match` (được tạo bằng cách chọn cột thói quen **theo loại coupon**, nên khó giải thích vị trí của nó trong DAG) và dùng thẳng 5 cột thói quen gốc `Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20`, `Restaurant20To50`, mỗi cột gộp thành `low` / `mid` / `high` / `missing`. Mạng có 18 nút (bản 04: 14), 105 cạnh cấm (bản 04: 65). Dòng thiếu thói quen được giữ lại, nên dữ liệu là 12.197 dòng của 587 người (bản 04: 12.037 dòng, 580 người).

| File | Nội dung |
|---|---|
| `notebooks/05_coupon_original_habits.ipynb` | Notebook đã chạy, có output: 70 cell, cùng mục lục 1–15 với notebook 04 |
| `scripts/nb_coupon_original_habits_src.py` | Mã nguồn của notebook 05, khớp với `.ipynb` (đã kiểm bằng `build_notebook.py` ngày 2026-11-12) |
| `artifacts/coupon_original_habits/` | Output của notebook 05 (81 file), cùng cấu trúc với `artifacts/coupon_simple/` và thêm `overlap_by_seed.csv` |

Chạy lại (khoảng 12 phút trên CPU Apple M-series 8 lõi, đo ngày 2026-11-12; một lượt chạy trước đó cùng ngày mất 23 phút):

```bash
.venv313/bin/python scripts/build_notebook.py scripts/nb_coupon_original_habits_src.py notebooks/05_coupon_original_habits.ipynb
```

```bash
.venv313/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/05_coupon_original_habits.ipynb
```

`COUPON_QUICK=1` dùng được như ở notebook 04 và sẽ ghi đè `artifacts/coupon_original_habits/`; thời gian của chế độ này chưa được đo trên notebook 05.

Kết quả chính của notebook 05 (trung bình 5 seed): ATE = −0,157 (B0), −0,155 (H1), −0,150 (H2), cả 15 khoảng tin cậy 95% nằm dưới 0; AUROC 0,726 / 0,731 / 0,727; ước lượng tham chiếu của DoWhy −0,162 và −0,164. So với notebook 04:

- Kết luận không đổi: ATE lệch 0,006–0,008, nhỏ hơn độ rộng khoảng ±0,02 của khoảng tin cậy. AUROC thấp hơn khoảng 0,01.
- Trong 5 thói quen gốc, chỉ `CoffeeHouse` trở thành cha của Y, ở cả 15 mô hình. Mũi tên `habit_match → coupon` (có ở cả 15 DAG của bản 04) không còn; `Bar → coupon` chỉ xuất hiện ở DAG của B0 tại seed 0 và 1.
- H2 kém ổn định hơn bản 04: ở seed 0, 2, 4 nó cho Y tới 4–5 cha ngoài T, bảng xác suất của Y lên tới 1.440 ô và 15–19% dòng TEST phải dùng ô TRAIN trống khi can thiệp (B0, H1: không quá 2,9%). ATE của H2 ở seed 0 là −0,130, gần 0 hơn hai mô hình kia.
- Đây không phải thí nghiệm chỉ đổi một yếu tố: quần thể và các split khác bản 04 vì giữ thêm 160 dòng.

Những điểm cần biết khi dùng notebook 05:

- Notebook được chạy đầy đủ hai lần ngày 2026-11-12. Cha của Y và của T, các chỉ số dự đoán, ATE và khoảng tin cậy của ba mô hình trùng nhau ở cả 15 dòng. Có hai chỗ khác: DoWhy chọn tập điều chỉnh khác ở 5/10 trường hợp (ước lượng tham chiếu của DoWhy lệch tối đa 0,008), và DAG của B0 ở seed 3 khác chiều 9 cạnh và thêm 1 cạnh, đều không đi vào T hay Y. Output và `artifacts/` hiện tại là của lượt chạy thứ hai.
- Ô chạy chính đặt `torch.set_num_threads(1)`. PC ở mục 8.1 chỉ là minh hoạ, giới hạn tập điều kiện tối đa 2 biến (`max_cond_vars=2`), không tham gia chọn mô hình.
- Hệ số phạt L1 cho các cạnh đi vào T của H2 (0,1) được dùng lại từ notebook 03, chưa dò lại trên bộ 18 biến.
- Chưa chạy thử trên Google Colab và chưa kiểm tra trong môi trường sạch.

## Bản có bộ luật DAG chặt hơn (notebook 06)

Bản sao của notebook 04, chỉ đổi **bộ ràng buộc DAG** ở mục 6: thay cho 4 tầng (cấm 65/182 mũi tên), mỗi nút có một danh sách parents được phép (cấm 127/182 mũi tên), ví dụ `weather`, `age_bin`, `gender` là nút gốc, không nhận mũi tên nào. Mục 6 có thêm ô in **bảng lý do của từng luật** (`RULES_TABLE`, lưu ra `rules_table.csv`) và ô kiểm lại bằng Cramér's V.

| File | Nội dung |
|---|---|
| `notebooks/06_coupon_rules.ipynb` | Notebook đã chạy đủ 5 seed ngày 2026-11-13, có output: 74 cell |
| `scripts/nb_coupon_rules_src.py` | Mã nguồn của notebook 06, khớp với `.ipynb` (đã kiểm bằng `parse_cells` của `build_notebook.py` ngày 2026-11-13) |
| `artifacts/coupon_rules/` | Output của notebook 06 (81 file), cùng cấu trúc với `artifacts/coupon_simple/` và thêm `rules_table.csv` |

Chạy lại:

```bash
.venv313/bin/python scripts/build_notebook.py scripts/nb_coupon_rules_src.py notebooks/06_coupon_rules.ipynb
```

```bash
.venv313/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/06_coupon_rules.ipynb
```

Kết quả chính của notebook 06 (trung bình 5 seed): ATE = −0,164 (B0), −0,163 (H1), −0,160 (H2), cả 15 khoảng tin cậy 95% nằm dưới 0; AUROC 0,739 / 0,742 / 0,740. So với notebook 04 (so `runs_raw.csv` và `edges_*.json` của hai bản, ngày 2026-11-13):

- Không còn cạnh vô lý: theo bộ luật mới, DAG của bản 04 có trung bình 13,8 cạnh sai trên 40,8 (B0) và 7,4 trên 30,2 (H2), ví dụ `passenger → weather`, `has_children → age_bin`, `habit_match → coupon`.
- B0 và H1: parents của T và của Y giống hệt bản 04 ở cả 5 seed, nên các chỉ số dự đoán, ATE và khoảng tin cậy trùng nhau.
- H2: parents của Y đổi ở 2/5 seed (seed 2 bỏ `weather`, seed 4 thêm `destination`); ATE −0,156 → −0,160, AUROC 0,736 → 0,740, độ lệch chuẩn của ATE qua các seed 0,014 → 0,016.
- Mạng mô tả cả 14 biến kém đi một chút với B0 (log-likelihood trên VALIDATION ở seed 0: −9,95 → −10,25).

Những điểm cần biết khi dùng notebook 06:

- `h2_h_tol` đổi từ 1e-8 sang 1e-5: với mask chặt hơn, mức 1e-8 làm NOTEARS hỏng số (NaN).
- Thời gian ghi trong notebook (5 seed hết khoảng 21 phút; H2 khoảng 84 giây mỗi seed) đo khi máy đang bận việc khác, không so được với 6 phút của bản 04. Chế độ `COUPON_QUICK=1` dùng được và sẽ ghi đè `artifacts/coupon_rules/`.
- Notebook 04 hiện không chạy lại được nguyên trạng: ô `group_table` ở mục 5.2 đang đổi tên dở (`accept_1d`/`accept_2h` lẫn với `acc_1d`/`acc_2h`). Bản 06 đã dùng lại tên `acc_1d`/`acc_2h`.
- Sau lần chạy, chỉ các ô markdown được sửa cho khớp số liệu; không ô code nào được chạy lại.

## Bản không dùng `habit_match`, có bộ luật DAG chặt hơn (notebook 07)

Ghép notebook 05 và 06: dữ liệu 18 biến với 5 thói quen gốc (như notebook 05), và bộ luật viết cho từng nút kèm bảng lý do (như notebook 06). Bộ luật được viết lại cho 18 nút: cấm 212/306 mũi tên (notebook 05 cấm 105/306). Vì thói quen gốc là đặc điểm có trước cuộc khảo sát, không thói quen nào nhận mũi tên từ `coupon` hay từ biến kịch bản; cạnh duy nhất được phép đi từ nhóm người trả lời sang nhóm kịch bản là `has_children → passenger`.

| File | Nội dung |
|---|---|
| `notebooks/07_coupon_rules_original_habits.ipynb` | Notebook đã chạy đủ 5 seed ngày 2026-11-13, có output: 74 cell |
| `scripts/nb_coupon_rules_original_habits_src.py` | Mã nguồn của notebook 07, khớp với `.ipynb` (đã kiểm bằng `parse_cells` của `build_notebook.py` ngày 2026-11-13) |
| `artifacts/coupon_rules_original_habits/` | Output của notebook 07 (82 file), cùng cấu trúc với `artifacts/coupon_original_habits/` và thêm `rules_table.csv` |

Chạy lại:

```bash
.venv313/bin/python scripts/build_notebook.py scripts/nb_coupon_rules_original_habits_src.py notebooks/07_coupon_rules_original_habits.ipynb
```

```bash
.venv313/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/07_coupon_rules_original_habits.ipynb
```

Kết quả chính của notebook 07 (trung bình 5 seed): ATE = −0,157 (B0), −0,155 (H1), −0,148 (H2), cả 15 khoảng tin cậy 95% nằm dưới 0; AUROC 0,726 / 0,731 / 0,733; ATE tham chiếu của DoWhy −0,164 và −0,165; 10/10 refuter đạt.

So với notebook 05 (cùng dữ liệu, cùng split, luật 4 tầng; so `runs_raw.csv` và `edges_*.json` ngày 2026-11-13):

- B0 và H1: cấu hình được chọn, parents của Y và của T, các chỉ số dự đoán, ATE và khoảng tin cậy trùng bản 05 ở cả 5 seed. DAG của B0 ở bản 05 có trung bình 18,6 cạnh trái luật mới trên 58,6 (ví dụ `Bar → age_bin`, `passenger → weather`, `Bar → coupon`); phần lớn được thay bằng cạnh cùng cặp biến nhưng đảo chiều.
- H2: Y có cùng một parent set ở cả 5 seed (`T`, `coupon`, `CoffeeHouse`, `destination`, `weather`), bản 05 có 4 parent set khác nhau. ATE −0,150 → −0,148, AUROC 0,727 → 0,733, log loss 0,628 → 0,617, độ lệch chuẩn của ATE 0,013 → 0,009. Tỉ lệ dòng TEST phải dùng ô TRAIN trống tăng từ 10,7% lên 14,3%.
- `CoffeeHouse` vẫn là thói quen duy nhất làm parent của Y, ở cả 15 mô hình.

So với notebook 06 (cùng kiểu luật, có `habit_match`, dữ liệu ít hơn 160 dòng): ATE lệch 0,007–0,013, AUROC thấp hơn 0,007–0,014; không còn cạnh `coupon → habit_match`.

Những điểm cần biết khi dùng notebook 07:

- `h2_h_tol = 1e-5` như notebook 06 (bản 05 dùng 1e-8). Thay đổi của H2 so với bản 05 vì thế có thể đến từ bộ luật, từ ngưỡng này, hoặc cả hai; chưa tách riêng.
- Lượt chạy đầy đủ mất khoảng 10 phút rưỡi khi máy đang bận việc khác; mới có một lượt chạy đầy đủ, chưa kiểm tra tái lập cho H2 và DoWhy. Chế độ `COUPON_QUICK=1` dùng được và sẽ ghi đè `artifacts/coupon_rules_original_habits/`.
- Sau lần chạy, chỉ các ô markdown được sửa cho khớp số liệu; không ô code nào được chạy lại.

## Các notebook khác (giai đoạn trước)

- `notebooks/00_phase0_setup_tutoring.ipynb`: kiểm tra môi trường, tutorial pgmpy/dowhy.
- `notebooks/01_bank_marketing_cbn.ipynb`: thử nghiệm ban đầu trên Bank Marketing (đã bỏ, xem `research-plan-bank.md`).
- `notebooks/02_coupon_cbn.ipynb`: thăm dò dữ liệu coupon và B0 sơ bộ.
