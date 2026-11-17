# Kế hoạch thực nghiệm — Causal Bayesian Networks trên UCI 603 In-Vehicle Coupon

Môn: IT701100 Mô hình đồ thị có xác suất, HK1 2026–2027 (GV: TS. Nguyễn Chí Thiện). Lập ngày 2026-11-06. Thay thế kế hoạch Bank Marketing (đã lưu trữ thành `research-plan-bank.md`); phần hạ tầng (notebook 00, `fit_discrete_bn`, mẫu do-query, Table A) giữ nguyên.

Hồ sơ dataset và số liệu đã kiểm chứng: `docs/dataset-scouting/05_uci603_coupon_profile.md`. Mọi con số trong kế hoạch này đã đếm trên `data.csv` của UCI (12.684 dòng).

## 0. Câu hỏi và estimand

**Câu hỏi một câu.** Với một người lái xe trong một bối cảnh lái xe cụ thể (đích đến, người đi cùng, thời điểm, thời tiết) được hệ gợi ý trong xe đề xuất một coupon giảm giá 20%, nếu coupon được đặt hạn 2 giờ thay vì 1 ngày thì xác suất người đó chấp nhận coupon thay đổi bao nhiêu?

- **Population:** các cặp (người trả lời, kịch bản) trong khảo sát MTurk của Wang et al. 2017, loại ô Bar × Work vì không có overlap (487 dòng) và dòng thiếu tần suất đến đúng loại quán của coupon (160 dòng). Còn **12.037 dòng, 580 người** (đã chạy, xem mục 11).
- **T:** `expiration`, control = 1d (T=0), treatment = 2h (T=1). Thời điểm can thiệp: lúc coupon hiển thị, do nhà phát coupon đặt.
- **Y:** `Y` = 1 nếu "đi ngay" hoặc "đi sau, trước khi hết hạn"; 0 nếu từ chối.
- **Estimand:** ATE = P(Y=1 | do(T=1)) − P(Y=1 | do(T=0)) trên TEST reference cohort, chuẩn hoá theo phân phối baseline C của TEST. Báo cáo thêm CATE theo `coupon` (5 giá trị) vì hiệu ứng kỳ vọng khác nhau rõ theo loại quán.
- **Mốc sơ bộ (toàn dữ liệu, chưa split):** chênh thô −14,9 điểm %; chuẩn hoá theo (coupon, destination) −16,3 điểm %; CATE theo coupon: Restaurant(<20) −24,2, Restaurant(20–50) −24,7, Coffee House −15,2, Carry out −12,2, Bar −3,0 điểm %.

**Giả định nền (ghi vào bảng giả định).**
1. T được gán theo phiên bản bảng hỏi, độc lập với người trả lời nhưng phụ thuộc kịch bản. Đã kiểm trên dữ liệu đủ điều kiện: biên độ P(T=1) giữa các mức của biến người trả lời ≤ 0,036 (gender 0,000); của biến kịch bản lên tới 0,254 (coupon), 0,246 (weather), 0,142 (time_bin). Vì vậy backdoor chỉ đi qua biến kịch bản, và **weather cũng là confounder theo thiết kế**, không chỉ là C phụ.
2. Không có confounder ẩn giữa T và Y sau khi điều chỉnh biến kịch bản. Hợp lý vì T là tham số thiết kế, nhưng phiên bản bảng hỏi không có trong dữ liệu nên vẫn phải phát biểu là giả định.
3. Y là ý định khai báo trong tình huống giả định. Kết quả nói về hành vi khai báo, không suy ra hành vi thật.
4. Các dòng của cùng một người không độc lập; xử lý bằng group split, không bằng mô hình.

## 1. Từ điển biến và vai trò

| Biến | Mức sau tiền xử lý | Timing | Vai trò trong DAG |
|---|---|---|---|
| `coupon` | 5 | thiết kế kịch bản, trước T | **C** (→T, →Y) |
| `destination` | 3 | thiết kế, trước T | **C** (→T, →Y) |
| `time` → `time_bin` | 3 (morning 7–10AM, afternoon 2PM, evening 6–10PM) | thiết kế, trước T | C_ext (→T, →Y; có ô trống) |
| `passenger` | 4 | thiết kế, trước T | C phụ (→Y; →T yếu, biên độ 0,105) |
| `weather` | 3 | thiết kế, trước T | C_ext (→T mạnh, biên độ 0,246; →Y; có ô trống) |
| `temperature` | bỏ | trùng weather gần hoàn toàn | không dùng |
| `toCoupon_GEQ15min`, `toCoupon_GEQ25min` → `dist` | 3 (<15, 15–25, 25+) | thiết kế, trước T | →Y (không →T) |
| `direction_same` | 2 | thiết kế, trước T | →Y |
| `direction_opp`, `toCoupon_GEQ5min` | bỏ | thừa, hằng số | không dùng |
| `expiration` → **T** | 2 | — | treatment |
| `Y` | 2 | sau T | outcome |
| `Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20`, `Restaurant20To50` → `habit_match` | 3 (never/less1, 1–3, 4+) | người trả lời, trước T | →Y mạnh; **cấm →T** |
| `age` → `age_bin` | 4 (<26, 26–35, 36–50, 50+) | người trả lời | →Y; cấm →T |
| `income` → `income_bin` | 3 (<37.5k, 37.5–75k, 75k+) | người trả lời | →Y; cấm →T |
| `gender`, `has_children` | 2, 2 | người trả lời | →Y; cấm →T |
| `maritalStatus`, `education`, `occupation` | gộp hoặc bỏ | người trả lời | chỉ dùng ở ablation; cấm →T |
| `car` | bỏ | 99% thiếu | không dùng |

`habit_match` = tần suất đến đúng loại quán của coupon đang hiển thị (lấy `Bar` nếu coupon là Bar, `CoffeeHouse` nếu Coffee House, ...). Đây là biến tổng hợp quan trọng nhất theo bài gốc, và giúp DAG gọn thay vì 5 nút thói quen.

DAG mục tiêu 11–12 nút: coupon, destination, time_bin, passenger, weather, dist, direction_same, T, habit_match, age_bin, income_bin, gender (has_children tuỳ chọn), Y.

## 2. Tiền xử lý và split (Phase 2, 0,5 tuần)

1. Tải `data.csv` từ UCI, lưu `data/raw/coupon/` với checksum SHA-256, ngày tải, DOI 10.24432/C5GS4P, license CC BY 4.0.
2. Eligibility flow: 12.684 → loại Bar × Work (487) → loại dòng thiếu `habit_match` (160) → **12.037**. 73 dòng trùng hoàn toàn được giữ (có thể là hai người cùng hồ sơ) và chỉ ghi nhận. Báo cáo số dòng trước và sau mỗi bước.
3. Tái tạo `resp_id`: gom các dòng liên tiếp có cùng 13 cột người trả lời. Kết quả 587 block (517 block đúng 22 dòng, 12 block 6 dòng, 4 block 34 dòng), 580 người sau eligibility. Ghi phân phối kích thước block vào data audit.
4. Mã hoá theo bảng mục 1. Thiếu ở 5 cột thói quen chỉ ảnh hưởng khi đúng cột khớp với coupon bị thiếu (160 dòng): bỏ dòng.
5. **Split theo `resp_id`**, 70/15/15, 5 seed (0–4), stratify theo tertile tỉ lệ Y trung bình của người. Mỗi seed: 405 / 87 / 88 người, khoảng 8.400 / 1.820 / 1.820 dòng. Chỉ số `row_id` lưu ở `artifacts/coupon/splits.json`.
6. Overlap trên TRAIN (đã chạy 5 seed): T × (coupon, destination) có 14 ô, nhánh nhỏ nhất 89–102 dòng, không ô nào dưới 30. T × (coupon, destination, time_bin, weather) có 54 ô, **27 ô trống** ở mọi seed (Bar × No Urgent × afternoon, Bar × Snowy, Carry out × Snowy, Restaurant(<20) × Rainy...).
7. Khoá tập C: **C_core = {coupon, destination}** (identify, positivity đầy đủ); **C_ext = C_core + {time_bin, weather}** (độ nhạy; cả hai →T mạnh nhưng có ô trống); C_full = C_ext + {passenger, dist, direction_same, habit_match, age_bin, income_bin, gender, has_children} (dự đoán, CATE, reference cohort). TEST reference cohort = phân phối C_full trên TEST, giới hạn về các profile có overlap trên TRAIN nếu do-query điều kiện theo C_ext.

## 3. B0 — DAG truyền thống + CPD rời rạc (Phase 3, 1,5 tuần)

- Học cấu trúc trên TRAIN bằng `HillClimbSearch` với `scoring_method="bic-d"` và `ExpertKnowledge`:
  - `temporal_order`: tầng 1 = biến người trả lời (habit_match, age_bin, income_bin, gender); tầng 2 = biến kịch bản (coupon, destination, time_bin, passenger, weather, dist, direction_same); tầng 3 = T; tầng 4 = Y.
  - `forbidden_edges`: mọi cạnh ra từ Y; mọi cạnh từ tầng 1 vào T (T độc lập với người trả lời theo thiết kế); mọi cạnh từ T về tầng 1–2.
  - `required_edges`: T → Y; coupon → Y; habit_match → Y. Chạy thêm một phiên bản không có required_edges để báo cáo độ nhạy.
- So sánh với PC (`pgmpy.estimators.PC`) trên cùng ràng buộc; chọn DAG trên VALIDATION theo log-likelihood, cố định trước TEST.
- CPD: `BayesianEstimator` BDeu, ESS = 5 (dùng `fit_discrete_bn()` sẵn có).
- Identify bằng DoWhy 0.14: nạp DAG, `identify_effect()`, ghi adjustment set tìm được. Kỳ vọng tập backdoor ⊆ biến kịch bản.
- Do-query trên TEST: cắt cạnh vào T, đặt T = 0/1, tính E[Y | do(T), C_full = c] rồi trung bình theo cohort TEST. Báo cáo ATE và CATE theo coupon.
- Dự đoán: P(Y | mọi parent) trên TEST → AUC, Brier, log-loss.
- **Kiểm tra ngoài cho B0:** đếm số cạnh từ tầng 1 vào T mà thuật toán đề xuất khi bỏ forbidden_edges. Kỳ vọng 0. Đây là bằng chứng thiết kế quasi-randomized.

## 4. H1 — MLP thay CPD của Y (Phase 4, 1 tuần)

- Giữ DAG của B0. Input = one-hot của parents(Y) (khoảng 8–10 biến rời rạc, ~35 chiều). MLP 2 lớp ẩn 32–16, dropout 0,2, BCE, Adam, early stopping trên VALIDATION.
- Liệt kê mọi tổ hợp parent (ước lượng 2 × 5 × 3 × 3 × 4 × 3 × 3 × 2 × 3 ≈ 19k tổ hợp nếu Y có 9 parent; vẫn liệt kê được), tính P(Y=1 | parents) từ MLP → bảng CPD → chèn vào bản sao của B0 → `check_model()`. Do-query và dự đoán chạy y hệt B0.
- Lý do H1 có việc: với C_full, TRAIN chỉ khoảng 8,5k dòng cho hàng nghìn ô, BDeu sẽ co về prior ở các ô thưa, còn MLP chia sẻ thống kê giữa các ô. Kỳ vọng H1 thắng rõ ở CATE theo coupon và ở log-loss.
- Ghi runtime và số tham số.

## 5. H2 — NOTEARS nonlinear với mask (Phase 5, 1,5 tuần)

- Biểu diễn mỗi nút rời rạc bằng one-hot (hoặc embedding) và chạy NOTEARS-MLP trên TRAIN; mask M từ ràng buộc mục 3 (M_ij = 0 với cạnh cấm), W_eff = M ⊙ W.
- Chọn λ và ngưỡng cạnh trên VALIDATION; quy tắc sửa chu trình: bỏ cạnh yếu nhất trong mỗi chu trình còn lại.
- Re-identify bằng DoWhy trên DAG của H2; fit CPD bằng cùng BDeu của B0; do-query và dự đoán như B0.
- **Kiểm tra ngoài cho H2:** (a) có khôi phục T → Y không; (b) có cạnh nào từ tầng 1 vào T khi nới mask không; (c) SHD giữa DAG H2 và DAG B0.
- Phương án dự phòng nếu NOTEARS không hội tụ trên biến categorical: chạy trên 7 nút đã gộp thành ordinal (habit_match, dist, age_bin, income_bin, time_bin, T, Y) và ghi rõ giới hạn. Không đổi sang DAGMA/GOLEM.

## 6. Đánh giá (Phase 6, 1 tuần)

- **Table A (dự đoán):** AUC, Brier, log-loss trên TEST cho B0/H1/H2, trung bình ± sd qua 5 seed, kèm paired difference so với B0.
- **Table B (can thiệp):** cohort ID và kích thước, T/C value, adjustment set đã identify, E[Y|do(0)], E[Y|do(1)], ATE với bootstrap CI (resample TRAIN theo `resp_id`, 200 lần, TEST cohort cố định), CATE theo coupon.
- **Table C (robustness):** DoWhy `placebo_treatment_refuter` (permute T, kỳ vọng ATE ≈ 0) và `data_subset_refuter` (80%, 100 sim); chạy trên B0 và H2; với H1 ghi rõ refuter chạy qua BN đã chèn CPD từ MLP, có retrain riêng cho placebo.
- **So sánh với ước lượng độc lập:** DoWhy backdoor stratified với C_core, và chênh lệch thô. Khoảng cách giữa chênh thô (−14,9) và ATE điều chỉnh (−16,3 sơ bộ) nhỏ hơn Bank Marketing; nêu rõ đây là hệ quả của T gần như ngẫu nhiên, chứ không phải hạn chế.
- **Stability:** phương sai ATE giữa 5 seed; số cạnh thay đổi giữa seed (Jaccard).
- Runtime và phần cứng (CPU, Python 3.13, pgmpy 1.1.2, DoWhy 0.14, torch 2.15).

## 7. Viết (Phase 7, 1 tuần)

Báo cáo 2.1–2.9 theo slide 76–78; slide 6 bước demo; notebook `notebooks/02_coupon_cbn.ipynb` chạy sạch từ đầu; AI disclosure; phụ lục: variable dictionary, eligibility flow, bảng giả định, hai sơ đồ DAG (B0, H2) với cạnh T → Y tô đậm.

## 8. Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Overlap yếu khi điều chỉnh nhiều biến kịch bản | ô với < 10 dòng mỗi nhánh | Giữ C_core cho identify; biến kịch bản còn lại chỉ dùng cho dự đoán và CATE |
| Block người trả lời tái tạo sai | block 6 hoặc 34 dòng | Giữ nguyên cách gom; ghi phân phối; thử split theo block ±1 để xem ATE có đổi không |
| H1 ≈ B0 vì dữ liệu đủ dày | log-loss khác < 0,005 | Báo cáo trung thực; nhấn mạnh CATE theo coupon nơi H1 khác biệt |
| NOTEARS trên one-hot ra đồ thị dày hoặc rỗng | > 25 cạnh hoặc không có T → Y | Quét λ rộng hơn trên VALIDATION; fallback 7 nút ordinal |
| Y là ý định giả định | — | Nêu ở 2.1 và 2.9; không viết "khách hàng sẽ đến quán" |
| Khác biệt phụ lục bài báo và dữ liệu (time 3 so với 5 mức) | — | Ghi trong data audit, dùng dữ liệu làm chuẩn |

## 9. Lịch (bắt đầu 2026-11-10)

| Tuần | Việc | Sản phẩm |
|---|---|---|
| 1 (10–16/11) | Phase 1–2: audit, resp_id, mã hoá, split, overlap | `data/raw/coupon/`, `artifacts/splits_coupon.json`, mục 1–2 notebook |
| 2–3 (17–30/11) | Phase 3: B0 trên 5 seed, DoWhy identify, do-query, Table A/B sơ bộ | DAG B0, `artifacts/b0_coupon/` |
| 4 (1–7/12) | Phase 4: H1 | CPD từ MLP, `check_model()` pass |
| 5–6 (8–21/12) | Phase 5: H2, prototype NOTEARS tuần 5, chạy đủ tuần 6 | DAG H2, re-identify |
| 7 (22–28/12) | Phase 6: bootstrap, refuter, Table C, stability | Bảng kết quả cuối |
| 8 (29/12–4/1) | Phase 7: báo cáo, slide, dọn notebook | Bản nộp |

## 10. Việc cần làm trước Phase 3

1. **Môi trường DoWhy.** `.venv` dùng Python 3.14, DoWhy ≥ 0.12 yêu cầu Python < 3.14 nên pip chỉ cài được DoWhy 0.8 (API cũ, không có GCM). Cần một venv Python 3.13 (`brew install python@3.13` rồi tạo `.venv313`) hoặc chạy phần DoWhy trên Colab. pgmpy 1.1.2, torch 2.15, sklearn 1.9.1, pandas 3.1 đã chạy được trong `.venv`.
2. Quyết định cách xử lý 27 ô trống của C_ext trong do-query (giới hạn reference cohort, hoặc chỉ identify với C_core và báo cáo C_ext như độ nhạy).
3. Viết `scripts/learn_dag_coupon.py` theo mẫu `scripts/learn_dag_b0.py`, đọc `data/processed/coupon_discretized.csv` và `artifacts/coupon/splits.json`.

## 11. Trạng thái Phase 1–2 (hoàn thành 2026-11-06)

- Dữ liệu gốc: `data/raw/coupon/in-vehicle-coupon-recommendation.csv`, checksum và provenance trong `data/raw/README.md`.
- Script: `scripts/prep_coupon.py` (import được từ notebook) → `data/processed/coupon_discretized.csv` (12.037 dòng, 12 nút + `education_bin`, `row_id`, `resp_id`).
- Artifact: `artifacts/coupon/{audit.json, variable_dictionary.csv, splits.json, overlap_train_seed{0..4}_{C_core,C_ext}.csv, config_coupon.json}`.
- Notebook: `notebooks/02_coupon_cbn.ipynb` mục 1–3 đã chạy sạch bằng nbconvert (setup, nguồn và bảng giả định A1–A4, audit, resp_id, eligibility flow, từ điển biến, kiểm tra A1, split, overlap, mốc sơ bộ).
- Mốc sơ bộ trên toàn dữ liệu đủ điều kiện: Y = 57,7%, P(T=1) = 45,8%; chênh thô −15,0 điểm %; ATE chuẩn hoá theo C_core −16,4; theo C_ext −17,1 (trên 84% dòng có overlap). CATE thô theo coupon: Restaurant(<20) −24,3, Restaurant(20–50) −22,3, Coffee House −15,2, Carry out −12,0, Bar −3,1.
