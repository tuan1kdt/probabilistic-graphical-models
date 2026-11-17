# [LƯU TRỮ — thay bằng research-plan-coupon.md từ 2026-11-06] Kế hoạch nghiên cứu — Causal Bayesian Networks trên UCI Bank Marketing

Môn: IT701100 Mô hình đồ thị có xác suất, HK1 2026–2027 (GV:TS. Nguyen Chi Thien).

## Dataset đã chọn
- UCI ID 222 Bank Marketing, `bank-additional-full.csv` (~41.188 dòng, 20 input). 
- T = `contact` (telephone=0, cellular=1; Y = `y` (đăng ký tiền gửi có kỳ hạn).
- Estimand: ATE = P(Y=1|do(T=1)) − P(Y=1|do(T=0)) trên TEST reference cohort.
- Loại `duration` (post-treatment). `campaign`: mediator hoặc loại — **không** đưa vào adjustment set. `month`/`day_of_week`/chỉ số vĩ mô (`emp.var.rate`, `euribor3m`, `cons.conf.idx`…) = confounder "giai đoạn chiến dịch". `pdays`/`previous`/`poutcome` = lịch sử trước T.

## Phases
1. Đặc tả câu hỏi + data audit: câu hỏi một câu; variable dictionary (meaning, coding, timing, vai trò); overlap contact × month/poutcome/age; nguồn + checksum; eligibility flow; bảng giả định.
2. Tiền xử lý + split (0.5 tuần, slide 56): DAG 8–12 nút (gộp job, education, age bin, gom macro thành `period`); fit bin trên TRAIN; 5 split 70/15/15 stratified seed 0–4; khoá tập C baseline.
3. B0 (1.5 tuần, slide 57–59): forbidden edges (không cạnh ra từ y; contact không trỏ về pre-treatment); HillClimbSearch + `ExpertKnowledge(forbidden_edges, required_edges, temporal_order)`; DoWhy identify; CPD BDeu ESS=5; dự đoán + chuẩn hoá do-query trên TEST.
4. H1 (1 tuần, slide 60–61): MLP thay CPD của y (input = parents của y, dropout 0.2, BCE, early stopping); **liệt kê mọi tổ hợp parent → bảng CPD chuẩn hoá → chèn vào bản sao B0 → `check_model()`** (bắt buộc theo slide 60; mọi nút đều rời rạc nên bảng là chính xác, không mất thông tin).
5. H2 (1.5 tuần, slide 62–63): NOTEARS nonlinear với mask M (W_eff = M⊙W), ngưỡng/λ trên VALIDATION, quy tắc sửa chu trình; re-identify; CPD cùng estimator B0. Chỉ nhắc DAGMA/GOLEM/DCDI ở related work; không đổi phương pháp trừ khi NOTEARS thất bại với biến categorical.
6. Đánh giá (1 tuần, slide 68–71, 74–75): Table A/B/C; bootstrap CI (resample TRAIN, TEST cohort cố định); 2 refuter DoWhy (placebo permute, subset 0.8, 100 sim); kiểm tra retrain riêng cho H1; runtime + hardware.
7. Viết (1 tuần, slide 44–50, 76–78): báo cáo 2.1–2.9; slide theo 6 bước demo; notebook chạy sạch; nhất quán; AI disclosure; phụ lục A3 nếu còn thời gian.

## Trạng thái Phase 0 (2026-10-30)
- Cấu trúc thư mục: `requirements.txt`, `notebooks/00_phase0_setup_tutoring.ipynb` (đã chạy, output lưu), `notebooks/01_bank_marketing_cbn.ipynb` (khung 11 mục theo slide 76), `artifacts/config_phase0.json` (CONFIG + ENV), `data/raw/README.md`.
- Môi trường kiểm chứng: Python 3.13, pgmpy 1.1.2, DoWhy 0.14, torch 2.15 (CPU), sklearn 1.9.1.
- Kết quả tutoring (n=20k, 5 seed): BN ATE 0.295 ± 0.005, DoWhy ATE 0.295 ± 0.005, GCM ATE 0.297; placebo 0.0003 (p=0.98), subset 0.290 (p=0.94) — khớp tham chiếu 0.30/0.60/0.30.
- Lưu ý API: pgmpy ≥ 1.1 bỏ `prior_type` khỏi `fit()`; dùng `BayesianEstimator(bn, df, state_names).get_parameters(prior_type="BDeu", equivalent_sample_size=5)` rồi `add_cpds` — đã gói trong `fit_discrete_bn()`. Structure learning: `HillClimbSearch(df).estimate(scoring_method="bic-d", expert_knowledge=ExpertKnowledge(...))`; GES/PC/MmhcEstimator nằm trong `pgmpy.estimators`.
- Hàm dùng chung đã có trong notebook 00: `set_seed`, `fit_discrete_bn` (BDeu / ML cho A3), mẫu chuẩn hoá do-query theo cohort TEST, mẫu Table A.
- Môi trường Cowork cục bộ không có mạng; sandbox cloud bị chặn archive.ics.uci.edu → tải thủ công `bank-additional-full.csv` vào `data/raw/` (hoặc chạy notebook 01 trên Mac/Colab).

## Ghi chú related work (2026-11-02, từ bản tổng hợp Perplexity đã rà soát)
- Dùng được: `duration` leakage; 5 biến vĩ mô là proxy thời điểm chiến dịch; kênh liên hệ do ngân hàng quyết định theo thời kỳ nhưng không randomized → chỉ viết "plausibly bank-driven, non-random", kết quả có điều kiện trên giả định; NOTEARS/DAG-GNN/Reisach 2021 (varsortability); DoWhy refuter không chứng minh DAG đúng.
- Đã loại bỏ: gợi ý không lập bảng CPD cho H1 (trái slide 60); đưa `campaign` vào backdoor set; `contact` có `unknown`.
- Chưa có nguồn: "neural CPD trong BN" → dẫn Koller & Friedman ch.5 (structured CPDs) + DoWhy GCM `ClassifierFCM` làm tiền lệ. Survey deep causal discovery cần kiểm tra: Vowels, Camgoz & Bowden (2022) *ACM Computing Surveys* (arXiv:2103.02582); Zanga, Ozkirimli & Stella (2022) *Int. J. Approx. Reasoning*.
- Trích dẫn cần thay: link GitHub cá nhân → trang UCI + DOI Moro; DoWhy docs → v0.14 (slide 52); Pearl *Causality* → DOI 10.1017/CBO9780511803161.

## Rủi ro cần kiểm tra sớm
- Positivity: `period` có thể quyết định gần hoàn toàn `contact` ở một số tháng → thu hẹp dân số mục tiêu nếu cần (Phase 1).
- NOTEARS với biến categorical: thử prototype nhỏ cuối Phase 3.
