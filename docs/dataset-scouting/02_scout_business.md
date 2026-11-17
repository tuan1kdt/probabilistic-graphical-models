# Báo cáo scout: Business / Social Science / Law / Games (shard 43 dataset)

Đã rà toàn bộ 43 dataset trong shard. Đã đọc metadata đầy đủ qua API UCI của 8 dataset (697, 603, 563, 350, 332, 468, 488, 117). Đã tải CSV của 7 dataset (697, 603, 563, 350, 332, 468, 488) vào `scratchpad/dl_business/<id>/` và đếm bằng `python3 -I` (không dùng pandas).
Ký hiệu: **[KC]** = đã kiểm chứng bằng đếm trên dữ liệu thật; **[MD]** = lấy từ metadata hoặc bài báo, chưa kiểm chứng. License của mọi dataset UCI dưới đây là CC BY 4.0 (mặc định của UCI; với 697 thì bài báo gốc ghi rõ). DOI lấy từ API.

---

## 1. ID 603: In-Vehicle Coupon Recommendation (thời hạn coupon 2h so với 1 ngày → chấp nhận coupon)
- URL: https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation. Năm 2017, DOI 10.24432/C5GS4P [MD]. n=12.684 [KC], 25 biến, toàn bộ là categorical.
- Nguồn: khảo sát Amazon Mechanical Turk. Wang, Rudin et al., JMLR 18 (2017).
- **Câu hỏi:** Xét các tình huống lái xe nhận được coupon đề xuất trên xe (population). Nếu doanh nghiệp đặt coupon hết hạn sau **2 giờ (T=1)** thay vì **1 ngày (T=0)**, thì xác suất tài xế chấp nhận coupon (Y=1) thay đổi bao nhiêu? T được quyết định ngay lúc phát coupon.
- **Số đếm [KC]:**
  - T=2h có 5.593 dòng (44%).
  - P(Y=1) là 0,626 khi T=1d và 0,496 khi T=2h, tức chênh lệch thô −13 điểm %.
  - Tỉ lệ Y tổng là 57%.
  - Tỉ lệ 2h dao động theo coupon×destination từ 0,30 đến 0,65. Riêng ô **Bar×Work = 0% (vi phạm positivity)**.
  - Thời tiết Snowy chỉ có 28% là 2h.
  - Biến `toCoupon_GEQ5min` là hằng số. Cột `car` có 99% NaN. Cột Bar có 107 NaN.

| Vai trò | Biến |
|---|---|
| T | expiration (2h=1, 1d=0) |
| Y | Y (accept) |
| Confounder C (biến thiết kế kịch bản, quyết định cách gán T) | coupon (5 loại), destination, time, passenger, weather (temperature trùng lặp với weather) |
| Baseline của người trả lời (tiền xử lý, làm biến điều chỉnh hoặc tăng độ chính xác) | age, gender, income, has_children, maritalStatus, education, CoffeeHouse/Bar/... (tần suất thói quen) |
| Biến khác | toCoupon_GEQ15/25min, direction_same: thuộc tình huống và cùng thời điểm với T. Coi là C hoặc biến song song, không phải mediator. |
| Bỏ | car (99% NaN), toCoupon_GEQ5min (hằng số), direction_opp (=1−direction_same), occupation (25 mức, gộp lại hoặc bỏ) |

- **Rủi ro:**
  - (a) Y là **ý định khai báo trong tình huống giả định**, không phải hành vi thật.
  - (b) Mỗi người trả lời khoảng 20 tình huống nhưng dữ liệu **không có respondent ID**, nên các quan sát không độc lập (cluster). Split ngẫu nhiên có thể làm cùng một người xuất hiện ở cả train và test.
  - (c) Positivity: phải loại Bar×Work hoặc gộp coupon.
  - (d) Thiết kế khảo sát không phải RCT thuần, nên T vẫn bị confound bởi coupon/destination/weather. Đây lại là điểm tốt cho bài tập điều chỉnh confounder, vì cơ chế gán T đo được gần như đầy đủ.
- **Điểm:**
  - Thực tiễn **3**: thời hạn khuyến mãi là quyết định marketing thật (urgency vs. flexibility), nhưng Y là ý định giả định.
  - Can thiệp **5**: T là thuộc tính coupon do doanh nghiệp đặt, do-operator có nghĩa thao tác trực tiếp.
  - Confounding/timing **4**: cơ chế gán T nằm trong các biến thiết kế đã đo; timing rõ (kịch bản → coupon → quyết định). Trừ điểm vì cluster theo người.
  - Kỹ thuật **4**: n=12,7k, toàn categorical (không cần bin), dễ chọn DAG 10–13 nút; có một ô positivity bằng 0 và cần xử lý NaN.
  - Mới **5**: khác hẳn 222/320 (marketing theo ngữ cảnh, T là thiết kế ưu đãi chứ không phải kênh liên lạc).
  - **Tổng 21/25**
- Nguồn: API UCI, CSV đã tải và đếm, bài JMLR 2017 (đọc qua mô tả trên UCI).

## 2. ID 332: Online News Popularity (đăng bài cuối tuần → bài "phổ biến")
- URL: https://archive.ics.uci.edu/dataset/332/online+news+popularity. Năm 2015, DOI 10.24432/C5NS3V. n=39.644 [KC] (metadata ghi 39.797), 58 biến dự báo.
- Bài gốc: Fernandes, Vinagre & Cortez, EPIA 2015.
- **Câu hỏi:** Với các bài Mashable 2013–2014, nếu biên tập viên đăng bài vào **cuối tuần (T=1)** thay vì ngày thường (T=0), thì xác suất bài đạt ≥1.400 lượt share (median, Y=1) thay đổi bao nhiêu? T được quyết định lúc lên lịch xuất bản.
- **Số đếm [KC]:**
  - is_weekend=1 có 5.190 bài (13,1%).
  - P(Y=1) là 0,715 khi đăng cuối tuần và 0,506 khi đăng ngày thường, tức chênh lệch thô +21 điểm %.
  - Y tổng khoảng 53%.
  - Tỉ lệ cuối tuần theo data channel dao động 9–19%, nên overlap tốt ở mọi kênh.
  - T thay thế: num_imgs≥2 (37% treated, P(Y) 0,59 vs 0,50); has_video (37%, 0,55 vs 0,52).

| Vai trò | Biến |
|---|---|
| T | is_weekend (hoặc T thay thế: imgs≥2) |
| Y | shares ≥ 1400 |
| C (đặc tính nội dung có trước khi xuất bản) | data_channel (gộp 6 dummy thành 1 biến 7 mức), n_tokens_content (bin), n_tokens_title, num_imgs, num_videos, num_hrefs, num_keywords, LDA topic chính (argmax LDA_00–04), title_subjectivity/sentiment (bin) |
| Bỏ (đo sau khi đăng hoặc là aggregate của shares, có nguy cơ leakage) | kw_* (share trung bình của keyword, tính trên cả tập), self_reference_*_shares, các weekday_is_* (trùng với T) |
| Biến phơi nhiễm | timedelta (số ngày từ lúc đăng đến lúc thu thập): dùng để điều chỉnh, hoặc lọc timedelta ≥ 21 |

- **Rủi ro:**
  - Nhiễu ẩn mạnh: cuối tuần đăng ít bài hơn, có thể là bài "chọn lọc" hơn. Ngoài ra ít cạnh tranh tin tức, nên "ít cạnh tranh" là cơ chế của T chứ không phải confounder.
  - Hiệu ứng có thể đến từ việc khối lượng bài thấp vào cuối tuần, chứ không phải do ngày đăng tự thân.
  - Hầu hết biến là liên tục, phải bin cẩn thận.
  - Y được lấy tại một thời điểm duy nhất (8/1/2015).
- **Điểm:**
  - Thực tiễn **4**: lên lịch xuất bản là quyết định hằng ngày của tòa soạn và marketer nội dung.
  - Can thiệp **4**: ngày đăng do người quyết định, do(T) có nghĩa (giả định nội dung giữ nguyên).
  - Confounding/timing **3**: nội dung có trước T và đo được; chất lượng biên tập và tin nóng thì không đo được.
  - Kỹ thuật **4**: n lớn, không missing, nhưng phải bin khoảng 10 biến liên tục và loại các biến kw/self_ref.
  - Mới **5**: lĩnh vực truyền thông số, khác hẳn hai dataset mẫu.
  - **Tổng 20/25**
- Nguồn: API UCI (danh sách 61 thuộc tính), CSV đã tải và đếm.

## 3. ID 697: Predict Students' Dropout and Academic Success (học bổng → bỏ học)
- URL: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success. Năm 2021, DOI 10.24432/C5MC89, CC BY 4.0 [MD, bài gốc].
- n=4.424 [KC], 36 biến, không missing. Sinh viên Polytechnic of Portalegre nhập học 2008/09–2018/19, 17 ngành [MD, Realinho et al., *Data* 7(11):146, 2022].
- **Câu hỏi:** Với sinh viên đại học năm nhất, nếu được cấp **học bổng (T=1)** thay vì không (T=0), thì xác suất **bỏ học** trước khi hết thời gian chuẩn của khóa (Y=Dropout) thay đổi bao nhiêu? Học bổng được quyết định vào khoảng thời điểm nhập học.
- **Số đếm [KC]:**
  - Scholarship=1 có 1.099 sinh viên (24,8%).
  - P(Dropout) là 0,122 khi có học bổng và 0,387 khi không.
  - Dropout tổng là 32%.
  - Tỉ lệ có học bổng theo nhóm tuổi: <20 là 34%, 20–24 là 24%, 25+ là 10%. Nữ 30%, nam 15%. Học buổi tối 13%. Có ở mọi bin admission grade (lo/mid/hi), nên overlap đủ.
  - Một số application mode nhỏ không có ai có học bổng (ví dụ mode 7: 1/139), cần gộp lại.
  - T thay thế: học buổi tối (11%, nhưng gần như 0% ở tuổi <20, nên positivity kém).

| Vai trò | Biến |
|---|---|
| T | Scholarship holder |
| Y | Target = Dropout (1) vs. Graduate+Enrolled (0); hoặc bỏ nhóm Enrolled (còn n=3.630) |
| C (lúc nhập học) | Admission grade (bin), Previous qualification (grade), Age at enrollment, Gender, Mother's/Father's qualification (gộp 3–4 mức), Mother's/Father's occupation, Displaced, Application mode (gộp), Course (gộp theo lĩnh vực), Daytime/evening, Marital status, macro (Unemployment/Inflation/GDP theo năm nhập học) |
| Post-treatment/mediator cần loại | 12 biến Curricular units sem 1–2 (kết quả học tập sau T); Debtor và Tuition fees up to date (có thể là hệ quả của việc có hoặc không có học bổng, timing không rõ) |
| Bỏ | Nacionality/International (rất lệch), Educational special needs (hiếm) |

- **Rủi ro:**
  - (a) **Timing không rõ:** bài báo xếp Scholarship holder vào nhóm "socioeconomic data", không ghi rõ đo lúc nhập học. Nếu là snapshot sau đó (ví dụ học bổng bị cắt khi trượt môn) thì có **reverse causation**. Đây là rủi ro lớn nhất.
  - (b) Học bổng DGES cấp theo thu nhập hộ gia đình, mà thu nhập không đo được, chỉ có proxy là trình độ và nghề nghiệp của cha mẹ.
  - (c) n=4,4k hơi nhỏ cho CPD có nhiều cha, nên phải gộp mức mạnh tay.
  - (d) Cùng lĩnh vực giáo dục với ví dụ mẫu.
- **Điểm:**
  - Thực tiễn **5**: phân bổ học bổng và hỗ trợ tài chính để giảm bỏ học là chính sách được tranh luận thật (cả EU và Việt Nam).
  - Can thiệp **4**: tổ chức hoặc nhà nước quyết định; do(grant) rõ nghĩa.
  - Confounding/timing **3**: baseline phong phú hơn ID 320, nhưng timing của T và thu nhập ẩn là vấn đề.
  - Kỹ thuật **4**: hầu hết là discrete, không missing, overlap đủ; n vừa phải.
  - Mới **3**: tốt hơn rõ rệt so với 320 (n gấp 7 lần, outcome thật là bỏ học), nhưng vẫn là giáo dục và trợ cấp.
  - **Tổng 19/25**
- Nguồn: API UCI, CSV đã tải và đếm, PDF bài gốc (đã đọc Table 1 và mục Data Description), DOAJ/Zenodo.

## 4. ID 350: Default of Credit Card Clients (hạn mức tín dụng cao → vỡ nợ)
- URL: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients. Năm 2009, DOI 10.24432/C55S3H. n=30.000 [KC], 23 biến, không missing. Taiwan, tháng 4–9/2005.
- **Câu hỏi:** Với chủ thẻ tín dụng của một ngân hàng Đài Loan, nếu ngân hàng cấp **hạn mức ≥ 140.000 NT$ (T=1, median)** thay vì thấp hơn, thì xác suất vỡ nợ tháng 10/2005 thay đổi bao nhiêu?
- **Số đếm [KC]:**
  - T=1 có 15.459 người (51,5%).
  - P(default) là 0,161 khi T=1 và 0,286 khi T=0.
  - Default tổng là 22,1%.
  - Overlap theo education (35–68%), tuổi (36–64%) và PAY_0 (28–84%) đều tốt.

| Vai trò | Biến |
|---|---|
| T | LIMIT_BAL ≥ 140k |
| Y | default payment next month |
| C | SEX, EDUCATION (gộp 4,5,6,0), MARRIAGE, AGE (bin); có thể thêm PAY_6/BILL_AMT6 (tháng 4) làm proxy lịch sử *nếu giả định hạn mức được cấp trước tháng 4/2005* |
| Mediator/không rõ timing | PAY_0..PAY_5, BILL_AMT1–5, PAY_AMT1–6 (cùng thời kỳ với hạn mức đang có hiệu lực: chi tiêu và trả nợ là hệ quả của hạn mức) |

- **Rủi ro:**
  - Hạn mức được ngân hàng đặt dựa trên thu nhập và điểm tín dụng, mà **cả hai đều không có trong dữ liệu**, nên nhiễu ẩn rất mạnh. Dấu của hiệu ứng thô gần như chắc chắn bị confounding chi phối.
  - Không biết thời điểm cấp hoặc điều chỉnh hạn mức.
  - T là liều liên tục bị nhị phân hoá.
- **Điểm:**
  - Thực tiễn **4**: chính sách hạn mức là quyết định rủi ro tín dụng thật.
  - Can thiệp **3**: ngân hàng quyết định, nhưng ngưỡng nhị phân là nhân tạo.
  - Confounding/timing **2**: confounder chính (thu nhập, credit score) không đo được, timing mơ hồ.
  - Kỹ thuật **5**: n 30k, sạch, cân bằng.
  - Mới **4**
  - **Tổng 18/25**
- Nguồn: API UCI (mô tả X1–X23), CSV đã tải và đếm.

---

## Xếp hạng
1. **603 In-Vehicle Coupon, expiration 2h → accept (21)**: can thiệp rõ ràng nhất, cơ chế gán T đo được, toàn categorical. Điểm yếu là outcome giả định và cluster theo người trả lời.
2. **332 Online News, đăng cuối tuần → popular (20)**: n lớn, quyết định biên tập thật, lĩnh vực mới. Cần loại các biến kw/self_ref vì leakage.
3. **697 Student Dropout, học bổng → bỏ học (19)**: câu hỏi chính sách hấp dẫn nhất, nhưng phải chấp nhận và thảo luận rủi ro timing/reverse causation của biến học bổng.
4. 350 Credit Default, hạn mức → vỡ nợ (18): chỉ nên dùng làm phương án dự phòng vì confounding không đo được quá mạnh.

## Đáng chú ý nhưng đã loại
- **563 Iranian Churn:** Tariff Plan contractual chỉ chiếm 7,8% (245 khách, chỉ khoảng 6 người churn) [KC]. Complains thì không phải can thiệp, và mọi biến đều là aggregate 9 tháng nên timing lẫn lộn. Ngoài ra n=3.150.
- **468 Online Shoppers / 488 FB Live Sellers / 2, 117 Adult & Census-KDD:**
  - 468: Weekend và SpecialDay không phải quyết định của ai; PageValues gây leakage.
  - 488: video vs. photo là quyết định có thật (P(reactions≥100) 0,63 vs 0,29 [KC]), nhưng không có seller ID trong status_id và chỉ có giờ đăng làm confounder, nên quá ít biến.
  - Adult/Census: T khả dĩ (học vấn, giờ làm, công đoàn) gần với thuộc tính nhân khẩu học; dataset quá quen thuộc.
- **522/573/144 German Credit, 27/143 Credit Approval, 373 Drug Consumption, 183/211 Communities & Crime:**
  - Credit datasets: n ≤ 1.000.
  - 373: T (ví dụ dùng cannabis) là hành vi tự chọn, và chỉ có tính cách và nhân khẩu học làm C với n=1.885; chấp nhận được nhưng yếu.
  - 183/211: đơn vị là cộng đồng, aggregate, nhiều missing.
  - Shard còn có Games (Poker, Chess, Connect-4, Dota2, Tic-Tac-Toe): là trạng thái hoặc luật chơi deterministic, không có T.
  - Graph/network (GitHub MUSAE, Facebook, LastFM), time series (Bike Sharing, Seoul Bike) và tài chính doanh nghiệp (Polish/Taiwanese Bankruptcy, toàn tỉ số tài chính, không có quyết định nhị phân) cũng bị loại.

Nguồn web: [Realinho et al. 2022 PDF](https://dorsu.edu.ph/faculty/jscabrera/wp-content/uploads/2022/12/data-07-00146.pdf), [DOAJ](https://doaj.org/article/01a8ef129677407983e785650966e062), [Zenodo](https://zenodo.org/records/5777340), các trang UCI và API `https://archive.ics.uci.edu/api/dataset?id=<id>`.
