# Đề xuất 5 dataset UCI thay thế cho đồ án Causal BN (IT701100)

Ngày rà soát: 2026-11-02. Quy trình: cào metadata 689 dataset trên UCI qua API chính thức → lọc còn 169 dataset bảng (n ≥ 500, p ≥ 5, không phải ảnh/văn bản/chuỗi thời gian) → 3 scout Opus rà theo lĩnh vực (y tế–sinh học 47, kinh doanh–xã hội 43, kỹ thuật–CS 79, kèm 99 dataset bị loại để vớt lại) → 12 ứng viên → 1 giám khảo Opus đếm lại trên dữ liệu thật và phản biện 7 ứng viên dẫn đầu → tổng hợp. Báo cáo gốc của từng agent nằm trong `docs/dataset-scouting/`; danh mục UCI đã cào ở `docs/dataset-scouting/uci_catalog_2026-11-02.tsv`.

Tiêu chí (1–5 mỗi mục): Thực tiễn (TT), Can thiệp rõ (CT), Confounding/timing (CF), Kỹ thuật (KT), Mới so với ví dụ môn học (M). Mọi tỉ lệ dưới đây đã được đếm trực tiếp trên file tải về, không lấy từ metadata.

## Bảng tổng hợp

| # | UCI | Dataset | T (control → treatment) | Y | n dùng được | T% | Y% | Chênh thô | TT | CT | CF | KT | M | Tổng |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 296 | Diabetes 130-US Hospitals | không → có chỉ định xét nghiệm HbA1c trong đợt nằm viện | tái nhập viện ≤30 ngày | 69.973 | 18,4 | 9,0 | −0,7 đ% | 5 | 4 | 3 | 5 | 5 | 22 |
| 2 | 603 | In-Vehicle Coupon | hạn coupon 1 ngày → 2 giờ | chấp nhận coupon | 12.684 (~587 người) | 44 | 57 | −13 đ% | 3 | 5 | 4 | 4 | 4 | 20 |
| 3 | 1025 | Turkish Crowdfunding | không → có video giới thiệu | gọi vốn thành công | 1.618 | 58 | 23 | +31 đ% | 4 | 5 | 3 | 2 | 4 | 18 |
| 4 | 697 | Students' Dropout & Success | không → có học bổng | bỏ học | 3.630 (bỏ "Enrolled") | 24,8 | 32 | −26 đ% | 5 | 3 | 2 | 4 | 2 | 16 |
| 5 | 332 | Online News Popularity | đăng ngày thường → cuối tuần | share ≥ 1.400 | ~38.000 | 13,1 | 53 | +21 đ% | 3 | 4 | 3 | 4 | 4 | 18 |
| — | 498 | ServiceNow Incident log | không → có dùng knowledge base | đạt SLA | 24.918 ticket | 14,3 | 63 | −45 đ% | 4 | 2 | 2 | 4 | 5 | 17 |
| — | 891 | CDC BRFSS Diabetes Indicators | chưa từng → từng hút thuốc | bệnh mạch vành/NMCT | 229.474 (bỏ trùng) | 44 | 9,4 | +6,7 đ% | 3 | 2 | 3 | 5 | 3 | 16 |
| mốc | 222 | Bank Marketing (hiện tại) | telephone → cellular | đăng ký tiền gửi | 41.188 | 63 | 11,3 | +9,5 đ% | 3 | 4 | 3 | 5 | — | — |
| mốc | 320 | Student Performance (ví dụ môn) | schoolsup no → yes | G3 ≥ 10 | 649 / 395 | ~12 | ~85 | — | 3 | 4 | 3 | 2 | — | — |

Thứ tự 1–5 là xếp hạng cuối của tôi sau khi cân giữa scout và giám khảo. Điểm là điểm của giám khảo, trừ 296 và 697 tôi điều chỉnh (giải thích trong từng mục). Hai dòng "—" là phương án dự phòng, không nằm trong top 5 vì có lỗi định nghĩa treatment (498) hoặc can thiệp không thao tác được (891).

## 1. UCI 296 — Diabetes 130-US Hospitals (1999–2008)

**Đề tài.** Với lượt nhập viện đầu tiên của mỗi bệnh nhân đái tháo đường tại 130 bệnh viện Mỹ, xuất viện còn sống (n = 69.973), nếu bác sĩ chỉ định xét nghiệm HbA1c trong đợt nằm viện (T = 1, A1Cresult ≠ None) thay vì không (T = 0), xác suất tái nhập viện trong 30 ngày (Y = readmitted "<30") thay đổi bao nhiêu? ATE trên TEST cohort.

- **Vì sao thực tế.** Tái nhập viện 30 ngày là chỉ số bị phạt tài chính ở Mỹ; chỉ định HbA1c là quyết định hằng ngày của bác sĩ, chi phí thấp, dễ đưa vào quy trình. Bài gốc của dataset (Strack et al. 2014) đặt đúng câu hỏi này nên có mốc đối chiếu.
- **Vai trò biến.** C = age (3–4 bin), gender, race, admission_type, admission_source, number_inpatient và number_emergency năm trước (0/1/2+), nhóm ICD của diag_1, payer_code. Loại vì post-treatment: discharge_disposition, time_in_hospital, num_lab_procedures, num_procedures, num_medications, change, diabetesMed, các cột thuốc. Bỏ: weight, medical_specialty (missing cao), diag_2/3.
- **Kỹ thuật.** n lớn nhất trong nhóm có can thiệp hợp lệ; mọi biến đã phân loại; Y có khoảng 216 tổ hợp parent với ~170 dòng TRAIN mỗi ô nên CPD không rỗng mà vẫn đủ thưa để MLP-CPD (H1) có việc.
- **Rủi ro.** Thói quen chỉ định HbA1c phụ thuộc bệnh viện nhưng không có ID bệnh viện (nhiễu ẩn chính). Một ô admission_source × admission_type (mã NULL, n = 1.779) có 0% chỉ định, cần loại. Hiệu ứng kỳ vọng nhỏ (chênh thô −0,7 điểm %), nên phải báo cáo CI bootstrap và chấp nhận kết luận "không khác 0" nếu xảy ra.
- **Lưu ý quan trọng.** Scout ban đầu đề xuất T = `change` (đổi thuốc). Giám khảo chỉ ra 24% bệnh nhân không dùng thuốc tiểu đường nên T = 0 về cấu trúc, và trong nhóm có dùng thuốc chênh lệch chỉ +0,05 điểm %; `change` gộp cả tăng, giảm, thêm, ngừng thuốc nên do() vô nghĩa. Vì vậy chỉ dùng biến thể HbA1c. Tôi giữ KT = 5 và TT = 5 nên tổng 22 thay vì 19 của giám khảo (giám khảo trừ vì ATE có thể ≈ 0, tôi xem đó là kết quả hợp lệ chứ không phải lỗi dataset).

## 2. UCI 603 — In-Vehicle Coupon Recommendation

**Đề tài.** Với các tình huống lái xe trong khảo sát MTurk 2017 (n = 12.684 phản hồi từ ~587 người), nếu coupon hết hạn sau 2 giờ (T = 1) thay vì 1 ngày (T = 0), xác suất người lái chấp nhận coupon (Y) thay đổi bao nhiêu? ATE trên TEST, split theo nhóm người trả lời.

- **Vì sao thực tế.** Thời hạn coupon là đòn bẩy marketing có thật và rẻ; câu hỏi "khan hiếm thời gian có kích thích hành vi không" vẫn được dùng cho push notification và ứng dụng gọi xe.
- **Vai trò biến.** C = coupon type, destination hoặc time (gộp), passenger, weather, age, income, tần suất đi Bar/CoffeeHouse/Restaurant tương ứng. Loại: car (99% NaN), toCoupon_GEQ5min (hằng), direction_opp (trùng direction_same), temperature (trùng weather). Phải loại ô Bar × Work vì 0% có T = 1.
- **Kỹ thuật.** Toàn biến phân loại, không cần bin. Không có respondent ID nhưng giám khảo tái tạo được 587 block dòng liên tiếp (517 block đúng 22 dòng), đủ để group split. T do 19 phiên bản bảng hỏi quyết định nên gần như quasi-randomized theo kịch bản: DAG có các cạnh kịch bản → T rõ ràng để H2 (NOTEARS) phải tìm ra.
- **Rủi ro.** Y là ý định khai báo trong tình huống giả định, không phải hành vi thật (TT = 3). Ở mức kịch bản đầy đủ (6 biến thiết kế) chỉ 52% dòng nằm trong ô có cả hai giá trị T, nên tập C phải thô (ở mức coupon × time thì 95% dòng có overlap).

## 3. UCI 1025 — Turkish Crowdfunding Startups

**Đề tài.** Với dự án gọi vốn cộng đồng dạng phần thưởng trên các nền tảng Thổ Nhĩ Kỳ 2013–2020 (n = 1.618), nếu chủ dự án đăng video giới thiệu (T = 1) thay vì không (T = 0), xác suất đạt mục tiêu gọi vốn (Y) thay đổi bao nhiêu?

- **Vì sao thực tế.** Làm video tốn tiền và thời gian; đây là lời khuyên phổ biến nhất trên mọi nền tảng crowdfunding nhưng hiếm khi được kiểm chứng có điều chỉnh. Người ra quyết định là founder hoặc chính nền tảng (bắt buộc video hay không).
- **Vai trò biến.** C = platform (3 + other), năm (3 mức), kategori (gộp ~6), hedef_miktari (tertile), fon_sekli, kinh nghiệm chủ dự án (số dự án đã làm/đã ủng hộ), ekip_kisi_sayisi, gun_sayisi. Loại vì leakage: destekci_sayisi, toplanan_tutar, destek_orani, yorumlar, guncellemeler, sm_takipci, video_uzunlugu, các trường text.
- **Rủi ro.** n nhỏ: chỉ 34 dự án thành công không có video trên toàn bộ dữ liệu, nên nhiều ô T = 0 trong TRAIN có 0–1 ca dương; KT = 2. Nhiễu ẩn "chất lượng dự án" tác động cả T lẫn Y, cần ghi rõ. Nền tảng arıkovanı có 96,6% dự án có video, cân nhắc loại. Bắt buộc báo cáo phương sai ATE giữa 5 split. Tuy nhỏ, dataset này vẫn lớn gấp 2,5 lần ví dụ mẫu của môn (Student Performance, 649 dòng).

## 4. UCI 697 — Predict Students' Dropout and Academic Success

**Đề tài.** Với sinh viên nhập học Polytechnic of Portalegre 2008–2019, đã có kết quả cuối (n = 3.630 sau khi bỏ nhóm "Enrolled"), nếu được cấp học bổng (T = 1) thay vì không (T = 0), xác suất bỏ học (Y) thay đổi bao nhiêu?

- **Vì sao thực tế.** Học bổng là chính sách tốn ngân sách và câu hỏi "học bổng có giữ chân sinh viên không" là một trong những bài toán được nghiên cứu nhiều nhất trong kinh tế giáo dục. Người ra quyết định là nhà trường và bộ.
- **Vai trò biến.** C = điểm đầu vào, điểm bằng trước, tuổi lúc nhập học, giới, trình độ/nghề của cha mẹ, displaced, application mode (gộp), ngành (gộp), học ngày/tối, bộ ba biến vĩ mô (proxy khoá nhập học). Loại vì post-treatment: 12 biến kết quả học kỳ 1–2, Debtor, Tuition fees up to date.
- **Rủi ro.** Học bổng ở Bồ Đào Nha được xét theo thu nhập hộ gia đình (không có trong dữ liệu) và phải gia hạn hằng năm theo số tín chỉ đỗ. Giám khảo đếm được tỉ lệ học bổng 10,4% ở nhóm đỗ 0–1 tín chỉ kỳ 1 so với 35,5% ở nhóm đỗ ≥ 6, nên biến "Scholarship holder" có thể là trạng thái chụp về sau (nhân quả ngược). Hướng thiên lệch do thu nhập ẩn là làm hiệu ứng bảo vệ bị đánh giá thấp, có thể lập luận là ước lượng bảo thủ. Cùng lĩnh vực với ví dụ mẫu (M = 2). Tôi xếp trên 332 vì câu hỏi có giá trị thực tiễn rõ hơn và DAG không tầm thường.

## 5. UCI 332 — Online News Popularity (Mashable 2013–14)

**Đề tài.** Với bài viết có ít nhất 21 ngày thu thập (n ≈ 38.000), nếu đăng vào cuối tuần (T = 1) thay vì ngày thường (T = 0), xác suất bài đạt ≥ 1.400 lượt share (median) thay đổi bao nhiêu?

- **Vì sao thực tế.** Lịch đăng là quyết định của biên tập viên và vẫn là câu hỏi thường trực của mọi đội nội dung; chi phí thao tác gần bằng 0.
- **Vai trò biến.** C = data_channel, chủ đề LDA chính, n_tokens_content (bin), num_imgs, num_hrefs, timedelta (bin). Loại vì leakage: toàn bộ kw_* và self_reference_* (tính trên cả tập), weekday_is_* (trùng T), url.
- **Rủi ro.** Giám khảo chỉ ra C gần như không giải thích T (tỉ lệ cuối tuần 11–15% ở mọi chủ đề), nên ATE điều chỉnh ≈ chênh thô (+21,1 so với +20,9 điểm %) và B0 ≈ H1 ≈ H2 quanh T. Dự án sẽ an toàn về kỹ thuật nhưng phần so sánh ba mô hình kém hấp dẫn. Dữ liệu 2013 đã cũ. Phải bin ~10 biến liên tục. Chỉ chọn nếu ưu tiên độ an toàn.

## Hai phương án dự phòng và vì sao không vào top 5

- **UCI 498 ServiceNow Incident log** (knowledge base → đạt SLA). Câu hỏi rất thời sự cho ITSM năm 2026 và n = 24.918 ticket đẹp, nhưng giám khảo phát hiện log là "enriched": các trường trạng thái cuối (closed_at, resolved_by) được điền ngược vào mọi event, và cờ knowledge ổn định 99,2% theo cùng kiểu đó, nên không chứng minh được T đặt lúc mở ticket. Thời gian resolve trung vị 94 giờ khi T = 1 so với 10,6 giờ khi T = 0 gợi ý ticket khó mới tra KB (nhân quả ngược). Chỉ dùng nếu xác nhận được ngữ nghĩa cờ từ tài liệu ServiceNow và chấp nhận phân tích độ nhạy bắt buộc.
- **UCI 891 CDC BRFSS** (từng hút thuốc → bệnh tim). Kỹ thuật lý tưởng (229 nghìn dòng sau khi bỏ 24.206 dòng trùng, toàn biến nhị phân) nhưng cross-sectional, phơi nhiễm trọn đời không có thời điểm can thiệp, survivor bias, và với ~1.000 dòng mỗi ô CPD thì H1/H2 không có gì để cải thiện so với B0.

## So với Bank Marketing hiện tại

Bank Marketing (contact → y) có n tốt và can thiệp hợp lệ, nhưng như đã bàn: câu hỏi cellular vs telephone đã lỗi thời, và `contact` vừa là quyết định của ngân hàng vừa là proxy cho việc khách có số di động. Hai ứng viên đầu khắc phục đúng hai điểm này: 296 có câu hỏi lâm sàng còn nguyên giá trị với n lớn hơn; 603 có T gần như được gán theo thiết kế bảng hỏi nên lập luận nhân quả sạch nhất. Đổi dataset tốn khoảng 1 tuần cho Phase 1 (audit, variable dictionary, eligibility flow) vì các hàm dùng chung trong notebook 00 vẫn tái sử dụng được.

**Khuyến nghị.** Chọn 296 với T = chỉ định HbA1c nếu ưu tiên tính thực tiễn và quy mô; chọn 603 nếu ưu tiên độ sạch của lập luận nhân quả và muốn H2 có cấu trúc rõ để khôi phục. Nếu giữ Bank Marketing, dùng bảng trên làm phần "lựa chọn dataset" trong báo cáo 2.1 để biện minh.

## Những dataset đáng chú ý đã loại (tóm tắt)

- Y tế: 880 SUPPORT2 (DNR ra lệnh ngay trước khi chết), 579 MI complications (immortal-time bias, n = 1.700), 890 ACTG 175 (RCT, không có confounding để điều chỉnh, n = 2.139; hợp làm dataset kiểm chứng pipeline), 544 Obesity (77% dòng synthetic).
- Kinh doanh: 350 Credit Card Default (hạn mức tín dụng là T nhân tạo, thiếu thu nhập), 563 Iranian Churn (treated 7,8%), 468 Online Shoppers (Weekend không phải quyết định), Adult/Census (T là nhân khẩu học).
- Kỹ thuật: 601 AI4I (synthetic, failure 3,4%), 165 Concrete (427 cấp phối, Water là mediator), 597 Garment (panel, positivity vỡ), các dataset cảm biến không có biến hành động.
