# Báo cáo giám khảo phản biện: 7 ứng viên dataset cho đồ án Causal BN

Mọi số có ký hiệu **[TĐ]** là tôi tự đếm lại trên file scout đã tải (`python3 -I`, chỉ dùng module `csv`). Script nằm ở `/private/tmp/claude-501/-Users-tuanla-workspace-master-probabilistic-graphical-models/b68b6685-0d78-4bbd-bd39-e6e6064e182c/scratchpad/scripts_judge/` (gồm j296.py, j296b.py, j498.py, j498b.py, j498c.py, j603.py, j603b.py, j332.py, j697.py, j891.py, j1025.py). Số không có ký hiệu là suy luận hoặc ước lượng.

## 1. Kiểm chứng số then chốt

| ID | Số của scout | Số tôi đếm [TĐ] | Khớp không? |
|---|---|---|---|
| 296 | n=69.973; T(change)=45%; Y=8,97%; Y 9,43% so với 8,59% | 69.973; 45,0%; 8,97%; 9,43% so với 8,59% | Khớp. **Nhưng** scout nói "overlap tốt ở mọi tầng" là sai: tầng `diabetesMed=No` (16.680 dòng, 24%) có **T=0% về cấu trúc**. Khi chỉ xét `diabetesMed=Yes` (n=53.293, T=59,1%), chênh lệch thô là **9,43% so với 9,39%, tức +0,05 điểm %**. Toàn bộ chênh lệch thô 0,8 điểm % đến từ nhóm không dùng thuốc (Y=7,5%). |
| 296 (biến thể HbA1c) | T=18,4%; Y 8,39% so với 9,10%; tầng thấp nhất 5,8% | 18,36%; 8,39% so với 9,10%; adm_source=2 là 5,8% | Khớp. Thêm phát hiện: ô admission_source 17 × admission_type 5 (đều là mã "NULL/Not available", n=1.779) có **0% chỉ định HbA1c**. Cách mã hoá và thói quen chỉ định phụ thuộc bệnh viện, mà dataset không có ID bệnh viện. |
| 498 | 24.918 incident; T=14,3%; Y=63,4%; 0,698 so với 0,252 | 24.918; 14,27%; 63,42%; 0,698 so với 0,252 | Khớp về số. **Sai về lập luận timing** (xem mục 2). Tầng xấu nhất: Category 28 có T=1,0% (n=522), Category 45 có **T=89,3%** (n=625); opened_by 17 có 3,2% (n=7.214). Scout báo "3–16% theo category" là thiếu hai đầu này. |
| 603 | n=12.684; 2h=44%; Y 0,626 so với 0,496; Bar×Work=0% | 12.684; 44,1%; 0,626 so với 0,496; Bar×Work 0/487 | Khớp. |
| 332 | 5.190 bài cuối tuần (13,1%); Y 0,715 so với 0,506; tỉ lệ cuối tuần theo kênh 9–19% | 5.190 (13,09%); 0,715 so với 0,506; 9,4–18,7% | Khớp. Chênh lệch sau khi chuẩn hoá theo channel là **+21,1 điểm %**, gần như bằng chênh lệch thô +20,9. |
| 697 | T=24,8%; Dropout 0,122 so với 0,387; Y=32% | 24,84%; 0,122 so với 0,387; 32,1% | Khớp. Tầng xấu nhất: Application mode 7 có 1/139 (0,7%), Previous qualification 3 có 2/126 (1,6%). |
| 891 | T=44,3%; Y=9,4%; 13,17% so với 6,44% | 44,32%; 9,42%; 13,17% so với 6,44% | Khớp. Thêm: **24.206 dòng trùng lặp hoàn toàn** (9,5%, sau khi bỏ ID). Chênh lệch chuẩn hoá theo Age còn 5,3 điểm %. Ô Age×Sex×Edu×Income xấu nhất là 3,8% / 89,7% (n khoảng 50). |
| 1025 | n=1.628; T=58%; Y=23%; 0,361 so với 0,050; 34 ca thành công không có video | 1.628; 58,2%; 23,1%; 0,361 so với 0,050; 34 | Khớp. Positivity: arıkovanı có 96,6% video (n=58), crowdfon×2015 chỉ 1,7% (n=59); crowdfon 18% so với fongogo 69%. |

## 2. Tấn công điểm yếu quyết định

### 296 Diabetes 130-US: T bị trộn, hiệu ứng gần như bằng 0
- **(a) Ngữ nghĩa của T.** Bảng `change` × các cột thuốc [TĐ]:
  - `Ch` có ít nhất một thuốc Up/Down: 17.248.
  - `Ch` không có Up/Down nào trong 23 cột thuốc: **14.243 (45% số Ch)**, tức là thêm hoặc ngừng thuốc. Chéo với insulin, `Ch` gồm 6.777 ca Up, 7.321 Down, 10.694 Steady và 6.699 No.
  - Vậy T trộn "tăng liều, giảm liều, thêm thuốc, ngừng thuốc". Hiệu ứng trung bình của các hành động ngược chiều nhau dễ triệt tiêu, và đó chính là con số +0,05 điểm %.
  - `change=No` và `diabetesMed=No` trùng nhau 100% ở nhóm không dùng thuốc, nên T không được định nghĩa ở nhóm này.
  - `Ch` có thời gian nằm viện trung bình 4,64 ngày so với 3,98 ở nhóm còn lại [TĐ]. Như vậy T đồng thời với LOS, và LOS phải bị loại khỏi tập điều chỉnh.
- **(b) Positivity.** Phải giới hạn population ở `diabetesMed=Yes`. Nhưng `diabetesMed` cũng được ghi trong đợt nằm viện, nên lập luận timing cần nói rõ: "đang dùng thuốc tiểu đường" được coi là điều kiện của population.
  - Trong population đó, overlap tốt (T từ 35% đến 65% theo age, adm, A1C).
- **(c) Kỹ thuật.**
  - DAG khoảng 11–12 nút.
  - Y có parent là T, number_inpatient(3), age(3), diag1(4), adm_src(3), tức 216 cấu hình. TRAIN khoảng 37k dòng, nên trung bình khoảng 170 dòng mỗi ô. CPD không rỗng.
  - H1 ít việc để làm vì n lớn. H2 có việc vì có nhiều biến dịch vụ y tế tương quan nhau.
  - Rủi ro thật là **ATE khoảng 0**: B0, H1, H2 đều cho kết quả xấp xỉ 0 trong khi nhiễu ước lượng khoảng ±0,5 điểm % trên TEST 8k dòng. So sánh phương pháp sẽ không có tín hiệu.
- **(d) Thực tế.** Người ra quyết định là bác sĩ nội trú và chương trình HRRP. Dữ liệu 1999–2008 không có GLP-1 hay SGLT2, nên kết quả khó áp dụng vào điều trị năm 2026.
- **Khuyến nghị.** Đổi sang **T = chỉ định HbA1c** (biến thể B, có bài gốc Strack 2014 làm mốc đối chiếu). Hoặc định nghĩa lại T = `insulin ∈ {Up}` so với `{Steady}` trong nhóm đang dùng insulin để T là một hành động thuần.

### 498 Incident management: T gần như chắc chắn là trạng thái ghi lúc kết thúc
- **(a) Nhân quả ngược.** Scout lập luận "knowledge=true ngay ở event đầu, nên T là quyết định lúc tiếp nhận". Lập luận này **không đứng vững**.
  - Ở event đầu (sys_mod_count=0), các trường `closed_at`, `resolved_by`, `closed_code` đã có giá trị ở **100%** ticket và bằng giá trị ở event cuối [TĐ]. `resolved_at` có ở 93,8%.
  - Tức đây là log "enriched", các thuộc tính trạng thái cuối được điền ngược vào mọi event.
  - `knowledge` ổn định 99,2%, cùng kiểu với các trường điền ngược. Trong khi đó `assignment_group` chỉ khớp first-last 59%, `u_priority_confirmation` chỉ khớp 23%.
  - Vì vậy không có bằng chứng T được đặt lúc mở. Ngữ nghĩa khả dĩ nhất theo UCI là "KB đã được dùng để giải quyết", tức sự kiện xảy ra trong lúc xử lý.
  - Thời gian tới khi resolve có trung vị **94,1 giờ khi T=1 so với 10,6 giờ khi T=0** [TĐ]. Ticket khó và kéo dài mới phải tra KB, nên −45 điểm % chủ yếu là nhân quả ngược hoặc lựa chọn theo độ khó.
  - Thêm một dấu hiệu "loại ticket khác": `u_priority_confirmation` cuối là 22% ở T=1 so với 86% ở T=0 [TĐ].
- **Chiều của made_sla.** Đã kiểm tra [TĐ]: 24.917/24.918 ticket bắt đầu với true, 9.114 ticket chuyển true→false, không có ticket nào false→true. Vậy true nghĩa là đạt SLA (đóng trước hạn).
  - Theo priority (bản ghi đầu), P(made_sla): Moderate 63,9%, Low 85,3%, High 2,9%, Critical 19,5%. Scout ghi 0,5–1,9% là số theo priority ở bản ghi cuối.
- **(b) Positivity.** Category 45 có 89% treated, Category 28 và 35 chỉ 1–1,6%; Group 64 có 0,6%. Phải gộp hoặc cắt.
- **(c) Kỹ thuật.**
  - Y có parent là T, priority(3), category(9), group(6), opened_by(5), tức 1.620 cấu hình. TRAIN khoảng 17,4k dòng, nên khoảng 11 dòng mỗi ô. H1 có việc để làm.
  - Cần khoảng 40 dòng code để gộp event về mức ticket.
- **(d) Thực tế.** Câu hỏi "KB hoặc trợ lý AI có giúp đạt SLA" rất thời sự năm 2026. Nhưng **do(T) ở đây không phải chính sách "bắt buộc tra KB lúc mở"** mà là một dấu vết hậu kiểm.
- **Kết luận.** Chỉ giữ 498 nếu tìm được tài liệu chứng minh cờ được đặt lúc tiếp nhận. Nếu không, mọi ATE có dấu âm sẽ bị hội đồng bác bỏ ở Phase 1.

### 603 In-Vehicle Coupon: tốt nhất về can thiệp, xử lý được cluster
- **(a) Pseudo-ID: làm được.** Dữ liệu được sắp theo người trả lời.
  - Đánh dấu một block mới mỗi khi bộ cột nhân khẩu học + thói quen thay đổi, được **587 block liên tiếp**, trong đó 517 block có đúng 22 dòng [TĐ].
  - Nếu khoá theo 13 cột nhân khẩu học + thói quen thì có 567 nhóm (528 nhóm cỡ khoảng 20).
  - Kết luận: dùng block-ID cho GroupShuffleSplit với khoảng 587 nhóm. Mỗi split 70/15/15 có khoảng 88 người trong TEST.
- **Cơ chế gán T là thiết kế bảng hỏi.** Có **19 phiên bản** chuỗi 22 kịch bản [TĐ], và T được cố định theo phiên bản.
  - Trong ô kịch bản đầy đủ (6 biến thiết kế), chỉ 36/138 ô có cả hai giá trị T. Số dòng nằm trong các ô có đủ hai giá trị chỉ là 52%.
  - Tức là với 48% dòng, T xác định hoàn toàn bởi kịch bản (positivity vỡ nếu điều chỉnh đủ 6 biến).
  - Ở mức coupon×time, 95% dòng nằm trong ô có đủ hai giá trị. Ở mức destination×time×coupon là 93%.
  - Vậy **tập C phải thô**: coupon, destination hoặc time, passenger, weather. Bỏ ô Bar×Work (0/487) hoặc gộp Bar.
- **(c) Kỹ thuật.**
  - DAG khoảng 10–12 nút.
  - Y có parent là T, coupon(5), dest(3), passenger(4), weather(3), tức 360 cấu hình. TRAIN khoảng 8,9k dòng, nên khoảng 25 dòng mỗi ô, nhưng nhiều ô bị trống theo thiết kế. H1 có việc để làm (làm mượt các ô trống).
  - Phần H2 thú vị: NOTEARS phải tìm được các cạnh kịch bản→T, và kiểm tra được điều đó.
- **(d) Thực tế.** Người dùng kết quả là marketer hoặc nền tảng ưu đãi in-car/app. T thao tác trực tiếp được. Điểm yếu: Y là **ý định** trong khảo sát MTurk năm 2017.

### 332 Online News: hợp lệ nhưng DAG tầm thường
- **(a) Bài cuối tuần không phải "loại bài khác hẳn".**
  - Tỉ lệ cuối tuần theo channel là 9,4–18,7%, theo LDA topic là 11–15% [TĐ].
  - Bài cuối tuần dài hơn (trung vị 482 so với 400 token), nhiều link hơn (10 so với 7), số ảnh và video như nhau.
  - Chênh lệch chuẩn hoá theo channel (+21,1) gần bằng chênh lệch thô. Tức là **C gần như không gây nhiễu T**.
  - Hiệu ứng chủ yếu là cơ chế "ít bài cạnh tranh hơn": Thứ Bảy 74,5%, Chủ nhật 68,7%, các ngày thường 48,7–54,6% [TĐ].
  - Nhiễu ẩn chính là chiến lược biên tập (để dành bài "evergreen" cho cuối tuần), không đo được.
- **(b) Positivity.** Tầng xấu nhất là timedelta >720 ngày với 6,2% (n=706); các tầng còn lại đều trên 9%.
- **(c) Kỹ thuật.**
  - Phải bin khoảng 8 biến liên tục; tốn công nhưng làm được.
  - T gần như không có parent, nên **ATE B0 ≈ H1 ≈ H2 ≈ chênh lệch thô**. H2 chỉ khác ở phần cấu trúc quanh Y. Kết quả so sánh phương pháp sẽ nghèo nàn.
  - Y có parent là T, channel(7), LDA(5), tokens(4), imgs(3), tức 840 cấu hình. TRAIN khoảng 27,7k dòng, nên khoảng 33 dòng mỗi ô.
- **(d) Thực tế.** "Share mạng xã hội" của Mashable 2013–14 đã lỗi thời so với thuật toán phân phối năm 2026. Biên tập viên vẫn chọn lịch đăng.

### 697 Student Dropout: timing học bổng đáng ngờ, lĩnh vực trùng mẫu
- **(a) Tương quan bất thường với biến học kỳ** [TĐ]:
  - Tỉ lệ có học bổng là 10,4% ở nhóm đỗ 0–1 tín chỉ kỳ 1, so với 35,5% ở nhóm đỗ ≥6. Kỳ 2 cũng tương tự: 10,2% so với 37,5%.
  - Ở nhóm học phí không đúng hạn, tỉ lệ có học bổng chỉ 8,7%, so với 27,0% ở nhóm đóng đúng hạn.
  - Gradient này có thể do confounding (năng lực), nhưng cũng khớp với cơ chế gia hạn học bổng DGES có điều kiện thành tích. Khi đó T là trạng thái ghi sau các học kỳ, tức nhân quả ngược.
  - Bằng chứng ngược chiều yếu: sinh viên đăng ký 0 tín chỉ kỳ 1 có tỉ lệ học bổng 24,4%, bằng mức chung.
  - **Không thể quyết định từ dữ liệu**, nên đây là rủi ro Phase 1 lớn nhất.
- **(b) Positivity.**
  - Application mode 7 có 0,7%, Previous qualification 3 có 1,6%, nhóm 25+ có 10,1% (trong đó 25+ học ban ngày là 9,3%).
  - Khoá nhập học, suy ra từ bộ ba biến vĩ mô (10 khoá), có tỉ lệ học bổng 11–31%. Năm nhập học là confounder cần đưa vào.
- **(c) Kỹ thuật.**
  - TRAIN khoảng 3,1k dòng. Y có parent là T, age(3), gender(2), admission grade(3), course group(4), tức 144 cấu hình, nên khoảng 21 dòng mỗi ô.
  - Ô T=1 ở tuổi 25+ chỉ có khoảng 80 dòng TRAIN.
- **(d) Thực tế/Mới.** Chính sách học bổng rất thực tế. Nhưng **gần như trùng với ví dụ mẫu** (giáo dục, T là một dạng hỗ trợ, giống schoolsup→pass).

### 891 CDC BRFSS: kỹ thuật hoàn hảo, can thiệp yếu
- **(a)**
  - "Smoker" nghĩa là đã hút ≥100 điếu trong đời: phơi nhiễm tích luỹ, không có thời điểm quyết định.
  - Thiết kế cross-sectional: HeartDisease là "đã từng bị", nên có survivor bias (người chết vì bệnh tim không còn trong khảo sát).
  - Income và Education có thể là hệ quả của bệnh.
- **(b)** Positivity tốt ở các biên (21,8–62,2%). Ô 4 chiều xấu nhất là 3,8% / 89,7%.
- **(c)** TRAIN khoảng 160–178k dòng; Y có parent là T, age(5), sex, edu(3), income(3), tức 180 cấu hình, nên khoảng 1.000 dòng mỗi ô.
  - **H1 không có việc để làm** vì CPD đã chính xác.
  - H2 với các biến nhị phân cross-sectional sẽ định hướng cạnh tuỳ ý nếu không có mask. Mask gần như quyết định toàn bộ.
  - Cần quyết định có xoá 24.206 dòng trùng hay không.
- **(d)** Chính sách y tế công cộng chỉ thao tác được "phòng ngừa bắt đầu hút", không thao tác được biến ever-smoker.

### 1025 Turkish Crowdfunding: can thiệp đẹp, quá nhỏ
- **(b)** Chỉ có 34 dự án thành công trong nhánh không video. Ở TRAIN còn khoảng 24, rải trên khoảng 50 cấu hình parent, nên phần lớn ô (T=0, ·) có 0–1 ca dương.
  - Positivity theo platform: arıkovanı 96,6%, crowdfon 18%, crowdfon×2015 1,7%.
  - Khác biệt giữa các platform lớn đến mức có thể là **artefact cào dữ liệu** (một số trang không hiển thị video).
- **(a)** Video được quyết định trước khi chiến dịch bắt đầu, timing rõ. Nhiễu ẩn "chất lượng/độ đầu tư" ảnh hưởng cả video lẫn web_sitesi và sosyal_medya: có web 77% video so với 52% không có.
- **(c)** Y có parent là T, platform(3), year(3), social(2), goal(3), tức 108 cấu hình. TRAIN khoảng 1.140 dòng, nên khoảng 10 dòng mỗi ô.
  - H1 có rất nhiều việc để làm.
  - Phương sai ATE giữa 5 split sẽ lớn (TEST khoảng 245 dòng, khoảng 8 ca thành công không video).
- **(d)** Nhà sáng lập quyết định trực tiếp. Bối cảnh Thổ Nhĩ Kỳ 2013–2020 vẫn còn hợp lệ.

## 3. Chấm điểm lại (độc lập với scout)

| ID | Thực tiễn | Can thiệp | Confounding/timing | Kỹ thuật | Mới | **Tổng** | Scout |
|---|---|---|---|---|---|---|---|
| 603 | 3 | 5 | 4 | 4 | 4 | **20** | 21 |
| 296 | 4 | 3 | 3 | 4 | 5 | **19** | 22 |
| 332 | 3 | 4 | 3 | 4 | 4 | **18** | 20 |
| 498 | 4 | 2 | 2 | 4 | 5 | **17** | 21 |
| 1025 | 3 | 5 | 3 | 2 | 4 | **17** | 19 |
| 891 | 3 | 2 | 3 | 5 | 3 | **16** | 19 |
| 697 | 5 | 3 | 2 | 3 | 2 | **15** | 19 |

Lý do từng tiêu chí, mỗi tiêu chí một dòng:

- **603**
  - Thực tiễn 3: thời hạn khuyến mãi là quyết định marketing thật, nhưng Y là ý định giả định trên MTurk 2017.
  - Can thiệp 5: doanh nghiệp đặt T trực tiếp.
  - Confounding 4: cơ chế gán T đo được hoàn toàn (19 phiên bản bảng hỏi). Trừ 1 vì cluster theo người và positivity vỡ nếu dùng đủ C.
  - Kỹ thuật 4: toàn biến phân loại, tạo được pseudo-ID (587 block). Có ô cấu trúc bằng 0.
  - Mới 4: marketing, gần Bank Marketing hơn scout thừa nhận.
- **296**
  - Thực tiễn 4: tái nhập viện vẫn là KPI, nhưng phác đồ 1999–2008 đã cũ.
  - Can thiệp 3: `change` trộn tăng, giảm, thêm, ngừng thuốc (45% số Ch không có Up/Down).
  - Confounding 3: thiếu mức độ nặng và ID bệnh viện.
  - Kỹ thuật 4: n lớn, nhưng phải cắt population và ATE khoảng 0.
  - Mới 5: lâm sàng.
- **332**
  - Thực tiễn 3: lên lịch đăng là quyết định thật, nhưng dữ liệu share năm 2013 đã cũ.
  - Can thiệp 4: ngày đăng thao tác được.
  - Confounding 3: C đo trước T, nhưng hầu như không gây nhiễu T nên DAG tầm thường; nhiễu ẩn là chiến lược biên tập.
  - Kỹ thuật 4: n lớn, phải bin nhiều.
  - Mới 4: truyền thông số.
- **498**
  - Thực tiễn 4: KB/AI trong ITSM rất thời sự.
  - Can thiệp 2: cờ được điền ngược từ trạng thái cuối, ngữ nghĩa là "đã dùng để giải quyết".
  - Confounding 2: nhân quả ngược theo độ khó (trung vị 94 giờ so với 10,6 giờ).
  - Kỹ thuật 4: n đủ, nhưng cần gộp event log và cắt category 1% / 89%.
  - Mới 5: ITSM.
- **1025**
  - Thực tiễn 3: quyết định của nhà sáng lập, có giá trị nhưng tầm ảnh hưởng hẹp.
  - Can thiệp 5: một hành động rời rạc, rõ ràng.
  - Confounding 3: timing rõ, nhưng chất lượng dự án là nhiễu ẩn.
  - Kỹ thuật 2: n=1.628, 34 ca thành công không video, positivity theo platform 1,7–97%.
  - Mới 4: crowdfunding.
- **697**
  - Thực tiễn 5: chính sách hỗ trợ tài chính để giảm bỏ học.
  - Can thiệp 3: cấp học bổng theo quy tắc thu nhập, nên do() nghĩa là đổi tiêu chí xét.
  - Confounding 2: timing không rõ, gradient mạnh với kết quả học kỳ, thu nhập ẩn.
  - Kỹ thuật 3: TRAIN khoảng 3,1k, có tầng 0,7–1,6%.
  - Mới 2: gần như bản sao của schoolsup→pass.
- **891**
  - Thực tiễn 3: phòng chống thuốc lá là ưu tiên thật, nhưng kết luận đã quá quen.
  - Can thiệp 2: ever-smoker không có thời điểm và người quyết định.
  - Confounding 3: C nhân khẩu học tốt nhưng thiết kế cross-sectional.
  - Kỹ thuật 5: n rất lớn, không missing.
  - Mới 3: dataset Kaggle phổ biến.

**Xếp hạng:** 1. 603 (20), 2. 296 (19, với điều kiện đổi T sang chỉ định HbA1c hoặc insulin Up vs Steady), 3. 332 (18), 4. 1025 (17), 5. 498 (17, có điều kiện), 6. 891 (16), 7. 697 (15).
- 1025 xếp trên 498 vì vấn đề của 1025 là phương sai, chữa được bằng H1 và gộp mức. Vấn đề của 498 là định nghĩa T, không chữa được bằng phương pháp.

**Chỗ tôi bất đồng với scout:**
1. **296.** Overlap "tốt ở mọi tầng" là sai: diabetesMed=No có T≡0. Chênh lệch thô thật trong population hợp lệ là +0,05 điểm %, không phải 0,8. `change` không phải một hành động thuần. Can thiệp 4→3, tổng 22→19, nhưng vẫn đứng thứ 2 nhờ n và tính mới.
2. **498.** Bằng chứng "T có ở event đầu" bị vô hiệu vì log điền ngược `closed_at` và `resolved_by` vào 100% event đầu. Đây là nhân quả ngược khả dĩ nhất trong 7 ứng viên. Tổng 21→17.
3. **603.** Scout nói không có respondent ID. Thực ra tái tạo được 587 block, nên cluster không còn là điểm trừ lớn. Ngược lại, scout bỏ sót việc T là hàm tất định của kịch bản ở 48% dòng.
4. **697.** Scout chấm Mới 3 và Confounding 3. Tôi chấm 2 và 2 vì cùng lĩnh vực với mẫu và có gradient học bổng theo kết quả học kỳ. Tổng 19→15.
5. **332.** Đồng ý là hợp lệ, nhưng scout bỏ qua việc C gần như không giải thích T. Đồ án sẽ không thể hiện được giá trị của điều chỉnh confounder hay của H1/H2.
6. **891.** Scout chấm Can thiệp 3; tôi chấm 2. Scout không nhắc 24.206 dòng trùng.

## 4. Phát biểu đề tài (5 ứng viên cao nhất)

**603 In-Vehicle Coupon.**
- Population: các phản hồi khảo sát MTurk năm 2017 (Wang et al., JMLR 2017), n=12.684 lượt từ khoảng 587 người. Loại ô Bar×Work (487 dòng) vì không có coupon 2h.
- T: thời hạn coupon hiển thị trên xe. T=1 là "2h", T=0 là "1d". T do bên phát coupon chọn tại thời điểm hiển thị.
- Y: người trả lời chấp nhận coupon (Y=1). Đo ngay sau khi coupon được hiển thị.
- Estimand: ATE = P(Y=1|do(T=2h)) − P(Y=1|do(T=1d)) trên TEST cohort.
- Split: 5 lần 70/15/15 theo nhóm, dùng pseudo-ID là block liên tiếp của các cột nhân khẩu học.
- Baseline C:
  - Biến thiết kế kịch bản (cơ chế gán T): coupon (5 mức), destination, time (gộp sáng/chiều/tối), passenger, weather.
  - Đặc điểm của người trả lời có trước kịch bản: age (bin 3 mức), income (bin 3 mức), tần suất đi các địa điểm tương ứng với loại coupon.
- Biến phải loại: car (99% NaN), toCoupon_GEQ5min (hằng số), direction_opp (bằng 1−direction_same), temperature (trùng với weather), occupation (25 mức, chỉ dùng nếu gộp lại).
- toCoupon_GEQ15/25min và direction_same xảy ra đồng thời với T. Coi chúng là thuộc tính kịch bản, không phải mediator.
- Không điều chỉnh đủ 6 biến kịch bản cùng lúc vì sẽ vỡ positivity.

**296 Diabetes 130-US (bản sửa).**
- Population: lượt nhập viện đầu tiên của mỗi bệnh nhân đái tháo đường tại 130 bệnh viện Mỹ giai đoạn 1999–2008, xuất viện còn sống (n=69.973).
- T: bác sĩ chỉ định xét nghiệm HbA1c trong đợt nằm viện. T=1 là A1Cresult ∈ {Norm, >7, >8}, T=0 là "None". T xảy ra trong đợt nằm viện, trước xuất viện.
- Y: tái nhập viện trong vòng 30 ngày (readmitted = "<30"). Đo sau xuất viện.
- Estimand: ATE = P(Y=1|do(T=1)) − P(Y=1|do(T=0)) trên TEST cohort.
- Baseline C:
  - age (4 bin), race, gender.
  - admission_type (gộp 3 mức), admission_source (ER/chuyển viện/khác/không rõ).
  - number_inpatient và number_emergency của năm trước (0/1/2+).
  - Nhóm ICD của diag_1.
  - payer_code (gộp).
- Biến phải loại:
  - Mediator hoặc biến đồng thời/sau T: discharge_disposition, time_in_hospital, num_lab_procedures, num_procedures, num_medications.
  - change, diabetesMed và các cột thuốc: quyết định điều trị có thể là hệ quả của kết quả HbA1c.
  - Thiếu dữ liệu nặng: weight, medical_specialty.
  - Không dùng: diag_2, diag_3, encounter_id, patient_nbr.
- Phương án phụ: T = insulin Up so với Steady, trên population đang dùng insulin.

**332 Online News Popularity.**
- Population: các bài Mashable đăng năm 2013–2014, giữ bài có timedelta ≥ 21 ngày để số share đã ổn định (n khoảng 38k).
- T: thời điểm đăng do biên tập viên lên lịch. T=1 là cuối tuần (is_weekend), T=0 là ngày thường.
- Y: bài đạt ≥ 1.400 share (trung vị). Đo tại thời điểm thu thập dữ liệu.
- Estimand: ATE = P(Y=1|do(T=1)) − P(Y=1|do(T=0)) trên TEST cohort.
- Baseline C (đặc tính nội dung có trước khi đăng):
  - data_channel (7 mức), LDA topic chính (5 mức).
  - n_tokens_content (4 bin), num_imgs (0/1/2+), num_hrefs (bin 3 mức).
  - timedelta (bin 3 mức), để kiểm soát thời gian bài tồn tại.
- Biến phải loại:
  - kw_* và self_reference_*: aggregate của share, gây leakage.
  - weekday_is_*: là định nghĩa của T.
  - url.
- Ghi rõ giả định: "khối lượng bài cạnh tranh trong ngày" là cơ chế của T, không phải confounder.

**1025 Turkish Crowdfunding.**
- Population: các dự án crowdfunding dạng phần thưởng (ödül) trên các nền tảng Thổ Nhĩ Kỳ giai đoạn 2013–2020 (n=1.618).
  - Loại 10 dự án dạng quyên góp (bağış).
  - Cân nhắc loại platform arıkovanı (n=58, 97% có video).
- T: nhà sáng lập đăng video giới thiệu (tanitim_videosu=var) hay không (yok). T được quyết định khi tạo trang dự án.
- Y: dự án đạt mục tiêu gọi vốn (basari_durumu = başarılı). Đo khi chiến dịch kết thúc.
- Estimand: ATE = P(Y=1|do(T=1)) − P(Y=1|do(T=0)) trên TEST cohort.
- Baseline C:
  - platform_adi (3 mức), yil (gộp 3 mức: 2013–15, 2016–18, 2019–20).
  - kategori (gộp khoảng 4 mức).
  - hedef_miktari (tertile).
  - fon_sekli.
  - Kinh nghiệm của chủ dự án: kac_projenin_sahibi > 1.
  - Tuỳ chọn thêm web_sitesi, sosyal_medya, gun_sayisi.
- Biến phải loại:
  - Leakage hoặc đo sau T: destekci_sayisi, toplanan_tutar, destek_orani, yorumlar, guncellemeler, sm_takipci.
  - video_uzunlugu: chỉ có khi T=1.
  - Không dùng: id, proje_adi, proje_sahibi, proje_aciklamasi.
- Bắt buộc báo cáo phương sai ATE giữa 5 split.

**498 Incident management (có điều kiện).**
- Population: các incident IT của một service desk ServiceNow, mở từ 02/2016 đến 02/2017. Gộp event log về mức ticket (n=24.918).
- T: cờ knowledge (sử dụng tài liệu KB). T=1 là true, T=0 là false.
- Y: ticket được đóng trong hạn SLA (made_sla=true ở bản ghi cuối). Đo khi đóng ticket.
- Estimand: ATE = P(Y=1|do(T=1)) − P(Y=1|do(T=0)) trên TEST cohort.
- Phải phát biểu rõ giả định timing: T được đặt lúc tiếp nhận. Đây là giả định chưa kiểm chứng được, vì log điền ngược trạng thái cuối vào mọi event. Bắt buộc có phân tích độ nhạy và thảo luận nhân quả ngược theo độ khó.
- Baseline C (giá trị ở bản ghi đầu):
  - priority (gộp Critical+High, Moderate, Low).
  - category (top-8 + other, loại hoặc gộp Category 45/40).
  - assignment_group (top-6 + other).
  - opened_by (top-4 + other).
  - Ca và ngày trong tuần lúc mở ticket.
- Biến phải loại:
  - Mediator hoặc đo sau T: reassignment_count, reopen_count, sys_mod_count, u_priority_confirmation, assigned_to, các trường priority/impact/urgency ở bản ghi cuối, incident_state.
  - Leakage: resolved_by, resolved_at, closed_at, closed_code, sys_updated_*.
  - Thiếu dữ liệu >98%: problem_id, rfc, vendor, caused_by, cmdb_ci.
  - notify: hầu như chỉ có một giá trị.
- Nếu không xác nhận được ngữ nghĩa của cờ knowledge thì thay ứng viên này bằng 891 (T=Smoker, Y=HeartDiseaseorAttack, C = Age, Sex, Education, Income, AnyHealthcare; loại HighBP, HighChol, BMI, Diabetes, GenHlth, PhysHlth, DiffWalk, Stroke).
