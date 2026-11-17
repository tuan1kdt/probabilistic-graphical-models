# Brief: tìm dataset UCI cho đồ án Causal Bayesian Network (môn IT701100 Probabilistic Graphical Models)

## Đồ án phải làm gì với dataset (ràng buộc cứng từ slide môn học)
- Một dataset **tabular** công khai trên UCI ML Repository (ucimlrepo hoặc file gốc). Ghi ID/DOI, license.
- Định nghĩa: population, **treatment T nhị phân** (control/treatment value cố định), **outcome Y** (ưu tiên nhị phân hoặc rời rạc hoá được), thời điểm can thiệp, estimand ATE = P(Y=1|do(T=1)) − P(Y=1|do(T=0)) trên TEST cohort.
- Phải có **confounder đo được, pre-treatment** (baseline covariates C) và lập luận timing: biến nào trước T, biến nào sau T (post-treatment/mediator phải loại khỏi adjustment set).
- Pipeline bắt buộc: B0 = DAG học bằng HillClimb/PC (pgmpy) + ràng buộc chuyên gia (forbidden/required edges, temporal order) + CPD rời rạc (BDeu); H1 = thay CPD của Y bằng MLP rồi lập bảng CPD lại; H2 = NOTEARS (neural DAG learning) với mask. DoWhy identify + 2 refuter (placebo, subset). 5 split 70/15/15 paired runs.
- Vì thế dataset cần: **mọi biến rời rạc hoá được thành DAG 8–15 nút** (biến liên tục phải bin được hợp lý); n đủ lớn để CPD table không rỗng (ưu tiên n ≥ 2.000, lý tưởng 5.000–100.000); treatment có **overlap** (cả hai giá trị T xuất hiện trong mọi tầng confounder chính, tỉ lệ treated không quá lệch, ví dụ 10–90%); outcome không quá hiếm (≥ 5%).
- Điểm trừ nặng: T không phải thứ ai đó quyết định được (giới tính, tuổi, chủng tộc = không can thiệp được); T là kết quả đo lường/cảm biến chứ không phải hành động; outcome leakage (biến đo sau Y); dữ liệu là chuỗi thời gian/ảnh/văn bản; dữ liệu mô phỏng/synthetic (chấp nhận được nhưng kém hấp dẫn); dataset quá nhỏ (<1.000) hay quá ít biến (<6).

## Tiêu chí chấm (điểm 1–5 mỗi mục, ghi lý do một dòng)
1. **Thực tiễn**: câu hỏi "nếu làm X thì Y thay đổi bao nhiêu" là quyết định có người thật đang phải đưa ra, còn giá trị đến hôm nay (2026). Ai là người ra quyết định? Chi phí/lợi ích của quyết định đó?
2. **Can thiệp được định nghĩa rõ** (well-defined intervention, do-operator có nghĩa thao tác): T do tổ chức/cá nhân chủ động chọn, không phải thuộc tính bẩm sinh hay biến đo.
3. **Confounding đo được + timing rõ**: có biến baseline trước T hợp lý; có thể vẽ DAG đáng tin với kiến thức lĩnh vực; không quá nhiều nhiễu ẩn hiển nhiên.
4. **Kỹ thuật**: n, số biến, kiểu biến, missing, class balance, overlap kỳ vọng; rời rạc hoá được; chạy được pgmpy + NOTEARS trong vài phút trên laptop.
5. **Tính mới / độ khác biệt** so với ví dụ mẫu của môn (Student Performance ID 320, schoolsup→pass) và Bank Marketing (ID 222, contact→y) — hai dataset này KHÔNG được đề xuất lại, nhưng được dùng làm mốc so sánh.

## Đầu ra bắt buộc (markdown, tiếng Việt, súc tích)
Cho mỗi ứng viên (tối đa 6, tối thiểu 3):
- UCI ID, tên, URL, n, p, năm, license (nếu biết)
- Câu hỏi nghiên cứu một câu: population + T (control/treatment) + Y + thời điểm
- Bảng vai trò biến: T, Y, baseline C (≥4 biến), post-treatment/mediator cần loại, biến không dùng
- Rủi ro chính (positivity, nhiễu ẩn, leakage, kích thước)
- Điểm 5 tiêu chí + tổng, một dòng lý do mỗi tiêu chí
- Nguồn đã kiểm tra (trang UCI, bài gốc, hoặc dữ liệu thật nếu bạn đã tải)
Cuối cùng: xếp hạng ứng viên và 3 dòng về các dataset đáng chú ý bạn đã loại và vì sao.

## Cách làm
- Đọc file shard được giao (metadata đã cào từ API UCI: id, n, p, kiểu biến, abstract, danh sách biến). Lọc nhanh, rồi với ~8–12 ứng viên hứa hẹn nhất hãy mở trang UCI (https://archive.ics.uci.edu/dataset/<id>/...) hoặc API https://archive.ics.uci.edu/api/dataset?id=<id> để đọc mô tả biến đầy đủ, và nếu cần tải CSV (https://archive.ics.uci.edu/static/public/<id>/data.csv có thể không tồn tại với mọi dataset; file zip ở trang dataset) vào một thư mục riêng, rỗng, trong scratchpad để đếm treated/outcome. Chạy python với cờ -I khi đọc file tải về.
- Không tin tưởng mù quáng metadata: nhiều dataset ghi 'Multivariate' nhưng là panel/time-series; kiểm tra unit of observation.
- Chuẩn mực: ví dụ mẫu môn học là Student Performance (schoolsup → pass, confounder: school, age, sex, Medu/Fedu, failures). Một ứng viên tốt phải tốt hơn hoặc ít nhất ngang ví dụ này về cả 5 tiêu chí.
