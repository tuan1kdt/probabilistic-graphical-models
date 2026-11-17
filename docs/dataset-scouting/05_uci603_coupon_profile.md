# Hồ sơ dataset UCI 603 — In-Vehicle Coupon Recommendation

Ngày: 2026-11-02. Mọi con số đếm trực tiếp trên `data.csv` tải từ UCI (12.684 dòng, 26 cột). Bài báo gốc: Wang, Rudin, Doshi-Velez, Liu, Klampfl, MacNeille. *A Bayesian Framework for Learning Rule Sets for Interpretable Classification.* JMLR 18(70):1–37, 2017. CC BY 4.0. Hai tác giả cuối thuộc Ford Motor Company.

## 1. Bài báo nói gì

Bài báo không phải về coupon. Đóng góp chính là thuật toán **Bayesian Rule Sets (BRS)**: học bộ phân loại dạng tuyển của các hội (DNF) gồm ít luật ngắn, có prior Bayes cho phép người dùng điều khiển số luật và độ dài luật, suy diễn MAP bằng khai phá luật kết hợp rồi simulated annealing, kèm chặn lý thuyết để tỉa không gian tìm kiếm. Dataset coupon là **ứng dụng minh hoạ** ở mục 6.2 và phụ lục B.

Phần liên quan trực tiếp đến dataset:

- **Bối cảnh.** Hệ gợi ý trong xe (in-vehicle context-aware recommender) đẩy coupon giảm giá 20% của một cơ sở gần đó lên màn hình xe. Câu hỏi của Ford là người lái có nhận coupon hay không tuỳ bối cảnh lái xe và đặc điểm người lái.
- **Thu thập.** Khảo sát trên Amazon Mechanical Turk, chỉ nhận Turker có rating ≥ 95%. 752 bài khảo sát, 652 được chấp nhận, sinh ra 12.684 dòng sau khi bỏ dòng thiếu thuộc tính. Phần 1 hỏi nhân khẩu học và thói quen; phần 2 mô tả **20 kịch bản lái xe** cho mỗi người (kèm bản đồ vị trí xe, đích đến, quán, thời gian lái) và hỏi "bạn có dùng coupon này không".
- **Nhãn.** "Đi ngay" hoặc "đi sau, trước khi coupon hết hạn" → Y = 1. "Không, tôi không muốn coupon" → Y = 0.
- **Thiết kế bảng hỏi.** Bảng hỏi chia nhiều phiên bản; người không có con không bao giờ gặp kịch bản "có trẻ em trong xe". Thuộc tính coupon duy nhất được biến thiên là **thời hạn: 2 giờ hoặc 1 ngày**.
- **Kết quả của họ.** Xây 5 bộ phân loại riêng cho 5 loại coupon. AUC 5-fold của BRS 0,68–0,78, ngang Lasso (dùng ~93 hệ số) và hơn C4.5/CART/RIPPER. Nhận xét định tính: thói quen đến đúng loại quán là thuộc tính mạnh nhất, xuất hiện trong mọi luật. Ví dụ luật: "nếu đi bar ≥ 1 lần/tháng, không chở trẻ em, nghề không phải nông/ngư/lâm nghiệp thì dự đoán nhận coupon bar".
- **Họ không làm gì.** Không có phân tích nhân quả, không ước lượng hiệu ứng của thời hạn coupon, không xử lý việc một người trả lời 20 dòng (coi mọi dòng là độc lập khi chia fold).

## 2. Dữ liệu thực tế trên UCI

### Cấu trúc

| Nhóm | Biến (số mức) | Ghi chú |
|---|---|---|
| Kịch bản (do bảng hỏi gán) | destination (3), passenger (4), weather (3), temperature (3), time (5), coupon (5), **expiration (2)**, toCoupon_GEQ15min, toCoupon_GEQ25min, direction_same | Mỗi dòng là một kịch bản; 138 tổ hợp bối cảnh khác nhau, 201 kịch bản đầy đủ khi tính cả expiration |
| Người trả lời | gender (2), age (8), maritalStatus (5), has_children (2), education (6), occupation (25), income (9), car, Bar, CoffeeHouse, CarryAway, RestaurantLessThan20, Restaurant20To50 (mỗi thói quen 5 mức + thiếu) | Lặp lại y hệt trong ~20–22 dòng liên tiếp của cùng một người |
| Kết quả | Y (2) | 57% là 1 |

### Số liệu then chốt

- Tỉ lệ nhận coupon theo loại: Carry out 73,5%, Restaurant(<20) 70,7%, Coffee House 49,9%, Restaurant(20–50) 44,1%, Bar 41,0%.
- Tỉ lệ nhận theo thời hạn: 1 ngày 62,6%, 2 giờ 49,6%. Chênh thô −13 điểm %.
- Chênh lệch theo từng loại coupon (1d → 2h): Restaurant(<20) 83,2 → 58,9; Restaurant(20–50) 52,1 → 29,9; Coffee House 58,4 → 43,2; Carry out 78,2 → 66,4; **Bar 41,2 → 40,3** (gần bằng 0).
- Khoảng cách: Y giảm dần 61,4% (<15 phút) → 56,1% (15–25) → 42,9% (>25).
- **Thời hạn không phụ thuộc người trả lời**: P(2h) = 44% ở cả nam lẫn nữ, 40–47% ở mọi mức thu nhập, trung bình 44% trong mỗi block người (dao động 12–67% vì mỗi người chỉ thấy 20 kịch bản). Thời hạn **phụ thuộc kịch bản**: Bar 23%, Coffee House 56%; ô Bar × Work có 0/487 dòng là 2h.
- 587 block người trả lời (517 block đúng 22 dòng), 567 khoá nhân khẩu học khác nhau. Không có cột ID nhưng tái tạo được bằng cách gom dòng liên tiếp cùng 13 cột nhân khẩu học.
- 74 dòng trùng hoàn toàn. `car` thiếu 99%. `toCoupon_GEQ5min` là hằng số. `direction_opp` = 1 − `direction_same`. Bar/CoffeeHouse/CarryAway/Restaurant* thiếu 0,3–1,7%.

### Khác biệt giữa dữ liệu và phụ lục B của bài báo

- Phụ lục ghi time có 3 mức (10AM, 2PM, 6PM); dữ liệu có 5 mức (thêm 7AM, 10PM).
- Phụ lục liệt kê "số lần đi bar" hai lần, bỏ sót nhà hàng $20–50; dữ liệu có đủ 5 thói quen.
- Phụ lục nói 652 người, dữ liệu gom được 587 block. Chênh lệch do bỏ dòng thiếu và do hai người liền kề có nhân khẩu học giống hệt bị gộp.

## 3. Ý nghĩa cho đồ án Causal BN

**Treatment** T = expiration (1d = 0, 2h = 1). **Outcome** Y = chấp nhận coupon.

- **Vì sao do(T) có nghĩa.** Thời hạn là thuộc tính coupon duy nhất bảng hỏi biến thiên, và là tham số nhà phát coupon đặt được với chi phí 0. Dữ liệu cho thấy T độc lập với mọi đặc điểm người trả lời, nên **confounder chỉ nằm trong các biến kịch bản** (coupon, destination, time, passenger, weather). Đây là tình huống "quasi-randomized theo thiết kế", hiếm có trong dữ liệu quan sát.
- **DAG kỳ vọng.** Biến thiết kế → T; biến thiết kế → Y; thói quen và nhân khẩu học → Y (không → T); khoảng cách/hướng → Y. Học cấu trúc (B0) và NOTEARS (H2) phải khôi phục được việc **không có cạnh từ nhân khẩu học vào T**; đây là một kiểm tra ngoài rất rõ cho H2.
- **Estimand.** ATE = P(Y=1 | do(2h)) − P(Y=1 | do(1d)) trên TEST cohort, điều chỉnh theo coupon và destination (hoặc time). Hiệu ứng rất có thể **khác nhau theo loại coupon** (Bar ≈ 0, nhà hàng rẻ ≈ −24 điểm %), nên nên báo cáo thêm CATE theo coupon; MLP-CPD của H1 có lợi thế tự nhiên ở đây.
- **Positivity.** Loại ô Bar × Work (0% 2h). Ở mức tổ hợp đầy đủ 6 biến kịch bản chỉ 36/138 ô có cả hai giá trị T (52% dòng); ở mức coupon × time thì 95% dòng có overlap. Tập C phải thô, không điều chỉnh cả 6 biến thiết kế cùng lúc.
- **Độc lập quan sát.** Phải split theo block người trả lời (tái tạo như trên), không split theo dòng như bài báo gốc.
- **Hạn chế phải ghi.** Y là ý định trong tình huống giả định (hypothetical choice), không phải hành vi thật; mẫu MTurk 2015–2017 ở Mỹ; temperature gần như trùng weather; 74 dòng trùng cần quyết định giữ hay bỏ.

## 4. Nguồn

- UCI: https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation (DOI 10.24432/C5GS4P, tạo 2017, CC BY 4.0)
- Bài báo: https://jmlr.org/papers/volume18/16-003/16-003.pdf
- Dataset gốc được công bố riêng: Wang et al. 2015, "In-vehicle coupon recommendation dataset", ResearchGate 318645502 (trích trong bài báo)
