# Báo cáo scout: Engineering / CS / Physics-Chemistry / Climate / Other (+ vớt từ nhóm bị loại TS/Text)

**Kết luận nhanh:** shard kỹ thuật (79 dataset) gần như toàn là dữ liệu cảm biến/vật lý, phân loại ảnh/chữ, malware, mô phỏng. Không có treatment là "hành động do con người quyết định". Ứng viên đáng giá nhất lại nằm trong nhóm bị loại vì gắn nhãn Sequential/Text: **498 Incident management** (gộp event log về mức từng incident) và **1025 Turkish Crowdfunding**. Trong shard chính chỉ còn **165 Concrete** đáng giữ, ở mức dự phòng. Chỉ 498 vượt rõ mốc Student Performance trên cả 5 tiêu chí; các ứng viên còn lại ngang mốc hoặc yếu hơn ở n.

Quy ước: **[ĐẾM]** là số tự đếm từ dữ liệu đã tải về (python3 -I, csv thuần, file tại `.../scratchpad/dl_eng/<id>/`). **[META]** là số lấy từ trang hoặc API UCI. Script nằm ở `.../scratchpad/scripts_eng/`.

---

## 1. UCI 498 — Incident management process enriched event log (ServiceNow)
- URL: https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log · DOI 10.24432/C57S4H · CC BY 4.0 (mặc định của UCI) · năm 2018 (Amaral, Fantinato, Peres)
- Cấu trúc: 141.712 event × 36 cột. Sau khi gộp về mức incident còn **24.918 incident [ĐẾM]**, tức một dòng cho mỗi ticket. Ticket mở trong khoảng 29/02/2016–16/02/2017 [ĐẾM].
- Cách gộp: lấy bản ghi đầu (sys_mod_count nhỏ nhất) làm baseline lúc mở ticket, bản ghi cuối (state=Closed, có ở 100% ticket [ĐẾM]) làm outcome.
- **Câu hỏi:** Với các incident IT báo vào service desk của một công ty (2016–17), nếu gắn hoặc dùng tài liệu knowledge base cho ticket (T=knowledge: true, control=false) thì xác suất ticket đạt SLA (Y=made_sla) thay đổi bao nhiêu? T được xác định lúc mở ticket, Y xác định lúc đóng.
- Số liệu [ĐẾM]:
  - Treated 3.555/24.918 = **14,3%**.
  - Y=true 63,4% (made_sla=false 36,6%).
  - Chênh lệch thô: P(Y|T=0)=0,698, P(Y|T=1)=0,252, tức −45 điểm %. Mức chênh lớn như vậy gợi ý confounding hoặc nhân quả ngược mạnh, nên đây là đề tài hay để điều chỉnh.
  - Overlap theo priority: tỉ lệ treated 13,7% (Moderate), 29% (Low), 18% (High), 23% (Critical).
  - Theo 10 category lớn: treated 3–16%. Theo assignment group: 0,6–25%. Theo người mở ticket: 3–36%.
  - T ổn định trong ticket: chỉ 208/24.918 ticket đổi giá trị knowledge, và 3.555 ticket đã có knowledge=true ngay ở event đầu. Điều này ủng hộ việc coi T là quyết định lúc tiếp nhận.

| Vai trò | Biến |
|---|---|
| T | knowledge (bản ghi đầu) |
| Y | made_sla (bản ghi cuối). Phương án khác: resolved trong ≤24h (48,4% [ĐẾM]) |
| Baseline C | priority (hoặc impact và urgency lúc mở), category (gộp top-8 + other), location (gộp top-k), assignment_group lúc mở (gộp; thiếu 8,7%), opened_by (gộp top-k: tác nhân ra quyết định), opened weekday/weekend, giờ mở (ca), u_symptom (thiếu 23%, gộp), contact_type (99% là Phone, có thể bỏ) |
| Post-treatment / mediator (loại) | reassignment_count (45,6% >0; treated 55% vs 44%), reopen_count, sys_mod_count, u_priority_confirmation (đổi trong quá trình xử lý ở 19.124 ticket), priority/impact/urgency ở bản ghi cuối, incident_state, assigned_to |
| Không dùng / leakage | resolved_by, resolved_at, closed_at, closed_code, sys_updated_*, problem_id/rfc/vendor/caused_by (thiếu >98%), cmdb_ci (thiếu 99,8%), caller_id (5.245 giá trị), notify (99,9% một giá trị) |

- **Rủi ro chính:**
  - Ngữ nghĩa của T chưa chắc chắn. UCI viết "KB document was used to resolve", nhưng trong ServiceNow cờ Knowledge cũng có thể nghĩa là "đề xuất tạo bài KB". Cần ghi rõ giả định.
  - Nhiễu ẩn: độ phức tạp thực của sự cố (không có text).
  - Positivity của Y theo priority: High/Critical gần như luôn trễ SLA (made_sla=true chỉ 0,5–1,9%) nhưng T vẫn đủ overlap.
  - Chiều của made_sla: tài liệu viết "exceeded the target SLA". Số liệu (Critical/High → true hiếm) cho thấy true = đạt SLA, nhưng cần kiểm tra lại.
  - Ticket cùng caller/agent không độc lập (cluster). Cần một bước tiền xử lý từ event log.
- **Điểm:**
  - Thực tiễn **4**: quản lý service desk đang thật sự quyết định đầu tư hoặc bắt buộc dùng KB (và trợ lý KB/AI) để giảm vi phạm SLA, rất thời sự năm 2026.
  - Can thiệp **4**: dùng hoặc gắn KB là hành động của agent hay chính sách, do được; chỉ bị trừ vì ngữ nghĩa cờ còn mơ hồ.
  - Confounding/timing **3**: có priority, category, group, agent, thời điểm mở đo trước T. Thiếu độ khó nội tại, và có nguy cơ nhân quả ngược.
  - Kỹ thuật **5**: n≈25k, treated 14%, Y 37/63, mọi biến đều phân loại hoặc gộp được, DAG 10–12 nút.
  - Mới **5**: ITSM/vận hành, hoàn toàn khác giáo dục và marketing.
  - **Tổng 21/25**
- Nguồn đã kiểm tra: API UCI id=498, dữ liệu thật (zip → incident_event_log.csv), WebSearch trang UCI và tài liệu ServiceNow incident process.

## 2. UCI 1025 — Turkish Crowdfunding Startups (bị loại do nhãn "Text"; thực ra là bảng, mỗi dòng một dự án)
- URL: https://archive.ics.uci.edu/dataset/1025/turkish+crowdfunding+startups · CC BY 4.0 (mặc định UCI, chưa xác nhận) · năm 2023 · bài gốc Kilinc & Aydin, *Electronic Commerce Research and Applications* 2023 (doi 10.1016/j.elerap.2023.101340)
- n = **1.628 [ĐẾM]**, 38 cột, file CSV phân cách bằng ";".
- **Câu hỏi:** Với các dự án gọi vốn cộng đồng thể loại "reward" trên các nền tảng Thổ Nhĩ Kỳ (2013–2020), nếu chủ dự án đăng video giới thiệu (T: tanitim_videosu=var, control=yok) thì xác suất dự án đạt mục tiêu gọi vốn (Y: basari_durumu=başarılı) thay đổi bao nhiêu? T được quyết định lúc khởi tạo trang dự án.
- Số liệu [ĐẾM]:
  - Treated 947/1.628 = **58%**.
  - Y=thành công 376 = **23%**.
  - Chênh lệch thô: P(Y|video)=0,361, P(Y|không video)=0,050.
  - Platform: fongogo 1.075, crowdfon 237, fonbulucu 235, nhóm khác nhỏ.
  - Năm phân bố 2013–2020. Hình thức all-or-nothing chiếm 1.509.

| Vai trò | Biến |
|---|---|
| T | tanitim_videosu |
| Y | basari_durumu |
| Baseline C | platform_adi (gộp 3+other), kategori (gộp ~6), yil, bolge, fon_sekli, hedef_miktari (mục tiêu, bin theo quantile), kinh nghiệm chủ dự án (kac_projenin_sahibi, kac_proje_destekledi), ekip_kisi_sayisi, gun_sayisi (thời lượng chiến dịch), web_sitesi, sosyal_medya |
| Post-treatment / leakage (loại) | destekci_sayisi, toplanan_tutar, destek_orani (suy ra trực tiếp Y), yorumlar, guncellemeler, sm_takipci (đo lúc thu thập) |
| Không dùng | id, proje_adi, proje_sahibi, proje_aciklamasi (text), video_uzunlugu (chỉ có khi có video), proje_sahibi_cinsiyet (không can thiệp được, chỉ dùng như C nếu cần) |

- **Rủi ro chính:**
  - n nhỏ (1.628): với DAG 10 nút và test 15% (≈245 dòng) thì nhiều ô CPD sẽ thưa.
  - Nhiễu ẩn "chất lượng hoặc độ nghiêm túc của dự án" tác động cả lên T lẫn Y. Đây là vấn đề kinh điển, có thể nêu như một hạn chế.
  - Positivity: nhánh không video chỉ có 34 dự án thành công.
  - Lĩnh vực Business có thể trùng với scout khác.
- **Điểm:**
  - Thực tiễn **4**: nhà sáng lập hoặc nền tảng quyết định có đầu tư làm video hay không (tốn chi phí); câu hỏi kinh điển trong nghiên cứu Kickstarter.
  - Can thiệp **5**: một hành động rời rạc, rõ ràng.
  - Confounding/timing **3**: có nhiều biến lúc khởi tạo, nhưng chất lượng dự án là nhiễu ẩn lớn.
  - Kỹ thuật **3**: n≈1,6k, đạt ngưỡng tối thiểu; Y 23%; nhiều biến cần gộp.
  - Mới **4**
  - **Tổng 19/25**
- Nguồn đã kiểm tra: API UCI id=1025 (variable_info đầy đủ), dữ liệu thật.

## 3. UCI 165 — Concrete Compressive Strength (dự phòng, lĩnh vực vật liệu)
- URL: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength · DOI 10.24432/C5PK67 · CC BY 4.0 · năm 1998 (Yeh, *Cement and Concrete Research* 28(12))
- n = **1.030 [ĐẾM]**, nhưng chỉ có **427 cấp phối khác nhau [ĐẾM]** vì cùng một cấp phối được đo ở nhiều tuổi.
- **Câu hỏi:** Với các mẻ bê tông thí nghiệm, nếu dùng phụ gia siêu dẻo (T: Superplasticizer>0, control=0) thì xác suất mẫu đạt cường độ ≥35 MPa (Y) thay đổi bao nhiêu? T được chọn lúc thiết kế cấp phối, Y đo ở tuổi Age.
  - Phương án thay thế: T = dùng tro bay (Fly Ash>0), câu hỏi giảm CO2 rất thời sự. Nhưng tro bay thay thế xi măng nên Cement trở thành mediator, khó hơn.
- Số liệu [ĐẾM]:
  - Treated 651 = **63%**.
  - Y(≥35 MPa) = **49%**.
  - Chênh lệch thô: 0,57 vs 0,36.
  - Overlap tốt trong mọi tầng 3×3 (tầng xi măng × tuổi <28/=28/>28): n0 từ 28 đến 58, n1 từ 36 đến 150.
  - Water trung bình 173 (có siêu dẻo) vs 196 (không có).
  - Fly ash >0: 464. Slag >0: 559. Age=28 ngày: 425.

| Vai trò | Biến |
|---|---|
| T | Superplasticizer>0 |
| Y | strength ≥35 MPa (hoặc ≥ median) |
| Baseline C | Cement, Blast Furnace Slag, Fly Ash, Coarse Aggregate, Fine Aggregate (bin tertile), Age (3–4 mức; là thời điểm đo chứ không phải confounder thật nhưng cần điều kiện hoá) |
| Mediator (loại) | Water (siêu dẻo cho phép giảm nước; nếu điều chỉnh thì chặn mất đường T→W/C→Y) |
| Không dùng | — |

- **Rủi ro chính:**
  - n nhỏ và dữ liệu cụm (427 cấp phối, nhiều tuổi): split phải theo cấp phối.
  - Dữ liệu tổng hợp từ nhiều nghiên cứu phòng thí nghiệm, nên có nhiễu ẩn "nguồn hoặc lab", loại xi măng, phụ gia.
  - Chỉ có 9 biến, DAG khoảng 8 nút là sát ngưỡng dưới.
  - Mọi thành phần được chọn đồng thời nên thứ tự thời gian dựa vào logic thiết kế chứ không phải thời gian thật.
- **Điểm:**
  - Thực tiễn **3**: kỹ sư vật liệu thật sự chọn phụ gia, nhưng câu hỏi đã được hoá học giải thích khá rõ.
  - Can thiệp **5**
  - Confounding/timing **2**
  - Kỹ thuật **2**
  - Mới **4**
  - **Tổng 16/25**
- Nguồn đã kiểm tra: API UCI id=165, dữ liệu thật (data.csv).

## 4. UCI 597 — Productivity Prediction of Garment Employees (dự phòng yếu)
- URL: https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees · CC BY 4.0 · năm 2020
- n = **1.197 [ĐẾM]**, là panel 12 team × 59 ngày.
- **Câu hỏi:** Với các team chuyền may, nếu trả thưởng khuyến khích trong ngày (T: incentive>0) thì xác suất đạt năng suất mục tiêu (Y: actual ≥ targeted) thay đổi bao nhiêu?
- Số liệu [ĐẾM]:
  - Treated 593 (50%), Y = 73%.
  - **Positivity vỡ:** ở bộ phận finishing chỉ 10/506 dòng có incentive. Phải giới hạn ở sewing, khi đó n=691 (treated 583 = 84%).
  - Trong sewing: P(Y|T=1)=0,92 vs P(Y|T=0)=0,36.
- Vai trò biến:
  - C: team, day, quarter, smv, no_of_workers, targeted_productivity, no_of_style_change.
  - Mediator: over_time, idle_time, idle_men.
  - Có thể leakage: wip (đo trong ngày).
- **Rủi ro chính:** n quá nhỏ sau khi lọc, dữ liệu là panel/chuỗi thời gian theo team, incentive có thể được trả *sau khi* đạt năng suất (nhân quả ngược).
- **Điểm:** Thực tiễn 4, Can thiệp 4, Confounding 2, Kỹ thuật 1, Mới 4 → **15/25**. Không khuyến nghị.
- Nguồn đã kiểm tra: API UCI id=597, dữ liệu thật.

---

## Xếp hạng
1. **498 Incident management — knowledge → made_sla (21/25):** vượt Student Performance (320) và Bank Marketing (222) ở n, tính thực tiễn và độ mới. Cần thêm một bước gộp event log về mức incident (khoảng 30 dòng code).
2. **1025 Turkish Crowdfunding — video → thành công (19/25):** câu hỏi đẹp, can thiệp rõ, nhưng n≈1,6k và nhiễu ẩn "chất lượng dự án".
3. **165 Concrete — siêu dẻo → ≥35 MPa (16/25):** can thiệp vật lý rõ, overlap tốt, nhưng nhỏ, phân cụm và ít biến.
4. 597 Garment (15/25): chỉ dùng làm phương án tham khảo.

## Đáng chú ý nhưng đã loại
- **601 AI4I Predictive Maintenance:** dữ liệu synthetic. "Type" L/M/H là biến thể sản phẩm chứ không phải quyết định bảo trì; các biến khác là cảm biến; machine failure chỉ 3,4% [ĐẾM] (<5%). **1186 Paddy:** 451 dòng trùng lặp [ĐẾM]; mọi lượng phân bón/giống là hàm tất định của Hectares (6 mức); thời tiết trùng 1–1 với Agriblock; yield/ha chỉ dao động 5.410–6.469. Gần như semi-synthetic, và T=Nursery wet/dry không có hiệu ứng.
- **Cảm biến/thiết bị không có can thiệp hợp lệ:** 551 Gas Turbine, 851 Steel Energy (chuỗi 15 phút), 421 APS Scania, 791 MetroPT-3, 963 UR3 CobotOps, 471 Grid Stability (mô phỏng). **Quá nhỏ:** 32 Cylinder Bands (n≈512; nhưng T=ink type/solvent → banding là ý tưởng hay), 3 Annealing (798).
- **Không có T hành động:** 1031 MathE (8 cột, T khả dĩ duy nhất là "Question Level" do giáo viên gán theo câu hỏi, không có covariate người học), 967/327 Phishing, 722 Android permissions, 542 Firewall (action là outcome). **Tuần tự thật hoặc không có T:** 553 Clickstream, 382 Cargo 2000, 498 ở dạng event log thô (đã vớt bằng cách gộp).

Nguồn: [UCI 498](https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log), [UCI 1025](https://archive.ics.uci.edu/dataset/1025/turkish+crowdfunding+startups), [UCI 165](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), [UCI 597](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees), [ServiceNow incident process](https://servicenow.com/docs/bundle/zurich-it-service-management/page/product/incident-management/concept/incident-management-process.html)
