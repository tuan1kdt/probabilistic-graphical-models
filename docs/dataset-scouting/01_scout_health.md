# Báo cáo scout: Health and Medicine + Biology (47 dataset)

Ký hiệu: **[KC]** là số đã kiểm chứng bằng cách đếm trên data.csv tải từ `https://archive.ics.uci.edu/static/public/<id>/data.csv` (python3 -I, thư viện csv). **[MD]** là số chỉ lấy từ metadata/API UCI, chưa đếm lại. License: trường `license` trong API trả về rỗng; UCI mặc định CC BY 4.0, nhưng tôi chưa xác minh từng trang.

Mốc so sánh: Student Performance (schoolsup→pass) và Bank Marketing (contact→y).

---

## 1. Diabetes 130-US Hospitals 1999–2008 (UCI ID 296)
- URL: https://archive.ics.uci.edu/dataset/296 · n = 101.766 lượt nhập viện [KC], 50 cột [KC], năm 2014. Sau khi giữ lượt đầu mỗi bệnh nhân và loại ca tử vong/hospice còn 69.973 [KC]. Bài gốc: Strack et al. 2014, *BioMed Res Int* 781670, "Impact of HbA1c Measurement on Hospital Readmission Rates". Bài này vốn đã là một phân tích quan sát theo kiểu nhân quả (exposure = đo HbA1c).
- **Câu hỏi nghiên cứu:** với bệnh nhân đái tháo đường nhập viện (lượt đầu, xuất viện còn sống), nếu bác sĩ **điều chỉnh thuốc tiểu đường trong lúc nằm viện** (`change`=Ch, so với No) thì xác suất **tái nhập viện trong 30 ngày** (`readmitted`=='<30') thay đổi bao nhiêu? Thời điểm can thiệp là trong đợt nằm viện, trước xuất viện.
  - Biến thể B (theo bài gốc): T = **có chỉ định xét nghiệm HbA1c** (`A1Cresult`≠None). Đây là quyết định chỉ định của bác sĩ, không phải giá trị đo.
- Cân bằng [KC]: Y = 8,97%. T(change) = 31.491/69.973 (45%); Y theo T: 9,43% so với 8,59%. T(A1C) = 12.845 (18,4%); Y: 8,39% so với 9,10%. Overlap tốt: tỉ lệ change là 35–52% ở mọi tầng age/admission_type/admission_source; tỉ lệ A1C là 6–50% (thấp nhất ở admission_source=2, 5,8%).

| Vai trò | Biến |
|---|---|
| T | `change` (biến thể B: chỉ báo A1Cresult≠None) |
| Y | readmitted=='<30' |
| Baseline C | age (gom 3–4 bin), gender, race, admission_type_id (cấp cứu/khẩn/chương trình), admission_source_id (ER/chuyển viện/khác), number_inpatient, number_emergency, number_outpatient của năm trước (bin 0/1/2+), nhóm ICD của diag_1 (tuần hoàn/hô hấp/tiểu đường/khác), payer_code (bảo hiểm). Với biến thể T=change: A1Cresult và max_glu_serum được xếp là pre-treatment (kết quả xét nghiệm dẫn tới quyết định đổi thuốc) |
| Post-T / mediator, loại | discharge_disposition_id (sau T, trước Y, là mediator), time_in_hospital, num_medications, num_lab_procedures, num_procedures (đồng thời với T hoặc bị T ảnh hưởng), từng cột thuốc như insulin, metformin Up/Down (chính là thành phần của T) |
| Không dùng | encounter_id, patient_nbr, weight (~97% missing [MD]), medical_specialty (49% NaN [KC]), diag_2, diag_3, examide, citoglipton (hằng số) |

- **Rủi ro:** nhiễu ẩn do mức độ nặng của bệnh (không có sinh hiệu, mức đường huyết lúc vào viện chỉ được đo một phần); timing trong đợt nằm viện là gần đúng (dữ liệu không có dấu thời gian); ATE kỳ vọng nhỏ (chênh lệch thô khoảng 0,8 điểm %), nên có thể không phân biệt được với 0 trên một test cohort khoảng 10k; phải bỏ trùng bệnh nhân để có i.i.d. Ưu điểm là n rất lớn nên CPD không bị rỗng.
- **Điểm:**
  - Thực tiễn 5/5: tái nhập viện 30 ngày bị phạt tài chính (HRRP) và quản lý đường huyết nội viện là quyết định lâm sàng hằng ngày.
  - Can thiệp 4/5: đổi thuốc hay chỉ định HbA1c đều do bác sĩ chọn. Trừ 1 điểm vì `change` gộp nhiều loại thay đổi khác nhau (tăng, giảm, đổi thuốc).
  - Confounding/timing 3/5: có tiền sử sử dụng dịch vụ và chẩn đoán làm C hợp lý, nhưng thiếu biến mức độ nặng và không có timestamp.
  - Kỹ thuật 5/5: khoảng 70k dòng, biến vốn đã phân loại, overlap tốt, Y = 9%.
  - Tính mới 5/5: bối cảnh lâm sàng/bệnh viện, hoàn toàn khác giáo dục hay marketing.
  - **Tổng 22/25**
- Nguồn đã kiểm tra: API UCI id=296, data.csv (đếm thật), tham chiếu bài Strack 2014 trong metadata.

## 2. CDC Diabetes Health Indicators, BRFSS 2015 (UCI ID 891)
- URL: https://archive.ics.uci.edu/dataset/891 · n = 253.680 [KC], 22 biến [KC], dữ liệu khảo sát BRFSS (cross-sectional, người lớn Mỹ). Metadata ghi năm 2017, mô tả dẫn link BRFSS 2014 [MD].
- **Câu hỏi nghiên cứu:** với người trưởng thành Mỹ, nếu một người **từng hút thuốc** (`Smoker`=1, tức ≥100 điếu trong đời, so với 0) thì xác suất **từng bị bệnh mạch vành/nhồi máu** (`HeartDiseaseorAttack`) thay đổi bao nhiêu?
  - Biến thể: T = `PhysActivity` (vận động trong 30 ngày qua), Y = `Diabetes_binary`. Biến thể này yếu hơn vì có nhân quả ngược.
- Cân bằng [KC]: T = 112.423 (44,3%); Y = 9,4%; Y theo T: 13,17% so với 6,44%. Tỉ lệ hút thuốc theo tầng là 22–54% (Age), 34–62% (Education), 36–56% (Income), nên overlap rất tốt. Với biến thể PhysActivity→Diabetes: 11,6% so với 21,1%.

| Vai trò | Biến |
|---|---|
| T | Smoker |
| Y | HeartDiseaseorAttack (hoặc Stroke, 5,4% so với 3,0% [KC]) |
| Baseline C | Age (13 nhóm, gom còn 4–5), Sex, Education, Income, AnyHealthcare. Có thể thêm HvyAlcoholConsump, Fruits, Veggies (hành vi đồng thời, nên cần lập luận) |
| Post-T / mediator, loại | HighBP, HighChol, BMI, Diabetes_binary (mediator trên đường hút thuốc → tim mạch), GenHlth, PhysHlth, DiffWalk, MentHlth (hệ quả của bệnh), Stroke |
| Không dùng | ID, CholCheck, NoDocbcCost (có thể đưa vào C) |

- **Rủi ro:** dữ liệu cross-sectional nên "từng bị" bệnh tim và "từng hút" không có thứ tự thời gian chắc chắn (dù hút ≥100 điếu thường xảy ra trước). Đây là bản Kaggle đã làm sạch, không có trọng số khảo sát. Nhiễu ẩn gồm di truyền và môi trường. Income có thể chịu ảnh hưởng của bệnh. Can thiệp "ever-smoker" khó thao tác; câu hỏi tự nhiên hơn là "không bao giờ bắt đầu hút".
- **Điểm:**
  - Thực tiễn 4/5: chính sách phòng chống thuốc lá vẫn là ưu tiên y tế công cộng.
  - Can thiệp 3/5: hút thuốc là hành vi cá nhân, nhưng biến ever-smoker là phơi nhiễm trọn đời và do() mơ hồ.
  - Confounding/timing 3/5: C nhân khẩu học tốt, nhưng thiết kế cross-sectional.
  - Kỹ thuật 5/5: n rất lớn, toàn biến nhị phân/thứ bậc, không missing.
  - Tính mới 4/5: dịch tễ học khác hẳn hai mốc, nhưng là dataset Kaggle phổ biến.
  - **Tổng 19/25**
- Nguồn đã kiểm tra: API UCI id=891, data.csv (đếm thật).

## 3. AIDS Clinical Trials Group Study 175 (UCI ID 890)
- URL: https://archive.ics.uci.edu/dataset/890 · n = 2.139 [KC], 25 cột [KC], năm 1996. Bài gốc: Hammer et al. 1996, NEJM, so sánh đơn trị nucleoside với phối hợp ở người HIV có CD4 200–500. Đây là **RCT**.
- **Câu hỏi nghiên cứu:** với người lớn nhiễm HIV có CD4 200–500, nếu dùng **phác đồ phối hợp/thay thế** (`treat`=1: ZDV+ddI, ZDV+ddC hoặc ddI đơn trị) thay vì **ZDV đơn trị** (`treat`=0), xác suất **biến cố (CD4 giảm >50%, AIDS hoặc tử vong)** (`cid`) thay đổi bao nhiêu? Thời điểm can thiệp là lúc randomization (baseline).
- Cân bằng [KC]: T = 1.607/2.139 (75%); Y = 24,4% (521). Y theo T: 34,0% (ZDV) so với 21,2%, nên ATE thô khoảng −12,8 điểm %. Theo từng arm, cid là trt0 34,0%, trt1 19,7%, trt2 20,8%, trt3 22,8%. Overlap được đảm bảo nhờ randomization.

| Vai trò | Biến |
|---|---|
| T | treat (hoặc trt=1 so với trt=0 nếu muốn so sánh cặp, nhưng khi đó n ≈ 1.054) |
| Y | cid |
| Baseline C | age (bin), wtkg (bin), karnof, cd40, cd80 (CD4/CD8 baseline, bin), gender, race, homo, drugs, hemo, symptom, str2/strat (tiền sử antiretroviral), preanti (bin), z30 |
| Post-T, loại | offtrt (ngừng thuốc trước 96±5 tuần), cd420, cd820 (CD4/CD8 ở tuần 20, là mediator), time (thời gian theo dõi, gắn với Y) |
| Không dùng | pidnum, zprior (hằng số = 1 [KC]), oprior (chỉ 47 ca = 1 [KC]) |

- **Rủi ro:** vì là RCT nên không có confounding thật. Đây vừa là điểm mạnh (có "ground truth" để kiểm tra DoWhy, NOTEARS và refuter: ATE điều chỉnh phải ≈ ATE thô) vừa là điểm yếu với yêu cầu "lập luận confounder". Có thể tạo một kịch bản quan sát bằng cách subsample có chọn lọc theo karnof/cd40 rồi kiểm tra phương pháp có khôi phục được ATE RCT hay không. n = 2.139 nằm ở ngưỡng, nên test cohort 15% chỉ khoảng 320 dòng và CPD với 10+ nút sẽ thưa. Câu hỏi lâm sàng đã lỗi thời vì ART hiện đại khác hẳn. Y có censoring (time).
- **Điểm:**
  - Thực tiễn 3/5: chọn phác đồ là quyết định thật, nhưng là thuốc thập niên 1990; giá trị chủ yếu nằm ở benchmark phương pháp.
  - Can thiệp 5/5: phác đồ được gán ngẫu nhiên, do() rõ ràng tuyệt đối.
  - Confounding/timing 4/5: timing rất sạch (baseline → randomize → tuần 20 → cid); trừ 1 điểm vì không có confounding để điều chỉnh.
  - Kỹ thuật 3/5: n ≈ 2.1k, Y 24%, T 75/25, không missing.
  - Tính mới 4/5: dữ liệu RCT cho phép so sánh kết quả ước lượng với ground truth, điều mà hai mốc không có.
  - **Tổng 19/25**
- Nguồn đã kiểm tra: API UCI id=890 (paper intro), data.csv (đếm thật).

## 4. Myocardial infarction complications (UCI ID 579)
- URL: https://archive.ics.uci.edu/dataset/579 · n = 1.700 [KC], 124 cột [KC], năm 2020 (dữ liệu Krasnoyarsk 1992–1995 [MD]). Bài intro: Golovenkin et al., "Trajectories, bifurcations and pseudo-time…".
- **Câu hỏi nghiên cứu:** với bệnh nhân nhồi máu cơ tim cấp nhập viện, nếu **dùng beta-blocker tại ICU** (`B_BLOK_S_n`=1 so với 0) thì xác suất **tử vong** (`LET_IS`≠0) thay đổi bao nhiêu? Thời điểm là những ngày đầu ICU, sau dữ liệu lúc nhập viện.
  - Biến thể: T = aspirin `ASP_S_n`, hoặc heparin `GEPAR_S_n`.
- Cân bằng [KC]: Y = 15,9%. Beta-blocker: T = 215 (12,7%), Y 7,4% so với 17,2%. Aspirin: T = 1.252 (74%), Y 11,7% so với 28,3%. Heparin: 1.203 (71%), Y 15,0% so với 18,5%. Missing của T khoảng 1%.

| Vai trò | Biến |
|---|---|
| T | B_BLOK_S_n (hoặc ASP_S_n) |
| Y | LET_IS != 0 |
| Baseline C (lúc nhập viện) | AGE, SEX, INF_ANAM (số lần MI trước), GB (tăng huyết áp), ZSN_A (suy tim mạn), endocr_01 (đái tháo đường), S_AD_ORIT (huyết áp tâm thu lúc vào), vị trí nhồi máu (ant_im/inf_im), TIME_B_S (thời gian từ khởi phát đến nhập viện) |
| Post-T, loại | R_AB_2_n/3_n, NA_R_2_n/3_n, NOT_NA_2_n/3_n (ngày 2–3), toàn bộ các biến chứng FIBR_PREDS…REC_IM, P_IM_STEN (là mediator/outcome khác) |
| Không dùng | ID, IBS_NASL (95,8% missing [MD]), KFK_BLOOD và các xét nghiệm missing cao |

- **Rủi ro:** timing của "dùng thuốc tại ICU" không có mốc giờ. Bệnh nhân chết sớm không kịp được cho thuốc, gây **immortal-time bias**: chênh lệch thô của aspirin 28% so với 12% gần như chắc chắn bị phóng đại. Có confounding by indication (beta-blocker chống chỉ định khi sốc hoặc suy tim, đúng nhóm có nguy cơ tử vong cao), nên positivity yếu trong tầng sốc tim. n = 1.700 nhỏ hơn mức ưu tiên 2.000. Missing rải rác nhiều cột.
- **Điểm:**
  - Thực tiễn 4/5: beta-blocker sớm trong MI vẫn được tranh luận (COMMIT, REDUCE-AMI 2024).
  - Can thiệp 4/5: kê thuốc là quyết định lâm sàng rõ ràng.
  - Confounding/timing 2/5: có biến lúc nhập viện, nhưng immortal-time và chỉ định theo mức độ nặng.
  - Kỹ thuật 2/5: n = 1.700, 124 cột cần chọn lọc, có missing, T = 12,7%.
  - Tính mới 5/5: dược lý lâm sàng, rất khác hai mốc.
  - **Tổng 17/25**
- Nguồn đã kiểm tra: API UCI id=579 (variable_info đầy đủ), data.csv (đếm thật).

---

## Xếp hạng
1. **296 Diabetes 130-US Hospitals** (22/25): T = đổi thuốc tiểu đường hoặc chỉ định HbA1c, Y = tái nhập viện 30 ngày. Tốt hơn Student Performance trên cả 5 tiêu chí, n lớn, overlap tốt và có bài gốc mang tính nhân quả.
2. **891 CDC BRFSS** (19/25): T = Smoker, Y = HeartDiseaseorAttack. Kỹ thuật lý tưởng nhưng là cross-sectional; nên dùng làm phương án dự phòng an toàn.
3. **890 ACTG 175** (19/25): là RCT nên có ground truth để kiểm chứng pipeline. Nhỏ, thiếu confounding thật; phù hợp làm dataset phụ để validate.
4. **579 MI complications** (17/25): câu hỏi lâm sàng hấp dẫn nhưng immortal-time bias và n nhỏ.

## Đáng chú ý nhưng đã loại
- **880 SUPPORT2** (n = 9.105): T = lệnh DNR là quyết định thật, nhưng DNR thường được ra ngay trước khi chết. Từ dữ liệu đếm được: hospdead 61,8% ở nhóm "dnr after sadm" so với 7,5% ở nhóm không DNR; median dnrday = 4, chỉ 44% DNR trong ≤3 ngày. Physiology đo ở ngày 3 đồng thời với T, nên có nhân quả ngược và chỉ định quá nặng. Can thiệp thật của thử nghiệm SUPPORT (nurse intervention) không có trong file.
- **544 Obesity levels**: các hành vi FAVC, FAF, SMOKE rất hợp làm T, nhưng 77% dòng là synthetic (SMOTE), n gốc chỉ khoảng 485 [MD]. **30 Contraceptive Method Choice** (n = 1.473): T = media_exposure khó thao tác, outcome là lựa chọn chứ không phải kết cục sức khỏe.
- **383 Cervical cancer, 936 NPHA, 887 NHANES subset, 503 HCV Egypt**: n < 1.000, hoặc quá ít biến (887 chỉ có 7 biến), hoặc "treatment" chỉ là kết quả xét nghiệm hay triệu chứng. 503 không có biến phác đồ (mọi bệnh nhân đều được điều trị). Các dataset Biology còn lại (Covertype, Mushroom, Dry Bean, gene expression…) không có biến hành động nào.
