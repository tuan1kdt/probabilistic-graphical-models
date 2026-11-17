# data/raw

Đặt file gốc của UCI ID 222 Bank Marketing vào đây, **không chỉnh sửa**:

- `bank-additional-full.csv` (41.188 dòng, 20 input + `y`, phân cách `;`)

Tải từ https://archive.ics.uci.edu/dataset/222/bank+marketing (zip `bank+marketing.zip` → `bank-additional/bank-additional-full.csv`),
hoặc để ô `load_bank_marketing()` trong `notebooks/01_bank_marketing_cbn.ipynb` tự tải bằng `ucimlrepo` nếu máy có mạng tới archive.ics.uci.edu.

Ghi lại ngày tải, checksum (ô setup in ra sha256) và license CC BY 4.0 vào mục 2 của notebook (slide 55).

## Provenance (tải 2026-11-02)
- Nguồn: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip → `bank-additional.zip` → `bank-additional/`
- License: CC BY 4.0. Trích dẫn: Moro, Cortez & Rita (2014), *Decision Support Systems* 62:22–31, DOI 10.1016/j.dss.2014.03.001
- `bank+marketing.zip` sha256 `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4`
- `bank-additional-full.csv` sha256 `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8` (41.188 dòng × 21 cột, không NA)
- `bank-additional-names.txt`: mô tả biến gốc của UCI

# data/raw/coupon — UCI ID 603 In-Vehicle Coupon Recommendation (tải 2026-11-06, 11:52 UTC)
- Nguồn: https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation
  (zip chính thức `https://archive.ics.uci.edu/static/public/603/in+vehicle+coupon+recommendation.zip`)
- DOI 10.24432/C5GS4P. License CC BY 4.0. Trích dẫn: Wang, Rudin, Doshi-Velez, Liu, Klampfl & MacNeille (2017),
  *A Bayesian Framework for Learning Rule Sets for Interpretable Classification*, JMLR 18(70):1–37.
- `in+vehicle+coupon+recommendation.zip` sha256 `22fc6bf1a03e28908c3a3ceb2fe0c7cad073ceb541f5fca60dd37e3b0af9fc46`
- `in-vehicle-coupon-recommendation.csv` sha256 `d47897367f5a12a31dece0f4e42d7bd9da76d7016b568da511b50335d08dcb43`
  (12.684 dòng × 26 cột, phân cách `,`; cột `passanger` viết sai chính tả trong file gốc, giữ nguyên ở raw và đổi thành `passenger` khi tiền xử lý)
- Không chỉnh sửa file trong thư mục này. Tiền xử lý: `scripts/prep_coupon.py` → `data/processed/`.
