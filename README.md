# Computer Project MAI391: Mathematics for Machine Learning
## Dự đoán giá Card đồ họa (VGA) bằng Hồi quy Tuyến tính (Linear Regression)

Dự án nghiên cứu và xây dựng mô hình Hồi quy Tuyến tính (Linear Regression) để dự đoán giá bán của các dòng card đồ họa (VGA), kết hợp nền tảng toán học máy học:
- **Ước lượng Hợp lý Cực đại (MLE)** và sai số chuẩn Gauss.
- **Phép chiếu trực giao (Orthogonal Projection - OLS)** và phương trình nghiệm chuẩn (Normal Equation).
- **Phân rã giá trị suy biến (SVD)** giải hệ phương trình suy biến và kiểm soát tính ổn định số học.
- **Giải tích Vector & Tối ưu hóa Gradient Descent (GD / Batch GD)**.
- **Ước lượng Hậu nghiệm Cực đại (MAP) & Hồi quy Ridge (L2 Regularization)** giảm thiểu đa cộng tuyến.

---

### Cấu trúc Thư mục

- `run.py`: Script thực thi toàn bộ pipeline:
  1. Trích xuất đặc trưng & điền khuyết chéo từ dữ liệu phần cứng.
  2. Thiết kế ma trận đặc trưng $X$, phân tích tương quan và kiểm định đa cộng tuyến (VIF).
  3. Cài đặt các thuật toán Hồi quy từ đầu bằng NumPy (OLS qua Inversion, SVD-based, Gradient Descent, Ridge OLS, Ridge GD) và đối chiếu với Scikit-Learn.
  4. Đánh giá thuật toán qua K-Fold Cross-Validation, đo lường thời gian huấn luyện và độ chính xác ($R^2$, RMSE, MAE).
  5. Trực quan hóa kết quả (sinh ra 6 slide biểu đồ).
- `Master_VGA_All_Columns_Aligned.csv`: Bộ dữ liệu thông số và giá bán VGA tổng hợp.
- `requirements.txt` / `Install.txt`: Các thư viện phụ thuộc của dự án.
- Các biểu đồ trực quan hóa kết quả:
  - `Slide1_Price_vs_LogPrice_Distribution.png`
  - `Slide2_EDA_Correlation_and_VIF.png`
  - `Slide3_Gradient_Descent_Convergence.png`
  - `Slide4_Ridge_Trace_and_CV_Tuning.png`
  - `Slide5_Actual_vs_Predicted_and_Residuals.png`
  - `Slide6_Feature_Weights_Interpretation.png`

---

### Hướng dẫn Cài đặt & Chạy

1. **Cài đặt môi trường:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Chạy thực nghiệm:**
   ```bash
   python run.py
   ```
