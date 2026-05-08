# ViSER-Vietnamese-Speech-Emotion-Recognition

---
> Khúc này không đưa vào README, chỉ tóm tắt cho anh em (Không cho Thiên) review thoii.
# Tổng quan kiến trúc ViSER Pipeline

Dưới đây là bức tranh toàn cảnh về hệ thống nhận dạng cảm xúc tiếng Việt (ViSER) mà chúng ta đã xây dựng. Hệ thống được chia làm 3 luồng chính: **Trích xuất đặc trưng (Feature Extraction)**, **Machine Learning**, và **Deep Learning**.

## 1. Feature Extraction (Trích xuất đặc trưng)
Hệ thống hỗ trợ 2 nguồn trích xuất mạnh mẽ, phục vụ cho 2 trường phái mô hình khác nhau.

### A. Librosa (Truyền thống + Âm học)
*   **1D Features (132 dims)**: Dùng cho Machine Learning. Rút trích các đặc trưng âm học cơ bản (MFCC, Delta, Pitch, ZCR, Spectral Centroid, Energy...). Không sử dụng zero-padding để tránh nhiễu do khoảng lặng giả tạo. Lưu vào file `_1d.csv`.
*   **2D Features (130 dims/frame)**: Dùng cho Deep Learning. Audio được ép cứng (pad/truncate) về độ dài 5 giây để tạo ra ma trận `(Thời_gian, 130)`. Mỗi file audio được lưu thành 1 file `.npy` độc lập để tiết kiệm RAM.

### B. Wav2Vec2 (Pre-trained Transformer)
*   **Mô hình**: Sử dụng mô hình `nguyenvulebinh/wav2vec2-base-vietnamese-250h` được train sẵn cho tiếng Việt.
*   **1D Features (768 dims)**: Lấy trung bình (mean pooling) toàn bộ các token đầu ra. Lưu vào `_1d.csv`.
*   **2D Features (768 dims/frame)**: Giữ nguyên chuỗi token đầu ra `(Thời_gian, 768)`. Lưu vào từng file `.npy` độc lập.

---

## 2. Pipeline Dữ liệu (Data Loaders)

Điểm nhấn quan trọng là **không nhét toàn bộ dữ liệu vào RAM** đối với ảnh 2D, và **tránh Data Leakage** nhờ chia tập chuẩn.

### Cho Machine Learning (`utils/data_loader.py`)
*   Đọc từ file CSV.
*   Tự động tách nhãn (`emotion_id`) và xử lý Metadata (Gender, Accent) bằng kỹ thuật **One-Hot Encoding**.
*   **Chia tập (80:20)**: Tách dữ liệu huấn luyện thành Train và Validation bằng cơ chế `stratify` để đảm bảo tỷ lệ các lớp cảm xúc cân bằng. Trả về: `X_train, y_train, X_val, y_val, X_test, y_test`. Tập Test hoàn toàn không được đụng tới trong quá trình tìm tham số.

### Cho Deep Learning (`utils/dl_data_loader.py`)
*   Dùng **PyTorch Dataset** (`SERDataset`).
*   Sử dụng cơ chế **Lazy Loading**: Khi nào batch cần training, model mới load file `.npy` tương ứng từ ổ cứng lên RAM. Chống tràn RAM cực kỳ hiệu quả với 12000 file 2D.
*   Metadata được Encode và trả về độc lập dưới dạng Tensor: `(features, meta, label)`.
*   Sử dụng `Subset` của Pytorch để chia tập Train/Val (80:20).

---

## 3. Các Mô Hình (Models)

### Cụm Machine Learning (`src/models/` và `train_ml.py`)
1. **Linear Regression**:
2. **KNN**: Sử dụng `StandardScaler` và tự động tìm `k` tốt nhất qua Validation set.
3. **SVM**: Dùng Kernel `RBF` xử lý đường biên phi tuyến tính, kèm GridSearch tìm cặp `(C, gamma)` chuẩn xác nhất.
4. **Random Forest**: Mô hình dạng cây, không cần scale dữ liệu. GridSearch tìm số lượng cây (`n_estimators`) và độ sâu (`max_depth`).

### Cụm Deep Learning
1.  **1D CNN (`CNN.py` & `train_1dcnn.py`)**
    *   3 Khối Conv1D -> BatchNorm -> ReLU -> MaxPool.
    *   Sử dụng **AdaptiveMaxPool1D** cuối cùng để nén toàn bộ chuỗi thời gian về 1 giá trị, miễn nhiễm với sự sai khác về độ dài đoạn âm thanh.
    *   Nối (Concat) nhánh Metadata trước khi qua Dense Classifier.
2.  **Bi-LSTM (`Bi_LSTM.py` & `train_bilstm.py`)**
    *   2 lớp Bi-directional LSTM để phân tích ngữ cảnh tiến và lùi.
    *   Dùng **Global Average Pooling** (`torch.mean(dim=1)`) để tạo embedding. Cách này ổn định và tốt hơn hẳn so với việc chỉ lấy hidden state cuối cùng (vốn hay bị nhiễu nếu audio bị zero-padding quá nhiều ở khúc cuối).
    *   Nối Metadata trước Classifier.

> Cả 2 mô hình DL đều tích hợp bộ tối ưu **AdamW**, giảm learning rate tự động (`ReduceLROnPlateau`), và cơ chế **Early Stopping** (ngừng nếu sau 10 epoch Val_Acc không tăng) để tiết kiệm thời gian train.

---

## 4. Đánh giá (Evaluation & Plotting)

Mọi file training khi chạy xong đều đi qua module `src/evaluation/`.
*   **`evaluate.py`**: In ra Accuracy, F1-Score, Recall, Precision cho từng cảm xúc (Happy, Sad, Angry, Neutral).
*   **`plotter.py`**:
    *   Tự động vẽ **Confusion Matrix** (lưu dạng ảnh heatmap cực kỳ dễ nhìn).
    *   Vẽ biểu đồ phân bố Class Distribution.
    *   Vẽ biểu đồ lịch sử Loss và Accuracy từng Epoch cho mạng DL.
    *   Tất cả ảnh sẽ tự lưu vào thư mục `results/` mà không làm gián đoạn terminal.