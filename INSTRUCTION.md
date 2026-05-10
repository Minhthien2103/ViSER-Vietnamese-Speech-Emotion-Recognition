# HƯỚNG DẪN CHẠY DỰ ÁN ViSER
> Vietnamese Speech Emotion Recognition

---

## Bước 0: Cài đặt môi trường

### 0.1 Clone project
```bash
git clone https://github.com/<your-repo>/ViSER-Vietnamese-Speech-Emotion-Recognition.git
cd ViSER-Vietnamese-Speech-Emotion-Recognition
```

### 0.2 Tạo Conda environment
```bash
conda create -n cs114 python=3.10 -y
conda activate cs114
```

### 0.3 Cài đặt thư viện
```bash
pip install pandas numpy librosa scikit-learn torch torchaudio transformers tqdm matplotlib seaborn joblib
```

> **Lưu ý:** Nếu dùng GPU, cài PyTorch phiên bản CUDA tương ứng tại [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Bước 1: Tải và đặt dữ liệu

### 1.1 Tải dữ liệu từ Google Drive
Tải 2 thư mục `train/` và `test/` từ link Drive được chia sẻ.

### 1.2 Đặt vào đúng vị trí
Giải nén (nếu cần) và đặt vào thư mục `data/` của project sao cho cấu trúc như sau:

```
ViSER-Vietnamese-Speech-Emotion-Recognition/
└── data/
    ├── train/
    │   ├── Female/
    │   │   ├── mien_bac/
    │   │   │   ├── Angry/
    │   │   │   │   ├── Female_mien_bac_Angry_orig_0000.wav
    │   │   │   │   ├── aug_Female_mien_bac_Angry_gain_dn3db_0010_0010.wav
    │   │   │   │   └── ...
    │   │   │   ├── Happy/
    │   │   │   ├── Neutral/
    │   │   │   └── Sad/
    │   │   ├── mien_nam/
    │   │   │   └── (tương tự)
    │   │   └── mien_trung/
    │   │       └── (tương tự)
    │   └── Male/
    │       └── (cấu trúc giống Female)
    └── test/
        └── (cấu trúc giống train)
```

> **Quan trọng:** Tên các thư mục phải chính xác: `Female`, `Male`, `mien_bac`, `mien_nam`, `mien_trung`, `Angry`, `Happy`, `Neutral`, `Sad`.

---

## Bước 2: Sinh Metadata

Chạy script này **một lần duy nhất** để quét toàn bộ thư mục data và tạo ra file CSV metadata:

```bash
python src/features/build_metadata.py
```

**Kết quả:** Tạo ra 2 file trong `data/processed/`:
- `train_metadata.csv`
- `test_metadata.csv`

Bạn sẽ thấy thống kê số lượng file theo giới tính, vùng miền, cảm xúc in ra trên terminal.

---

## Bước 3: Trích xuất đặc trưng (Feature Extraction)

### 3.1 Trích xuất bằng Librosa (bắt buộc)
```bash
python src/features/extract_librosa.py
```
- Tạo ra features 1D (CSV) cho Machine Learning.
- Tạo ra features 2D (thư mục .npy) cho Deep Learning.
- **Thời gian ước tính:** ~10-15 phút cho 12000 files.

### 3.2 Trích xuất bằng Wav2Vec2 (tùy chọn, cần GPU)
```bash
python src/features/extract_wav2vec.py
```
- Dùng mô hình `wav2vec2-base-vietnamese-250h` pre-trained cho tiếng Việt.
- **Thời gian ước tính:** ~30-60 phút (GPU) hoặc vài giờ (CPU).

**Kết quả:** Các file features được lưu trong `data/processed/`:
```
data/processed/
├── train_librosa_1d.csv       # 1D features cho ML
├── test_librosa_1d.csv
├── train_librosa_2d/          # 2D features cho DL (từng file .npy)
├── test_librosa_2d/
├── train_wav2vec_1d.csv       # (nếu chạy bước 3.2)
├── test_wav2vec_1d.csv
├── train_wav2vec_2d/
└── test_wav2vec_2d/
```

---

## Bước 4: Huấn luyện mô hình

### 4.1 Machine Learning (KNN, SVM, Random Forest)
```bash
python src/train/train_ml.py
```

Mặc định sử dụng features từ Librosa. Muốn đổi sang Wav2Vec, mở file `train_ml.py` và sửa:
```python
FEATURE_SOURCE = "wav2vec"  # thay "librosa" thành "wav2vec"
```

Trong file, các model được comment sẵn. Bỏ comment dòng nào thì train model đó:
```python
train_knn(X_train, y_train, X_val, y_val, X_test, y_test, source = FEATURE_SOURCE)
# train_svm(...)  # bỏ comment để chạy SVM
# train_rf(...)   # bỏ comment để chạy Random Forest
```

### 4.2 Deep Learning — 1D CNN
```bash
python src/train/train_1dcnn.py
```

### 4.3 Deep Learning — Bi-LSTM
```bash
python src/train/train_bilstm.py
```

> Tương tự ML, muốn đổi nguồn feature thì sửa biến `FEATURE_SOURCE` trong mỗi file train.

---

## Bước 5: Xem kết quả

Sau khi train xong, kết quả được lưu tự động:

### Models đã train
```
models/saved/
├── knn_librosa.joblib
├── svm_librosa.joblib
├── rf_librosa.joblib
├── cnn1d_librosa.pth
└── bilstm_librosa.pth
```

### Biểu đồ đánh giá
```
results/
├── distribution_librosa.png       # Phân bố số lượng theo cảm xúc
├── cm_KNN (k=...)_librosa.png     # Confusion Matrix cho từng model
├── cm_SVM (...)_librosa.png
├── cm_CNN1D_librosa.png
├── cm_BiLSTM_librosa.png
├── history_CNN1D_librosa.png      # Đồ thị Loss/Accuracy theo Epoch
└── history_BiLSTM_librosa.png
```

---

## Tóm tắt thứ tự chạy

| Bước | Lệnh | Mục đích |
|------|-------|----------|
| 1 | Tải data từ Drive, đặt vào `data/` | Chuẩn bị dữ liệu |
| 2 | `python src/features/build_metadata.py` | Sinh file CSV metadata |
| 3 | `python src/features/extract_librosa.py` | Trích xuất đặc trưng |
| 4a | `python src/train/train_ml.py` | Train ML (KNN/SVM/RF) |
| 4b | `python src/train/train_1dcnn.py` | Train 1D CNN |
| 4c | `python src/train/train_bilstm.py` | Train Bi-LSTM |
| 5 | Mở thư mục `results/` | Xem biểu đồ kết quả |

---

## Cấu trúc Project

```
ViSER-Vietnamese-Speech-Emotion-Recognition/
├── data/
│   ├── train/                    # Dữ liệu huấn luyện (.wav)
│   ├── test/                     # Dữ liệu kiểm tra (.wav)
│   └── processed/                # Features đã trích xuất (tự sinh)
├── models/saved/                 # Models đã train (tự sinh)
├── results/                      # Biểu đồ đánh giá (tự sinh)
├── src/
│   ├── config.py                 # Cấu hình đường dẫn
│   ├── features/
│   │   ├── build_metadata.py     # Quét data → sinh CSV metadata
│   │   ├── extract_librosa.py    # Trích xuất Librosa features
│   │   └── extract_wav2vec.py    # Trích xuất Wav2Vec2 features
│   ├── models/
│   │   ├── KNN.py, SVM.py, RF.py # Định nghĩa model ML (OOP)
│   │   ├── CNN.py                # Định nghĩa 1D CNN
│   │   └── Bi_LSTM.py            # Định nghĩa Bi-LSTM
│   ├── train/
│   │   ├── train_ml.py           # Train tất cả ML models
│   │   ├── train_1dcnn.py        # Train 1D CNN
│   │   └── train_bilstm.py       # Train Bi-LSTM
│   ├── evaluation/
│   │   ├── evaluate.py           # In metrics (Accuracy, F1, ...)
│   │   └── plotter.py            # Vẽ biểu đồ tự động
│   └── utils/
│       ├── data_loader.py        # Load data cho ML
│       └── dl_data_loader.py     # Load data cho DL (PyTorch)
└── requirements.txt
```

---

## Gặp lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `FileNotFoundError: train_metadata.csv` | Chưa chạy bước 2 | Chạy `python src/features/build_metadata.py` |
| `ModuleNotFoundError: No module named 'librosa'` | Chưa cài thư viện | Chạy `pip install librosa` |
| `CUDA out of memory` | GPU không đủ VRAM | Giảm `BATCH_SIZE` trong file train hoặc dùng CPU |
| `0 files loaded` | Data chưa đặt đúng thư mục | Kiểm tra lại cấu trúc thư mục `data/` |
