# ViSER - Vietnamese Speech Emotion Recognition
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%2334D058.svg?style=for-the-badge&logo=huggingface&logoColor=white)

## 📖 Mô tả dự án (Project Description)
ViSER (Vietnamese Speech Emotion Recognition) là một hệ thống nhận dạng cảm xúc qua giọng nói được thiết kế và xây dựng dành riêng cho tiếng Việt. Dự án ứng dụng đa dạng các phương pháp tiếp cận từ Machine Learning truyền thống (SVM, KNN, Random Forest, LGBM, XGBoost) cho đến Deep Learning nâng cao (1D-CNN, Bi-LSTM) nhằm trích xuất và phân tích cảm xúc từ tín hiệu âm thanh. Hệ thống hỗ trợ hai pipeline trích xuất đặc trưng mạnh mẽ là **Librosa** (đặc trưng âm học) và **Wav2Vec2** (Pre-trained Transformer) để tối ưu hóa khả năng dự đoán 4 trạng thái cảm xúc chính: **Happy, Sad, Angry, Neutral**.

## 💾 Dữ liệu (Our Data)
Dữ liệu của dự án được thu thập và gán nhãn một cách tỉ mỉ từ nhiều nguồn video và audio tiếng Việt đa dạng. Tập dữ liệu bao gồm các đoạn âm thanh đã được tiền xử lý (cắt gọn, loại bỏ nhiễu) kèm theo bộ Metadata chứa thông tin quan trọng về giới tính (Gender) và vùng miền (Accent), giúp mô hình học được sự đa dạng và phong phú trong giọng nói của người Việt Nam.

🎥 **Video Demo quy trình xử lý và gán nhãn Data**: [Link Video Demo](https://drive.google.com/drive/folders/1Dfe_ciWhRGS7le7JQgYVDZxollSi7xG5?usp=sharing)  
*(Video minh họa chi tiết toàn bộ Data Pipeline của nhóm: bắt đầu từ việc tự động crawl video trên mạng, cắt nhỏ đoạn âm thanh (chunking), lọc nhiễu, sử dụng AI để gán nhãn sơ bộ (Pre-labeling), cho đến bước cuối cùng là nghe kiểm tra và tinh chỉnh nhãn thủ công trên giao diện Label Studio).*

🔗 **Link Dataset (Google Drive)**: [Link Dataset](https://drive.google.com/drive/folders/1vZep6AEUFfs9YdOrtMnDRqO0B4CDXAQE?usp=sharing)

## 🛠 Yêu cầu hệ thống (Requirements)
Để chạy dự án, bạn cần cài đặt môi trường Python và các thư viện cần thiết đã được định nghĩa sẵn trong file `requirements.txt`.

Cài đặt nhanh thông qua lệnh:
```bash
pip install -r requirements.txt
```

## ⚙️ Hướng dẫn cài đặt và sử dụng (Settings)

**Bước 1: Clone repository**
```bash
git clone <đường-dẫn-repo-của-bạn>
cd ViSER-Vietnamese-Speech-Emotion-Recognition
```

**Bước 2: Tải và thiết lập dữ liệu**
- Tải dataset từ link Google Drive ở phần trên.
- Giải nén và đặt toàn bộ dữ liệu vào thư mục `data/`.

**Bước 3: Trích xuất đặc trưng (Feature Extraction)**
Bạn cần chạy các script trích xuất để tạo ra file `.csv` (1D) và `.npy` (2D):
- Để trích xuất bằng Librosa: `python src/features/extract_librosa.py`
- Để trích xuất bằng Wav2Vec2: `python src/features/extract_wav2vec.py`

**Bước 4: Sử dụng mô hình đã huấn luyện sẵn (Pre-trained Checkpoints - Tùy chọn)**
Nếu bạn không muốn mất thời gian train lại từ đầu, nhóm đã cung cấp sẵn các file weights tốt nhất:
- 🔗 **Link Checkpoints (Google Drive)**: [Link Checkpoints](https://drive.google.com/drive/folders/1-ZNFPH0DaMaxszqx11XNuFwgqRYGiyZ9?usp=sharing)
- Tải về và giải nén vào thư mục `models/saved/`.

**Bước 5: Huấn luyện mô hình (Training)**
Sau khi đã có đặc trưng, tiến hành chạy các file training tương ứng:
- Huấn luyện với Machine Learning: `python src/train/train_ml.py`
- Huấn luyện với Deep Learning (VD: 1D-CNN): `python src/train/train_1dcnn.py`

*Lưu ý: Các biểu đồ đánh giá (Confusion Matrix, biểu đồ Accuracy/Loss) sẽ tự động được lưu vào thư mục `results/`.*

## 📁 Cấu trúc thư mục (Repository Structure)
```text
ViSER-Vietnamese-Speech-Emotion-Recognition/
├── data/               # Chứa dữ liệu âm thanh thô và metadata
├── models/
│   └──saved/           # Chứa các model đã được lưu lại sau khi train (.pth, .pkl)
├── results/            # Chứa kết quả đánh giá (Confusion Matrix, Loss/Acc curves, report)
├── src/                # Thư mục mã nguồn chính của hệ thống
│   ├── config.py       # File cấu hình đường dẫn và các hằng số siêu tham số
│   ├── crawler/        # Các script thu thập dữ liệu âm thanh tự động
│   ├── evaluation/     # Mã nguồn dùng để đánh giá và vẽ biểu đồ kết quả
│   ├── features/       # Mã nguồn trích xuất đặc trưng (Librosa, Wav2vec2)
│   ├── models/         # Định nghĩa cấu trúc các mô hình DL (CNN, Bi-LSTM) & ML
│   ├── train/          # Các script chạy huấn luyện (train_ml.py, train_1dcnn.py)
│   └── utils/          # Các hàm tiện ích hỗ trợ (Data loader, xử lý I/O)
├── requirements.txt    # Danh sách thư viện Python cần thiết
└── README.md           # Tài liệu hướng dẫn sử dụng dự án (File này)
```

## 👥 Thành viên nhóm (Group Members)
| STT | Họ và tên | MSSV |
| :---: | :--- | :---: |
| 1 | Nguyễn Đăng Khoa | 24520820 |
| 2 | Trần Minh Thiện | 24521668 |
| 3 | Trần Ngọc Bảo Thiên | 24521669 |
| 4 | Nguyễn Tiến Thịnh | 24521698 |