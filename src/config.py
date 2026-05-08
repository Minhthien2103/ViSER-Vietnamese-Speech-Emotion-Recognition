# File: src/config.py
import os
from pathlib import Path

# 1. TÌM THƯ MỤC GỐC CỦA DỰ ÁN
# __file__ là vị trí của file config.py (đang ở trong src/)
# .parent.parent sẽ lùi 2 bước ra ngoài thư mục gốc (ViSER-Framework/)
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. CẤU HÌNH ĐƯỜNG DẪN DATA
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SEGMENTS_DIR = DATA_DIR / "segments"
PROCESSED_DIR = DATA_DIR / "processed"

# 3. CẤU HÌNH FILE METADATA
TRAIN_META = DATA_DIR / "train_metadata.csv"
TEST_META = DATA_DIR / "test_metadata.csv"
MASTER_META = DATA_DIR / "master_metadata.csv"

# 4. CẤU HÌNH TÊN FILE ĐẦU RA
    # Librosa
LIBROSA_TRAIN_1D = PROCESSED_DIR / "train_librosa_1d.csv"
LIBROSA_TRAIN_2D = PROCESSED_DIR / "train_librosa_2d"     
LIBROSA_TEST_1D = PROCESSED_DIR / "test_librosa_1d.csv"
LIBROSA_TEST_2D = PROCESSED_DIR / "test_librosa_2d"        

    # Wav2Vec
WAV2VEC_TRAIN_1D = PROCESSED_DIR / "train_wav2vec_1d.csv"
WAV2VEC_TRAIN_2D = PROCESSED_DIR / "train_wav2vec_2d"
WAV2VEC_TEST_1D = PROCESSED_DIR / "test_wav2vec_1d.csv"
WAV2VEC_TEST_2D = PROCESSED_DIR / "test_wav2vec_2d"

# 5. SIÊU THAM SỐ ÂM THANH
SAMPLE_RATE = 16000
MAX_DURATION = 5.0

# Tự động tạo thư mục nếu chưa tồn tại
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)