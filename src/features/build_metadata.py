"""
build_metadata.py
-----------------
Quét toàn bộ thư mục data/train và data/test, tạo ra các file CSV metadata
cần thiết cho pipeline trích xuất đặc trưng.

Cấu trúc thư mục dữ liệu:
    data/{train|test}/{Female|Male}/{mien_bac|mien_nam|mien_trung}/{Angry|Happy|Neutral|Sad}/*.wav

Tên file mẫu:
    Female_mien_bac_Angry_orig_0000.wav
    aug_Female_mien_bac_Angry_gain_dn3db_0010_0010.wav

Output:
    data/processed/train_metadata.csv
    data/processed/test_metadata.csv

Các cột trong CSV:
    - path       : đường dẫn tương đối từ BASE_DIR (dùng trong extractor)
    - file_id    : tên file không có đuôi
    - speaker_id : kết hợp gender + accent (e.g. Female_mien_bac)
    - gender     : Female | Male
    - accent     : mien_bac | mien_nam | mien_trung
    - emotion    : Angry | Happy | Neutral | Sad
    - emotion_id : 0=Happy, 1=Sad, 2=Angry, 3=Neutral
    - split      : train | test
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def scan_split(split_dir: Path, split_name: str) -> list[dict]:
    """Quét một thư mục split (train hoặc test), trả về list of row dicts."""
    rows = []

    # Cấu trúc: {split_dir}/{gender}/{accent}/{emotion}/*.wav
    for gender_dir in sorted(split_dir.iterdir()):
        if not gender_dir.is_dir():
            continue
        gender = gender_dir.name  # "Female" hoặc "Male"

        for accent_dir in sorted(gender_dir.iterdir()):
            if not accent_dir.is_dir():
                continue
            accent = accent_dir.name  # "mien_bac", "mien_nam", "mien_trung"

            for emotion_dir in sorted(accent_dir.iterdir()):
                if not emotion_dir.is_dir():
                    continue
                emotion = emotion_dir.name  # "Angry", "Happy", "Neutral", "Sad"

                emotion_id = config.EMOTION_LABELS.get(emotion)
                if emotion_id is None:
                    print(f"  [WARN] Emotion không nhận ra: {emotion}, bỏ qua.")
                    continue

                for wav_file in sorted(emotion_dir.glob("*.wav")):
                    # path tương đối so với BASE_DIR để load trong extractor
                    rel_path = wav_file.relative_to(config.BASE_DIR)

                    rows.append({
                        "path"       : str(rel_path).replace("\\", "/"),
                        "file_id"    : wav_file.stem,
                        "speaker_id" : f"{gender}_{accent}",
                        "gender"     : gender,
                        "accent"     : accent,
                        "emotion"    : emotion,
                        "emotion_id" : emotion_id,
                        "split"      : split_name,
                    })

    return rows


def build_metadata():
    print("Đang quét dữ liệu...")

    train_rows = scan_split(config.TRAIN_DIR, "train")
    test_rows  = scan_split(config.TEST_DIR,  "test")

    train_df = pd.DataFrame(train_rows)
    test_df  = pd.DataFrame(test_rows)

    train_df.to_csv(config.TRAIN_META, index=False)
    test_df.to_csv(config.TEST_META,   index=False)

    print(f"\nTrain metadata: {len(train_df)} files → {config.TRAIN_META}")
    print(f"Test  metadata: {len(test_df)} files → {config.TEST_META}")

    # Thống kê nhanh
    for split_name, df in [("TRAIN", train_df), ("TEST", test_df)]:
        print(f"\n--- {split_name} ---")
        print(df.groupby(["gender", "accent", "emotion"])["file_id"].count().to_string())


if __name__ == "__main__":
    build_metadata()
