import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


TRAIN_META = config.TRAIN_META
LIBROSA_TRAIN_1D = config.LIBROSA_TRAIN_1D
LIBROSA_TRAIN_2D = config.LIBROSA_TRAIN_2D

TEST_META = config.TEST_META
LIBROSA_TEST_1D = config.LIBROSA_TEST_1D
LIBROSA_TEST_2D = config.LIBROSA_TEST_2D


def extract_librosa_features(csv_in, csv_out_1d, dir_out_2d):
    df = pd.read_csv(csv_in)
    features_1d_list = []
    dir_out_2d.mkdir(parents=True, exist_ok=True)
    
    sr = config.SAMPLE_RATE
    max_length = int(config.MAX_DURATION * sr)
    
    print(f"Bắt đầu trích xuất Librosa cho {len(df)} files...")

    for idx, row in tqdm(df.iterrows(), total = len(df)):
        file_path = config.BASE_DIR / row['path']
        # audio_id: speaker_id + row index + file_id đảm bảo unique kể cả khi augment
        audio_id = f"{row['speaker_id']}_{idx}_{row['file_id']}"
        
        try:
            y, _ = librosa.load(file_path, sr=sr)
            
            # Padding / Truncating (Fix audio length = 5s) - For 2D (Advance models)
            if len(y) > max_length:
                y_2d = y[:max_length]
            else:
                y_2d = np.pad(y, (0, max_length - len(y)), mode = 'constant')
                
            # MFCC + Del ZCRtas
            mfcc_2d = librosa.feature.mfcc(y = y_2d, sr = sr, n_mfcc = 40)              # (40, T) - Âm sắc
            mfcc_delta_2d = librosa.feature.delta(mfcc_2d)                          # (40, T) - Vận tốc thay đổi âm sắc
            mfcc_delta2_2d = librosa.feature.delta(mfcc_2d, order = 2)                # (40, T) - Gia tốc thay đổi âm sắc

            mfcc_raw = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 40)
            mfcc_1d = np.mean(mfcc_raw.T, axis = 0)
            mfcc_delta_1d = np.mean(librosa.feature.delta(mfcc_raw).T, axis = 0) 
            mfcc_delta2_1d = np.mean(librosa.feature.delta(mfcc_raw, order = 2).T, axis = 0)

            # ZCR
            zcr_2d = librosa.feature.zero_crossing_rate(y = y_2d)                    # (1, T) - Nhiễu
            zcr_1d = np.mean(librosa.feature.zero_crossing_rate(y = y).T, axis = 0)

            # RMS
            rms_2d = librosa.feature.rms(y = y_2d)                                   # (1, T) - Âm lượng
            rms_1d = np.mean(librosa.feature.rms(y = y).T, axis = 0)

            # Spectral Centroid
            sc_2d = librosa.feature.spectral_centroid(y = y_2d, sr = sr)               # (1, T) - Độ sáng âm thanh
            sc_1d = np.mean(librosa.feature.spectral_centroid(y = y, sr = sr).T, axis = 0)

            # Spectral Contrast
            scon_2d = librosa.feature.spectral_contrast(y = y_2d, sr = sr)             # (7, T) - Tỉ số đỉnh-lõm trên 7 dải tần số
            scon_1d = np.mean(librosa.feature.spectral_contrast(y = y, sr = sr).T, axis = 0)

            # Pitch (F0) — 1D only (pyin has NaNs, not ideal for 2D tensors) - Âm / Ngữ điệu
            f0, _, _ = librosa.pyin(y, fmin = librosa.note_to_hz('C2'),
                                    fmax = librosa.note_to_hz('C7'), sr = sr)
            f0_voiced = f0[~np.isnan(f0)] if np.any(~np.isnan(f0)) else np.array([0.0])
            pitch_mean = np.mean(f0_voiced)
            pitch_std = np.std(f0_voiced)

            # 3. Gom 1D — Basic models (132 features)
            # MFCC: 40, MFCC_d: 40, MFCC_dd: 40, ZCR: 1, RMS: 1, SC: 1, SCON: 7, Pitch: 2
            feature_1d = np.hstack([
                mfcc_1d, mfcc_delta_1d, mfcc_delta2_1d,
                zcr_1d, rms_1d, sc_1d, scon_1d,
                [pitch_mean, pitch_std],
            ])

            row_data = row.to_dict()
            for i in range(len(feature_1d)):
                row_data[f'librosa_feat_{i}'] = feature_1d[i]

            features_1d_list.append(row_data)

            # 4. Gom 2D — Advance models (130 features per frame)
            # MFCC: 40, MFCC_d: 40, MFCC_dd: 40, ZCR: 1, RMS: 1, SC: 1, SCON: 7
            feature_2d = np.vstack([
                mfcc_2d, mfcc_delta_2d, mfcc_delta2_2d,
                zcr_2d, rms_2d, sc_2d, scon_2d,
            ]).T  # (T, 130)

            # Tránh tràn RAM
            np.save(dir_out_2d / f"{audio_id}.npy", feature_2d.astype(np.float32))
            
        except Exception as e:
            print(f"Lỗi file {file_path}: {e}")

            continue
            
    pd.DataFrame(features_1d_list).to_csv(csv_out_1d, index=False)

    print(f"Xong! 1D -> {csv_out_1d} | 2D -> {dir_out_2d}/ ({len(features_1d_list)} files)")


if __name__ == "__main__":
    extract_librosa_features(TRAIN_META, LIBROSA_TRAIN_1D, LIBROSA_TRAIN_2D)
    
    extract_librosa_features(TEST_META, LIBROSA_TEST_1D, LIBROSA_TEST_2D)