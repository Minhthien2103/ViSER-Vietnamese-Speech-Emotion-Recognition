import pandas as pd
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h" # "vinai/PhoWav2Vec2-base"

TRAIN_META = config.TRAIN_META
WAV2VEC_TRAIN_1D = config.WAV2VEC_TRAIN_1D
WAV2VEC_TRAIN_2D = config.WAV2VEC_TRAIN_2D

TEST_META = config.TEST_META
WAV2VEC_TEST_1D = config.WAV2VEC_TEST_1D
WAV2VEC_TEST_2D = config.WAV2VEC_TEST_2D


def extract_wav2vec_features(csv_in, csv_out_1d, dir_out_2d, model_name = MODEL_NAME):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    
    df = pd.read_csv(csv_in)
    features_1d_list = []
    dir_out_2d.mkdir(parents=True, exist_ok=True)
    
    sr = config.SAMPLE_RATE
    max_length = int(config.MAX_DURATION * sr)
    
    print(f"Bắt đầu trích xuất Wav2Vec2 cho {len(df)} files...")
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total = len(df)):
            file_path = config.BASE_DIR / row['path'] 
            audio_id = f"{row['speaker_id']}_{idx}_{row['file_id']}"
            
            try:
                y, _ = librosa.load(file_path, sr=sr)
                
                # 1D: audio gốc → mean pooling tự nhiên hơn
                input_orig = processor(y, sampling_rate = sr, return_tensors = "pt").input_values.to(device)
                hidden_orig = model(input_orig).last_hidden_state.squeeze(0)
                feature_1d = hidden_orig.mean(dim = 0).cpu().numpy()
                
                row_data = row.to_dict()
                for i in range(len(feature_1d)):
                    row_data[f'w2v_feat_{i}'] = feature_1d[i]
                features_1d_list.append(row_data)
                
                # 2D: pad/truncate → fixed-length for CNN/BiLSTM
                if len(y) > max_length:
                    y_padded = y[:max_length]
                else:
                    y_padded = np.pad(y, (0, max_length - len(y)), mode = 'constant')
                
                input_padded = processor(y_padded, sampling_rate = sr, return_tensors = "pt").input_values.to(device)
                hidden_padded = model(input_padded).last_hidden_state.squeeze(0)
                
                # Tránh tràn RAM
                np.save(dir_out_2d / f"{audio_id}.npy", hidden_padded.cpu().numpy().astype(np.float32))
                
            except Exception as e:
                print(f"Lỗi file {file_path}: {e}")
                continue

    pd.DataFrame(features_1d_list).to_csv(csv_out_1d, index = False)

    print(f"Xong! 1D -> {csv_out_1d} | 2D -> {dir_out_2d}/ ({len(features_1d_list)} files)")


if __name__ == "__main__":
    extract_wav2vec_features(TRAIN_META, WAV2VEC_TRAIN_1D, WAV2VEC_TRAIN_2D)

    extract_wav2vec_features(TEST_META, WAV2VEC_TEST_1D, WAV2VEC_TEST_2D)