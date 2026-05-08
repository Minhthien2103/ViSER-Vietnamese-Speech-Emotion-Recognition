import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

META_COLS = ['gender', 'accent']
VAL_RATIO = 0.2
RANDOM_STATE = 42


def load_features(csv_path, feature_prefix):
    """
        Tách X (features), metadata (gender, accent), và y (labels) từ CSV.
    """
    df = pd.read_csv(csv_path)

    feat_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    X = df[feat_cols].values
    meta = df[META_COLS]
    y = df['emotion_id'].values

    print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} features từ {csv_path.name}")

    return X, meta, y


def encode_metadata(*metas):
    """
        One-hot encode gender và accent.
        Fit categories trên tất cả sets để đảm bảo cùng số cột.
    """
    combined = pd.concat(metas, axis = 0)
    dummies = pd.get_dummies(combined, columns = META_COLS, dtype = float)

    result = []
    start = 0
    for m in metas:
        end = start + len(m)
        result.append(dummies.iloc[start:end].values)
        start = end

    print(f"  Metadata one-hot: {list(dummies.columns)} → {dummies.shape[1]} features")

    return result


def get_data(source = "librosa"):
    """
        Load train/val/test data dựa trên feature source (librosa hoặc wav2vec).
        Train được split thành train/val (80:20) với stratify.
        Trả về: X_train, y_train, X_val, y_val, X_test, y_test
    """
    if source == "librosa":
        train_csv = config.LIBROSA_TRAIN_1D
        test_csv = config.LIBROSA_TEST_1D
        prefix = "librosa_feat_"

    elif source == "wav2vec":
        train_csv = config.WAV2VEC_TRAIN_1D
        test_csv = config.WAV2VEC_TEST_1D
        prefix = "w2v_feat_"
    else:

        raise ValueError(f"Feature source không hợp lệ: {source}")

    print(f"\n Loading {source} features...")

    X_full, meta_full, y_full = load_features(train_csv, prefix)
    X_test, meta_test, y_test = load_features(test_csv, prefix)

    # Split train → train + val (80:20, stratified)
    X_train, X_val, meta_train, meta_val, y_train, y_val = train_test_split(
        X_full, meta_full, y_full,
        test_size = VAL_RATIO,
        stratify = y_full,
        random_state = RANDOM_STATE,
    )

    meta_train = meta_train.reset_index(drop = True)
    meta_val = meta_val.reset_index(drop = True)
    meta_test = meta_test.reset_index(drop = True)

    # One-hot encode metadata và ghép vào features
    meta_train_enc, meta_val_enc, meta_test_enc = encode_metadata(meta_train, meta_val, meta_test)
    X_train = np.hstack([X_train, meta_train_enc])
    X_val = np.hstack([X_val, meta_val_enc])
    X_test = np.hstack([X_test, meta_test_enc])

    print(f"  Final: {X_train.shape[1]} features (audio + metadata)")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test

