import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


META_COLS = ['gender', 'accent']


class MetaEncoder:
    """
        Fit one-hot encoding trên train, transform cho cả train và test.
    """

    def __init__(self):
        self.categories = {}

    def fit(self, df):
        for col in META_COLS:
            self.categories[col] = sorted(df[col].unique())
        return self

    def transform(self, df):
        encoded = []
        for col in META_COLS:
            for cat in self.categories[col]:
                encoded.append((df[col] == cat).astype(float).values)
        return np.column_stack(encoded)


class SERDataset(Dataset):
    """
        PyTorch Dataset cho 2D features (CNN / BiLSTM).
        Load từng .npy file từ disk theo audio_id.
    """

    def __init__(self, csv_path, feature_dir, meta_encoder = None):
        self.feature_dir = Path(feature_dir)
        self.df = pd.read_csv(csv_path)

        # Tạo audio_id giống lúc extract
        self.df['audio_id'] = self.df.apply(
            lambda row: f"{row['speaker_id']}_{row.name}_{row['file_id']}", axis = 1
        )

        # Lọc chỉ giữ các samples có file .npy tồn tại
        self.df['npy_path'] = self.df['audio_id'].apply(lambda x: self.feature_dir / f"{x}.npy")
        valid_mask = self.df['npy_path'].apply(lambda p: p.exists())
        n_missing = (~valid_mask).sum()
        if n_missing > 0:
            print(f"  {n_missing} files không tìm thấy, bỏ qua.")
        self.df = self.df[valid_mask].reset_index(drop = True)

        # One-hot encode metadata
        if meta_encoder is not None:
            self.meta_encoded = meta_encoder.transform(self.df[META_COLS])
        else:
            self.meta_encoded = None

        self.labels = self.df['emotion_id'].values

        print(f"  Dataset: {len(self)} samples từ {feature_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load 2D features từ disk (lazy loading)
        npy_path = self.df.iloc[idx]['npy_path']
        features = np.load(npy_path)
        features = torch.tensor(features, dtype = torch.float32)  # (T, C)

        label = torch.tensor(self.labels[idx], dtype = torch.long)

        if self.meta_encoded is not None:
            meta = torch.tensor(self.meta_encoded[idx], dtype = torch.float32)
            return features, meta, label

        return features, label


def get_dl_dataloaders(source = "librosa", batch_size = 32, num_workers = 2, val_ratio = 0.2):
    """
        Tạo DataLoader cho train, val, và test.
        Train được split thành train/val (80:20) với stratify.
        Trả về: train_loader, val_loader, test_loader, num_features, num_meta, num_classes
    """
    if source == "librosa":
        train_csv = config.LIBROSA_TRAIN_1D
        test_csv = config.LIBROSA_TEST_1D
        train_dir = config.LIBROSA_TRAIN_2D
        test_dir = config.LIBROSA_TEST_2D

    elif source == "wav2vec":
        train_csv = config.WAV2VEC_TRAIN_1D
        test_csv = config.WAV2VEC_TEST_1D
        train_dir = config.WAV2VEC_TRAIN_2D
        test_dir = config.WAV2VEC_TEST_2D
        
    else:
        raise ValueError(f"Feature source không hợp lệ: {source}")

    print(f"\n Loading {source} 2D features...")

    # Fit metadata encoder trên train
    train_df = pd.read_csv(train_csv)
    encoder = MetaEncoder().fit(train_df)

    full_dataset = SERDataset(train_csv, train_dir, meta_encoder = encoder)
    test_dataset = SERDataset(test_csv, test_dir, meta_encoder = encoder)

    # Split train → train + val (stratified)
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size = val_ratio,
        stratify = full_dataset.labels,
        random_state = 42,
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size = batch_size,
        shuffle = True, num_workers = num_workers, pin_memory = True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size = batch_size,
        shuffle = False, num_workers = num_workers, pin_memory = True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size = batch_size,
        shuffle = False, num_workers = num_workers, pin_memory = True,
    )

    # Lấy thông tin dimensions từ 1 sample
    sample = full_dataset[0]
    num_features = sample[0].shape[1]     # (T, C) → C
    num_meta = sample[1].shape[0] if len(sample) == 3 else 0
    num_classes = len(np.unique(full_dataset.labels))

    print(f"  Features: {num_features}, Metadata: {num_meta}, Classes: {num_classes}")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} | Batch: {batch_size}")

    return train_loader, val_loader, test_loader, num_features, num_meta, num_classes

