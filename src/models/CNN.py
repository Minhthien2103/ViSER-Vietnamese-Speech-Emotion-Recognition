import torch
import torch.nn as nn


class CNN1DModel(nn.Module):
    """
    Mô hình 1D CNN nâng cao cho Speech Emotion Recognition.
    - Residual connections để gradient chảy sâu hơn.
    - Spatial Dropout (dropout trên channel) chống overfit hiệu quả hơn.
    - Kết hợp AdaptiveAvgPool + AdaptiveMaxPool cho temporal pooling mạnh hơn.
    - Gắn thêm metadata (nếu có) trước khi qua các lớp Fully Connected cuối.
    """

    def __init__(self, input_channels, num_meta = 0, num_classes = 4, dropout = 0.3):
        super().__init__()

        # Projection layer — đưa input_channels về kích thước chuẩn
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size = 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Residual Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size = 5, padding = 2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size = 5, padding = 2),
            nn.BatchNorm1d(128),
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size = 2)

        # Residual Block 2
        self.downsample2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size = 1),
            nn.BatchNorm1d(256),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(256),
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size = 2)

        # Residual Block 3
        self.downsample3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size = 1),
            nn.BatchNorm1d(512),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(512),
        )
        self.relu3 = nn.ReLU()

        # Dual Pooling — gộp cả Avg và Max để không mất thông tin
        self.adaptive_avg = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max = nn.AdaptiveMaxPool1d(1)
        # → Concat: 512 + 512 = 1024

        # Nhánh xử lý Metadata (gender, accent)
        self.has_meta = num_meta > 0
        if self.has_meta:
            self.meta_fc = nn.Sequential(
                nn.Linear(num_meta, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
            )
            fc_in = 1024 + 32
        else:
            fc_in = 1024

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, meta = None):
        # x: (Batch, Seq_Len, Channels) → Conv1d cần (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_proj(x)

        # Residual Block 1 (same channels: 128 → 128)
        residual = x
        x = self.conv_block1(x)
        x = self.relu1(x + residual)
        x = self.pool1(x)

        # Residual Block 2 (128 → 256, needs downsample)
        residual = self.downsample2(x)
        x = self.conv_block2(x)
        x = self.relu2(x + residual)
        x = self.pool2(x)

        # Residual Block 3 (256 → 512, needs downsample)
        residual = self.downsample3(x)
        x = self.conv_block3(x)
        x = self.relu3(x + residual)

        # Dual Pooling
        avg_pool = self.adaptive_avg(x).squeeze(-1)   # (Batch, 512)
        max_pool = self.adaptive_max(x).squeeze(-1)    # (Batch, 512)
        x = torch.cat([avg_pool, max_pool], dim = 1)   # (Batch, 1024)

        # Metadata fusion
        if self.has_meta and meta is not None:
            meta_feat = self.meta_fc(meta)
            x = torch.cat([x, meta_feat], dim = 1)

        out = self.classifier(x)
        return out
