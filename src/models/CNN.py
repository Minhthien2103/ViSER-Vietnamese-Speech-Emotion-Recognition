import torch
import torch.nn as nn


class CNN1DModel(nn.Module):
    """
    Mô hình 1D CNN cho Speech Emotion Recognition.
    - Xử lý features đầu vào (Batch, Seq_Len, Channels)
    - Dùng các khối Conv1d + BatchNorm1d + ReLU + MaxPool1d
    - AdaptiveMaxPool1d để đưa Seq_Len về 1 bất kể đầu vào dài bao nhiêu.
    - Gắn thêm metadata (nếu có) trước khi qua các lớp Fully Connected cuối.
    """
    def __init__(self, input_channels, num_meta = 0, num_classes = 4):
        super().__init__()
        
        # Khối CNN trích xuất đặc trưng âm thanh
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels = input_channels, out_channels = 128, kernel_size = 5, padding = 2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 5, padding = 2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # Gom toàn bộ chiều thời gian lại
        )
        
        # Nhánh xử lý Metadata (gender, accent)
        self.has_meta = num_meta > 0
        if self.has_meta:
            self.meta_fc = nn.Sequential(
                nn.Linear(num_meta, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )
            fc1_in = 512 + 32
        else:
            fc1_in = 512
            
        # Khối phân loại (Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(fc1_in, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, meta=None):
        # x shape ban đầu: (Batch, Seq_Len, Channels)
        # Conv1d của PyTorch yêu cầu: (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        x = x.squeeze(-1) # Xóa chiều thời gian -> Shape: (Batch, 512)
        
        if self.has_meta and meta is not None:
            meta_feat = self.meta_fc(meta)
            x = torch.cat([x, meta_feat], dim=1)
            
        out = self.classifier(x)
        return out
