import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    """
    Mô hình Bi-LSTM cho Speech Emotion Recognition.
    - Xử lý chuỗi thời gian của features: (Batch, Seq_Len, Channels)
    - Tự động lấy đặc trưng trung bình trên toàn chuỗi (Global Average Pooling) để chống nhiễu từ padding.
    - Kết hợp nhánh metadata nếu có.
    """

    def __init__(self, input_size, hidden_size = 128, num_layers = 2, num_meta = 0, num_classes = 4):
        super().__init__()
        
        # Lớp Bi-LSTM trích xuất ngữ cảnh 2 chiều (quá khứ và tương lai)
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = 0.3 if num_layers > 1 else 0
        )
        
        # Tính kích thước sau LSTM (Do là Bi-LSTM nên output sẽ x2)
        lstm_out_dim = hidden_size * 2
        
        # Xử lý metadata
        self.has_meta = num_meta > 0
        if self.has_meta:
            self.meta_fc = nn.Sequential(
                nn.Linear(num_meta, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )
            fc_in_dim = lstm_out_dim + 32
        else:
            fc_in_dim = lstm_out_dim
            
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fc_in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )


    def forward(self, x, meta=None):
        # x shape: (Batch, Seq_Len, Channels)
        
        # lstm_out shape: (Batch, Seq_Len, hidden_size*2)
        lstm_out, _ = self.lstm(x)
        
        # Áp dụng Global Average Pooling theo chiều thời gian để lấy đại diện cho cả đoạn audio
        # Mean pooling tốt hơn việc chỉ lấy lstm_out[:, -1, :] (hidden state cuối cùng)
        x = torch.mean(lstm_out, dim = 1) # Shape -> (Batch, hidden_size*2)
        
        # Nối Metadata
        if self.has_meta and meta is not None:
            meta_feat = self.meta_fc(meta)
            x = torch.cat([x, meta_feat], dim = 1)
            
        out = self.classifier(x)
        return out
