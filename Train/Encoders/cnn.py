import torch
import torch.nn as nn

class CNN1DEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, kernel_size=3):
        super().__init__()
        # 1D 卷积层：提取局部波形特征
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1) # 将序列长度压缩为 1，方便对齐
        
        self.output_dim = out_channels

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = x.unsqueeze(1) # -> (batch_size, 1, seq_len)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)   # -> (batch_size, out_channels, 1)
        x = x.squeeze(-1)  # -> (batch_size, out_channels)
        return x