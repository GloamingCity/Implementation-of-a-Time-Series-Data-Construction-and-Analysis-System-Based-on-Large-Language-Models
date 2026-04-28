import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, seq_len=512, hidden_dim=256):
        super().__init__()
        # 直接把整条时间序列拍平，用线性层提取特征
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        return self.net(x) # -> (batch_size, hidden_dim)