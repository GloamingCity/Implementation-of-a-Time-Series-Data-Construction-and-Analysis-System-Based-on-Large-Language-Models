import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, seq_len=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.net(x)
