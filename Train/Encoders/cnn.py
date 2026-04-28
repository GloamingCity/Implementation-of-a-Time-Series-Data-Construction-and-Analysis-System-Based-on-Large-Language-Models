import torch
import torch.nn as nn

class CNN1DEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.output_dim = out_channels

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return x
