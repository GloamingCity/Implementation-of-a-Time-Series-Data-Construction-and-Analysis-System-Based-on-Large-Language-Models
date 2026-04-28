import torch
import torch.nn as nn

class PatchTSTEncoder(nn.Module):
    def __init__(self, seq_len=512, patch_len=16, stride=8, d_model=128, n_heads=4, num_layers=2):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.flatten = nn.Flatten(start_dim=1)
        self.output_proj = nn.Linear(self.num_patches * d_model, d_model)
        
        self.output_dim = d_model

    def forward(self, x):
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.patch_proj(patches) + self.pos_embedding
        x = self.transformer(x)
        x = self.flatten(x)
        x = self.output_proj(x)
        return x
