import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim * 4)
        self.fc2 = nn.Linear(output_dim * 4, output_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, output_dim, num_heads=8, proj_out_num=256):
        super().__init__()
        self.proj_out_num = proj_out_num
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPLayer(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, dim=1) if len(x) > 1 else x[0]
        B, N, C = x.shape
        residual = x
        x = self.norm1(x)
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        x = self.output_proj(x)
        x = self.output_norm(x)
        return x
