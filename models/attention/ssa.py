import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.dim)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_tokens, text_tokens):
        V = self.norm_v(visual_tokens)
        T = self.norm_t(text_tokens)
        attn_weights = torch.matmul(V, T.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, T)
        return out


class SSAFusionLayer(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.ssa = SliceSelfAttention(dim, num_heads, dropout)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, visual_tokens, text_tokens):
        fused = self.ssa(visual_tokens, text_tokens)
        out = visual_tokens + self.gate * fused
        out = self.norm(out)
        return out


class SSABridge(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=4, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            SSAFusionLayer(dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, visual_tokens, text_tokens):
        x = visual_tokens
        for layer in self.layers:
            x = layer(x, text_tokens)
        return x
