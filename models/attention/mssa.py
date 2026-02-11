import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceRelativeBias(nn.Module):
    def __init__(self, max_slices=512, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.bias_table = nn.Parameter(torch.zeros(2 * max_slices - 1, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        self.max_slices = max_slices

    def forward(self, seq_len):
        coords = torch.arange(seq_len, device=self.bias_table.device)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords = relative_coords + self.max_slices - 1
        relative_coords = relative_coords.clamp(0, 2 * self.max_slices - 2)
        bias = self.bias_table[relative_coords]
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        return bias


class MultiHeadSliceSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, max_slices=512, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)
        self.W_O = nn.Linear(dim, dim)
        self.relative_bias = SliceRelativeBias(max_slices, num_heads)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_tokens, text_tokens):
        Z = torch.cat([visual_tokens, text_tokens], dim=1)
        B, N, D = Z.shape
        Z = self.norm(Z)
        Q = self.W_Q(Z).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.W_K(Z).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.W_V(Z).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        R = self.relative_bias(N)
        attn = attn + R
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.W_O(out)
        return out


class MSSABlock(nn.Module):
    def __init__(self, dim, num_heads=4, max_slices=512, dropout=0.0, mlp_ratio=4):
        super().__init__()
        self.mssa = MultiHeadSliceSelfAttention(dim, num_heads, max_slices, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, visual_tokens, text_tokens):
        Z = torch.cat([visual_tokens, text_tokens], dim=1)
        attn_out = self.mssa(visual_tokens, text_tokens)
        Z = self.norm1(Z + attn_out)
        Z = self.norm2(Z + self.mlp(Z))
        n_v = visual_tokens.shape[1]
        return Z[:, :n_v, :], Z[:, n_v:, :]


class MSSAModule(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=4, max_slices=512, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            MSSABlock(dim, num_heads, max_slices, dropout) for _ in range(num_layers)
        ])

    def forward(self, visual_tokens, text_tokens):
        v, t = visual_tokens, text_tokens
        for block in self.blocks:
            v, t = block(v, t)
        return v, t
