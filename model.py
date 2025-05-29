# model.py

import torch
import torch.nn as nn
from relative_attention import RelativeSelfAttention


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, num_layers=6, d_ff=2048, dropout=0.1, max_len=2048):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_len)
            for _ in range(num_layers)
        ])

        self.ln = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.token_embedding(x)
        x = self.dropout(x)

        # Generate causal mask [T, T]
        attn_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()

        for layer in self.layers:
            x = layer(x, attn_mask)

        x = self.ln(x)
        return self.output_layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, max_len):
        super().__init__()
        self.attn = RelativeSelfAttention(d_model, n_heads, max_len)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        # Relative attention
        attn_output = self.attn(x, attn_mask)
        x = x + self.dropout(attn_output)
        x = self.ln1(x)

        # Feedforward
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.ln2(x)
        return x
