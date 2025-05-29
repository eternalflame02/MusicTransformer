# relative_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_len = max_len

        # projections for q, k, v
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        # Relative position embedding: (2L - 1, d_head)
        self.rel_embedding = nn.Parameter(torch.randn(2 * max_len - 1, self.d_head))

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        H = self.n_heads
        D = self.d_head

        # Linear projections
        Q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        K = self.k_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        V = self.v_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]

        # Content-based attention
        content_logits = torch.matmul(Q, K.transpose(-2, -1))  # [B, H, T, T]        # Relative attention
        rel_logits = self._relative_logits(Q, T)  # [B, H, T, T]

        logits = content_logits + rel_logits

        if mask is not None:
            # Ensure mask is broadcastable to [B, H, T, T]
            if mask.dim() == 2:  # [T, T]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
            elif mask.dim() == 3:  # [B, T, T]
                mask = mask.unsqueeze(1)  # [B, 1, T, T]
            logits = logits.masked_fill(~mask, float('-inf'))

        attn = F.softmax(logits / (D ** 0.5), dim=-1)
        out = torch.matmul(attn, V)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)  # [B, T, d_model]

        return self.out_proj(out)

    def _relative_logits(self, Q, T):
        rel_emb = self.rel_embedding[:2 * T - 1]  # [2T-1, D]
        rel_emb = rel_emb.to(Q.device)        # Matrix multiplication: [B, H, T, D] @ [D, 2T-1] → [B, H, T, 2T-1]
        rel_logits = torch.matmul(Q, rel_emb.transpose(0, 1))  # [B, H, T, 2T-1]

        # Skew to get [B, H, T, T]
        rel_logits = self._skew(rel_logits)
        return rel_logits

    def _skew(self, rel_logits):
        """
        Efficient skewing for relative attention. Converts [B, H, T, 2T - 1] → [B, H, T, T].
        """
        B, H, T, _ = rel_logits.shape

        # Step 1: Pad 1 column to the right → [B, H, T, 2T]
        pad = torch.zeros((B, H, T, 1), device=rel_logits.device, dtype=rel_logits.dtype)
        padded = torch.cat([rel_logits, pad], dim=-1)  # [B, H, T, 2T]

        # Step 2: Reshape to [B, H, 2T, T] 
        # This works because T * 2T = 2T * T (same number of elements)
        reshaped = padded.view(B, H, 2 * T, T)  # [B, H, 2T, T]
        
        # Step 3: Take the last T rows to get [B, H, T, T]
        # This aligns the relative positions correctly
        return reshaped[:, :, T:, :]  # [B, H, T, T]


