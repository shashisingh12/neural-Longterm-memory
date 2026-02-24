"""
Titans Memory — Attention Module
==================================
Multi-head self-attention used in the MAC (Memory as Context)
architecture (Paper §4.1, Eq. 22-23).

Processes the concatenated sequence:
    S̃^(t) = [p1, p2, ..., p_Np] || h_t || S^(t)

where:
    p_i   = learnable persistent memory vectors (task knowledge)
    h_t   = retrieved long-term memory  M*_{t-1}(q_t)
    S^(t) = current input segment embedding

The attention decides which information (persistent, historical,
or current) is relevant for the output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import TitansConfig


class MemoryAttention(nn.Module):
    """Multi-head self-attention for the MAC architecture.

    Attends over the concatenated sequence of:
        [persistent vectors || memory read || current input]

    This lets the model decide how to mix historical and current context.
    """

    def __init__(self, config: TitansConfig):
        super().__init__()

        self.d_model = config.d_model
        self.num_heads = config.num_attention_heads
        self.head_dim = config.d_model // config.num_attention_heads
        self.dropout_rate = config.attention_dropout

        assert config.d_model % config.num_attention_heads == 0, \
            f"d_model ({config.d_model}) must be divisible by num_heads ({config.num_attention_heads})"

        # QKV projections for attention
        self.W_attn_q = nn.Linear(self.d_model, self.d_model)
        self.W_attn_k = nn.Linear(self.d_model, self.d_model)
        self.W_attn_v = nn.Linear(self.d_model, self.d_model)
        self.W_out = nn.Linear(self.d_model, self.d_model)

        # Layer norm (pre-norm style)
        self.norm = nn.LayerNorm(self.d_model)

        self.dropout = nn.Dropout(self.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_attn_q, self.W_attn_k, self.W_attn_v, self.W_out]:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention over the concatenated sequence.

        Args:
            x: (batch, seq_len, d_model) — the concatenated
               [persistent || h_t || S^(t)] sequence.

        Returns:
            (batch, seq_len, d_model) — attention output.
        """
        residual = x
        x = self.norm(x)

        B, S, D = x.shape

        # Project to Q, K, V
        q = self.W_attn_q(x)  # (B, S, D)
        k = self.W_attn_k(x)  # (B, S, D)
        v = self.W_attn_v(x)  # (B, S, D)

        # Reshape to multi-head: (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, S, S)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # (B, H, S, head_dim)

        # Reshape back: (B, S, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection + residual
        out = self.W_out(attn_out)
        out = self.dropout(out)
        return residual + out


class PersistentMemoryVectors(nn.Module):
    """Learnable persistent memory parameter vectors.

    Paper §3.3 / §4.1:  P = [p1, p2, ..., p_Np]
    These are input-independent learnable parameters that encode
    task-level knowledge. Fixed at test time.

    Different from PersistentMemory (text tokens for prompt building) —
    these are actual nn.Parameter vectors that participate in attention.
    """

    def __init__(self, num_vectors: int, d_model: int):
        super().__init__()
        self.num_vectors = num_vectors
        self.d_model = d_model

        # Learnable persistent vectors: (N_p, d_model)
        self.vectors = nn.Parameter(torch.randn(num_vectors, d_model) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Expand persistent vectors for a batch.

        Returns:
            (batch_size, N_p, d_model)
        """
        return self.vectors.unsqueeze(0).expand(batch_size, -1, -1)
