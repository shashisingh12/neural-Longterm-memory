"""
Titans Memory — Text Encoder
==============================
Converts text → d_model vectors using a pretrained HuggingFace model.

Architecture:
    text → AutoTokenizer → token IDs
         → AutoModel (frozen by default) → contextual embeddings
         → pooling (mean / cls / max) → single vector
         → nn.Linear projection → d_model dimension
         → L2 normalize
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import List, Union

from transformers import AutoTokenizer, AutoModel

from .config import TitansConfig


class TextEncoder(nn.Module):
    """Encodes text into L2-normalised vectors of size d_model."""

    def __init__(self, config: TitansConfig):
        super().__init__()

        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.freeze_backbone = config.freeze_backbone
        self.pooling_strategy = config.pooling_strategy
        self._device = torch.device(config.device)

        # HuggingFace tokenizer and pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.backbone = AutoModel.from_pretrained(config.tokenizer_name)

        # Get backbone hidden size (works for BERT, DistilBERT, RoBERTa, etc.)
        backbone_dim = self.backbone.config.hidden_size

        # Projection from backbone dim → memory d_model
        self.projection = nn.Linear(backbone_dim, self.d_model)
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        # Freeze backbone if requested (default: True)
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Encode text(s) to L2-normalised (batch, d_model) vectors.

        Args:
            text: A single string or list of strings.

        Returns:
            Tensor of shape (batch, d_model), L2-normalised.
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}

        # Run backbone
        ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        with ctx:
            outputs = self.backbone(**tokens)

        hidden_states = outputs.last_hidden_state  # (batch, seq_len, backbone_dim)

        # Pool over sequence dimension
        if self.pooling_strategy == "cls":
            pooled = hidden_states[:, 0, :]
        elif self.pooling_strategy == "mean":
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling_strategy == "max":
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            # Set padding positions to large negative before max
            hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            pooled = hidden_states.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Project to d_model
        projected = self.projection(pooled)

        # L2 normalise
        return F.normalize(projected, p=2, dim=-1)

    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Public API matching the reference interface.

        For single strings, returns (d_model,) squeezed.
        For lists, returns (batch, d_model).
        """
        was_single = isinstance(text, str)
        out = self.forward(text)
        if was_single:
            out = out.squeeze(0)
        return out

    def to(self, device, *args, **kwargs):
        self._device = torch.device(device) if isinstance(device, str) else device
        return super().to(device, *args, **kwargs)
