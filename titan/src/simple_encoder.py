"""
Tiktoken Text Encoder
=======================
Encodes text into fixed-size d_model vectors using tiktoken BPE tokenization.

Pipeline:
    text → tiktoken BPE tokens → random embedding lookup → mean pool → L2 normalize

The embedding table is a fixed random projection (seeded, deterministic).
No learned parameters. Fast, lightweight, semantically richer than n-grams
because tiktoken's BPE captures subword structure.

Uses cl100k_base encoding (same as GPT-4 / GPT-3.5-turbo).
"""

import numpy as np
import tiktoken
import torch


class TiktokenEncoder:
    """
    Text encoder using tiktoken BPE tokenization + random embeddings.

    Each BPE token ID maps to a fixed random vector via a seeded
    embedding table. The final text vector is the mean of all token
    embeddings, L2-normalized.

    This gives proper subword-level representations:
      - "diabetes" → single token → specific vector
      - "Alice"    → single token → specific vector
      - "What is 2+2?" → multiple tokens → blended vector

    Much richer than character n-grams while still being deterministic
    and requiring zero GPU or learned weights.
    """

    def __init__(
        self,
        d_model: int = 64,
        seed: int = 0,
        encoding_name: str = "cl100k_base",
    ):
        self.d_model = d_model
        self.seed = seed
        self.encoding_name = encoding_name

        # tiktoken BPE encoder
        self.enc = tiktoken.get_encoding(encoding_name)
        vocab_size = self.enc.n_vocab

        # fixed random embedding table: (vocab_size, d_model)
        # seeded for reproducibility — same seed always gives same table
        rng = np.random.default_rng(seed)
        self.embed_table = rng.normal(
            0, 1.0 / np.sqrt(d_model), (vocab_size, d_model)
        ).astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to L2-normalized d_model vector.

        Args:
            text: input string

        Returns:
            (d_model,) numpy array, L2-normalized
        """
        token_ids = self.enc.encode(text)

        if not token_ids:
            return np.zeros(self.d_model, dtype=np.float32)

        # look up embeddings for each token
        embeddings = self.embed_table[token_ids]  # (n_tokens, d_model)

        # mean pool
        vec = embeddings.mean(axis=0)  # (d_model,)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    def encode_tensor(
        self, text: str, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Encode text and return as torch tensor.

        Args:
            text: input string
            device: torch device string
            dtype: torch dtype for the output tensor

        Returns:
            (d_model,) tensor on specified device
        """
        arr = self.encode(text)
        return torch.tensor(arr, dtype=dtype, device=device)
