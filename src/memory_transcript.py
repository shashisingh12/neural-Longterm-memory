"""
Titans Memory — Memory Transcript (Fixed-Size Ring Buffer)
============================================================
Replaces the unbounded TextDecoder (cosine RAG store) with a
fixed-size ring buffer that stores MLP-gated output vectors.

Key differences from TextDecoder:
    - Fixed capacity (FIFO eviction when full)
    - Stores MLP-space vectors (o_t = y_t * M*(y_t)), NOT raw BERT embeddings
    - Scoring via dot product in MLP-learned space (not cosine over BERT space)
    - The MLP actively shapes what gets retrieved through its learned weights
"""

import torch
from typing import List, Optional


class MemoryTranscript:
    """Fixed-size ring buffer storing (mlp_vector, text) pairs.

    The MLP memory's gated output o_t is stored alongside each text entry.
    Future retrievals score against these MLP vectors, meaning the neural
    memory actively drives what context gets surfaced.

    Args:
        max_size: Maximum number of entries (oldest evicted when full).
        top_k:    Default number of entries to retrieve.
        device:   Torch device for vector operations.
    """

    def __init__(self, max_size: int = 32, top_k: int = 3, device: torch.device = "cpu"):
        self.max_size = max_size
        self.top_k = top_k
        self.device = device
        self._pointer: int = 0
        self._count: int = 0
        self._vectors: List[Optional[torch.Tensor]] = [None] * max_size
        self._texts: List[Optional[str]] = [None] * max_size

    def register(self, mlp_vector: torch.Tensor, text: str) -> None:
        """Store a (vector, text) pair at the current write position.

        Args:
            mlp_vector: (d_model,) — the output-gated MLP vector o_t.
            text:       The Q+A text associated with this turn.
        """
        self._vectors[self._pointer] = mlp_vector.detach().cpu()
        self._texts[self._pointer] = text
        self._pointer = (self._pointer + 1) % self.max_size
        self._count = min(self._count + 1, self.max_size)

    def score_and_retrieve(
        self,
        query_vector: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """Score stored entries by dot product and return top-k texts.

        The query_vector should be an MLP-gated output o_t from the
        current turn. Dot product scoring in MLP-learned space means
        the neural memory actively determines what's relevant.

        Args:
            query_vector: (d_model,) — the gated output from current turn.
            top_k:        Override default top_k.

        Returns:
            List of top-k text strings, ordered by descending score.
        """
        if self._count == 0:
            return []

        top_k = top_k if top_k is not None else self.top_k
        query = query_vector.detach().squeeze().cpu()

        # Gather valid entries
        valid_vectors = []
        valid_indices = []
        for i in range(self._count):
            if self._vectors[i] is not None:
                valid_vectors.append(self._vectors[i])
                valid_indices.append(i)

        if not valid_vectors:
            return []

        mat = torch.stack(valid_vectors)              # (N, d_model)
        scores = torch.mv(mat, query)                 # (N,)
        k = min(top_k, len(valid_indices))
        _, top_idx = torch.topk(scores, k)

        results = []
        for idx in top_idx.tolist():
            real_idx = valid_indices[idx]
            results.append(self._texts[real_idx])
        return results

    def __len__(self) -> int:
        return self._count

    def state_dict(self) -> dict:
        """Serialize for checkpoint save."""
        return {
            "max_size": self.max_size,
            "pointer": self._pointer,
            "count": self._count,
            "vectors": [v.tolist() if v is not None else None
                        for v in self._vectors],
            "texts": list(self._texts),
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore from checkpoint."""
        self.max_size = d["max_size"]
        self._pointer = d["pointer"]
        self._count = d["count"]
        self._vectors = [
            torch.tensor(v, dtype=torch.float32) if v is not None else None
            for v in d["vectors"]
        ]
        self._texts = list(d["texts"])
