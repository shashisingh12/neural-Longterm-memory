"""
Titans Memory — Text Decoder (Retrieval Store)
================================================
Nearest-neighbour cosine lookup: vector → past text snippets.
Stores (vector, text) pairs and retrieves top-k by similarity.
"""

import torch
from typing import List, Optional


class TextDecoder:
    """Nearest-neighbor retrieval store mapping vectors back to text."""

    def __init__(self, top_k: int, similarity_threshold: float, device: torch.device):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.device = device
        self._vectors: List[torch.Tensor] = []
        self._texts: List[str] = []

    def register(self, vec: torch.Tensor, text: str) -> None:
        """Store a vector-text pair for future retrieval."""
        self._vectors.append(vec.detach().cpu())
        self._texts.append(text)

    def decode(self, query_vec: torch.Tensor, top_k: Optional[int] = None) -> List[str]:
        """Return top-k texts by cosine similarity to query_vec."""
        if not self._vectors:
            return []

        top_k = top_k if top_k is not None else self.top_k
        query = query_vec.detach().squeeze().cpu()

        mat = torch.stack(self._vectors)                # (N, d_model)
        sims = torch.mv(mat, query)                     # (N,)
        k = min(top_k, len(self._texts))
        values, indices = torch.topk(sims, k)

        results = []
        for idx, sim in zip(indices.tolist(), values.tolist()):
            if sim > self.similarity_threshold:
                results.append(self._texts[idx])
        return results

    def __len__(self) -> int:
        return len(self._texts)

    def state_dict(self) -> dict:
        return {
            "vectors": [v.tolist() for v in self._vectors],
            "texts": list(self._texts),
        }

    def load_state_dict(self, d: dict) -> None:
        self._vectors = [torch.tensor(v, dtype=torch.float32) for v in d["vectors"]]
        self._texts = list(d["texts"])
