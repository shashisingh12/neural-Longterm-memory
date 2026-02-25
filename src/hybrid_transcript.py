"""
Titans Memory — Hybrid Memory Transcript (FAISS-backed)
=========================================================
Replaces the fixed-size ring buffer with a persistent text store
indexed by FAISS for scalable inner-product retrieval.

Key differences from MemoryTranscript (ring buffer):
    - ALL texts are preserved (no FIFO eviction)
    - FAISS IndexFlatIP for SIMD-optimized inner product search
    - O(N) exact but hardware-accelerated scoring (vs naive torch.mv)
    - Graceful fallback to torch dot-product if faiss is not installed
    - Unbounded growth (text list + FAISS index expand together)

Storage model:
    - FAISS index: stores MLP-gated vectors o_t for inner-product search
    - Python list: stores text strings at matching indices
    - Both grow together: index[i] ↔ texts[i]
"""

import hashlib
import numpy as np
import torch
from typing import Dict, List, Optional, Set

# Graceful FAISS import — fall back to torch-based scoring if unavailable
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class HybridMemoryTranscript:
    """Persistent text store with FAISS inner-product vector index.

    Every (vector, text) pair is stored permanently. Retrieval uses
    FAISS IndexFlatIP for hardware-accelerated inner-product search
    over MLP-gated output vectors.

    Falls back to torch dot-product scoring if FAISS is not installed.

    Args:
        d_model:  Dimensionality of stored vectors.
        top_k:    Default number of entries to retrieve.
        device:   Torch device (vectors are converted to numpy for FAISS).
    """

    def __init__(self, d_model: int = 128, top_k: int = 3, device: torch.device = "cpu"):
        self.d_model = d_model
        self.top_k = top_k
        self.device = device
        self._count: int = 0
        self._texts: List[str] = []
        self._backend: str = "faiss" if _HAS_FAISS else "torch"

        # Deduplication: text hash → index in _texts / FAISS
        self._text_hashes: Dict[str, int] = {}

        if _HAS_FAISS:
            # IndexFlatIP = exact brute-force inner product (SIMD-accelerated)
            self._index = faiss.IndexFlatIP(d_model)
        else:
            # Fallback: store vectors as list of tensors
            self._vectors: List[torch.Tensor] = []

    @staticmethod
    def _hash_text(text: str) -> str:
        """Fast hash for deduplication."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def register(self, mlp_vector: torch.Tensor, text: str) -> None:
        """Store a (vector, text) pair. Skips if identical text already exists.

        If the same text was already registered, the old entry is kept
        (with its original vector). This prevents duplicate retrieval
        results and unbounded growth from repeated queries.

        Args:
            mlp_vector: (d_model,) — the output-gated MLP vector o_t.
            text:       The Q+A text associated with this turn.
        """
        text_hash = self._hash_text(text)

        # Skip duplicate text
        if text_hash in self._text_hashes:
            return

        vec = mlp_vector.detach().cpu().numpy().astype(np.float32)
        vec = vec.reshape(1, -1)  # (1, d_model) for FAISS

        if self._backend == "faiss":
            self._index.add(vec)
        else:
            self._vectors.append(mlp_vector.detach().cpu())

        self._text_hashes[text_hash] = self._count
        self._texts.append(text)
        self._count += 1

    def score_and_retrieve(
        self,
        query_vector: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """Search stored entries by inner product and return top-k texts.

        Uses FAISS IndexFlatIP for hardware-accelerated search, or
        falls back to torch dot-product if FAISS is unavailable.

        Args:
            query_vector: (d_model,) — the gated output from current turn.
            top_k:        Override default top_k.

        Returns:
            List of top-k text strings, ordered by descending score.
        """
        if self._count == 0:
            return []

        k = top_k if top_k is not None else self.top_k

        if self._backend == "faiss":
            # Request more than k to allow dedup filtering
            fetch_k = min(k * 2, self._count)
            query = query_vector.detach().cpu().numpy().astype(np.float32)
            query = query.reshape(1, -1)  # (1, d_model)
            scores, indices = self._index.search(query, fetch_k)
            results: List[str] = []
            seen: Set[str] = set()
            for idx in indices[0]:
                if idx < 0:
                    continue
                text = self._texts[idx]
                text_hash = self._hash_text(text)
                if text_hash not in seen:
                    seen.add(text_hash)
                    results.append(text)
                    if len(results) >= k:
                        break
            return results
        else:
            # Torch fallback with dedup
            query = query_vector.detach().squeeze().cpu()
            mat = torch.stack(self._vectors)  # (N, d_model)
            scores = torch.mv(mat, query)     # (N,)
            fetch_k = min(k * 2, self._count)
            _, top_idx = torch.topk(scores, fetch_k)
            results: List[str] = []
            seen: Set[str] = set()
            for i in top_idx.tolist():
                text = self._texts[i]
                text_hash = self._hash_text(text)
                if text_hash not in seen:
                    seen.add(text_hash)
                    results.append(text)
                    if len(results) >= k:
                        break
            return results

    def __len__(self) -> int:
        return self._count

    def state_dict(self) -> dict:
        """Serialize for checkpoint save.

        Stores vectors as a numpy array (compact) and texts as a list.
        The FAISS index is reconstructed from vectors on load.
        """
        if self._backend == "faiss" and self._count > 0:
            # Reconstruct vectors from FAISS index
            vectors = faiss.rev_swig_ptr(
                self._index.get_xb(), self._count * self.d_model
            ).reshape(self._count, self.d_model).copy().tolist()
        elif self._backend == "torch" and self._count > 0:
            vectors = [v.tolist() for v in self._vectors]
        else:
            vectors = []

        return {
            "backend": self._backend,
            "d_model": self.d_model,
            "count": self._count,
            "vectors": vectors,
            "texts": list(self._texts),
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore from checkpoint.

        Rebuilds FAISS index and dedup hashes from stored vectors/texts.
        """
        self.d_model = d["d_model"]
        self._count = d["count"]
        self._texts = list(d["texts"])

        # Rebuild dedup hashes
        self._text_hashes = {
            self._hash_text(t): i for i, t in enumerate(self._texts)
        }

        vectors = d["vectors"]

        if self._backend == "faiss":
            # Rebuild FAISS index
            self._index = faiss.IndexFlatIP(self.d_model)
            if vectors:
                mat = np.array(vectors, dtype=np.float32)
                self._index.add(mat)
        else:
            # Torch fallback
            self._vectors = [
                torch.tensor(v, dtype=torch.float32) for v in vectors
            ]

    def clear(self) -> None:
        """Remove all stored entries."""
        self._count = 0
        self._texts.clear()
        self._text_hashes.clear()
        if self._backend == "faiss":
            self._index = faiss.IndexFlatIP(self.d_model)
        else:
            self._vectors.clear()
