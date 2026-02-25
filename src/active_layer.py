"""
Titans Memory — Active Memory Layer
=====================================
Actively learning memory system using full MAC integration (Paper §4.1).
Replaces RAG-based retrieval with MLP-driven context selection.

Per-Turn Flow (MAC Eq. 21-25 + ring buffer):
    1. s_t = encoder.encode(prompt)              — BERT → d_model
    2. h_t = memory.read(s_t)                    — MLP recall (Eq. 21)
    3. S̃ = [persistent || h_t || s_t]           — three branches (Eq. 22)
    4. y_t = Attention(S̃)                       — self-attention (Eq. 23)
    5. memory.write(y_t)                         — update MLP weights (Eq. 24)
    6. o_t = y_t ⊙ memory.read(y_t)            — output gating (Eq. 25)
    7. transcript.score_and_retrieve(o_t)        — MLP-scored retrieval
    8. Build prompt → call LLM → register in ring buffer

Key difference from TitansMemoryLayer:
    - MLP memory actively drives context via attention + gating
    - No cosine RAG store
    - FAISS-backed hybrid transcript (persistent) or ring buffer (bounded)

Key difference from MACMemoryLayer:
    - Hybrid transcript or ring buffer replaces TextDecoder
    - FAISS inner-product search over MLP-gated vectors
    - Full training support via ActiveOuterLoopTrainer
"""

import time
from typing import Callable, List

import torch

from .config import TitansConfig
from .text_encoder import TextEncoder
from .neural_memory import NeuralMemoryMLP
from .persistent_memory import PersistentMemory
from .attention import MemoryAttention, PersistentMemoryVectors
from .memory_transcript import MemoryTranscript
from .hybrid_transcript import HybridMemoryTranscript
from .utils import get_device


class ActiveMemoryLayer:
    """Actively learning memory — MLP is the primary context provider.

    Usage:
        config = TitansConfig(d_model=128, num_persistent_vectors=4)
        layer  = ActiveMemoryLayer(config)

        def my_llm(prompt: str) -> str:
            return your_model.generate(prompt)

        response = layer.run("What is Python?", my_llm)
    """

    def __init__(self, config: TitansConfig):
        self.config = config
        self.device = get_device(config.device)
        self.verbose = config.verbose
        self._turn: int = 0

        # Text encoder: text → (d_model,) vectors
        self.encoder = TextEncoder(config).to(self.device)

        # Neural memory MLP (inner-loop, weights = memory)
        self.memory = NeuralMemoryMLP(config).to(self.device)

        # Learnable persistent memory vectors P = [p1, ..., p_Np]
        self.persistent_vectors = PersistentMemoryVectors(
            num_vectors=config.num_persistent_vectors,
            d_model=config.d_model,
        ).to(self.device)

        # Self-attention module (Eq. 23)
        self.attention = MemoryAttention(config).to(self.device)

        # Memory transcript: FAISS hybrid (persistent) or ring buffer (fixed-size)
        if config.use_hybrid_transcript:
            self.transcript = HybridMemoryTranscript(
                d_model=config.d_model,
                top_k=config.top_k,
                device=self.device,
            )
        else:
            self.transcript = MemoryTranscript(
                max_size=config.memory_transcript_size,
                top_k=config.top_k,
                device=self.device,
            )

        # Text-based persistent memory (for prompt building)
        self.persistent = PersistentMemory(
            tokens=config.persistent_tokens if config.persistent_tokens else None,
            sos_token=config.sos_token,
            eos_token=config.eos_token,
            max_length=config.persistent_max_length,
        )

    def run(self, prompt: str, llm_fn: Callable[[str], str]) -> str:
        """Execute one full active-memory turn.

        Implements Eq. 21-25 with ring-buffer retrieval:
            1. Encode prompt → s_t
            2. Read memory: h_t = M*(q_t)                     (Eq. 21)
            3. Concat: [persistent || h_t || s_t]              (Eq. 22)
            4. Attention: y_t = Attn(S̃)                       (Eq. 23)
            5. Update memory: write(y_t)                       (Eq. 24)
            6. Output gate: o_t = y_t ⊙ M*(y_t)              (Eq. 25)
            7. Score ring buffer with o_t → retrieve context
            8. Build prompt, call LLM, register Q+A
        """
        self._turn += 1
        t0 = time.time()

        # ── Step 1: Encode prompt → s_t ──
        s_t = self.encoder.encode(prompt)                    # (d_model,)
        s_t_seq = s_t.unsqueeze(0).unsqueeze(0)             # (1, 1, d_model)

        # ── Step 2: Memory read (Eq. 21) ──
        h_t = self.memory.read(s_t)                          # (d_model,)
        h_t_seq = h_t.unsqueeze(0).unsqueeze(0)             # (1, 1, d_model)

        # ── Step 3: Concatenate three branches (Eq. 22) ──
        p_vecs = self.persistent_vectors(batch_size=1)       # (1, N_p, d_model)
        s_tilde = torch.cat([p_vecs, h_t_seq, s_t_seq], dim=1)

        if self.verbose:
            N_p = self.config.num_persistent_vectors
            print(f"\n[Active | Turn {self._turn}]")
            print(f"  Sequence: {N_p} persistent + 1 memory_read + 1 current = {s_tilde.shape[1]} tokens")

        # ── Step 4: Self-attention (Eq. 23) ──
        y_t_full = self.attention(s_tilde)                   # (1, N_p+2, d_model)
        y_t = y_t_full[0, -1, :]                             # (d_model,)

        # ── Step 5: Update memory (Eq. 24) ──
        surprise = self.memory.write(y_t.detach())

        # ── Step 6: Output gating (Eq. 25) ──
        mem_gate = self.memory.read(y_t.detach())
        o_t = y_t * mem_gate                                 # (d_model,)

        # ── Step 7: Score ring buffer with MLP-gated output ──
        retrieved_texts = self.transcript.score_and_retrieve(
            o_t.detach(), self.config.top_k
        )

        if self.verbose:
            print(f"  Retrieved {len(retrieved_texts)} snippet(s) from transcript")
            for i, t in enumerate(retrieved_texts, 1):
                print(f"    [{i}] {t[:90]}")

        # ── Step 8: Build prompt, call LLM, register ──
        enriched = self._build_prompt(prompt, retrieved_texts)

        if self.verbose:
            extra = len(enriched) - len(prompt)
            print(f"  Enriched prompt: +{extra} chars added")

        response = llm_fn(enriched)

        # Register Q+A with MLP-gated vector in ring buffer
        combined = f"Q: {prompt}\nA: {response}"
        self.transcript.register(o_t.detach(), combined)

        if self.verbose:
            elapsed = time.time() - t0
            cap = self.config.memory_transcript_size if not self.config.use_hybrid_transcript else "unbounded"
            print(f"  Transcript: {len(self.transcript)}/{cap}")
            print(f"  Surprise: {surprise:.4f} | elapsed: {elapsed:.3f}s")

        return response

    def _build_prompt(self, user_prompt: str, memories: List[str]) -> str:
        """Assemble enriched prompt from persistent text + active memory + user input."""
        parts: List[str] = []

        p = self.persistent.render()
        if p:
            parts.append(p)

        if memories:
            parts.append("=== ACTIVE MEMORY CONTEXT ===")
            for i, m in enumerate(memories, 1):
                parts.append(f"[Memory {i}]\n{m}")
            parts.append("=== END MEMORY ===")

        parts.append(f"User: {user_prompt}")
        return "\n\n".join(parts)

    # ── Save / Load ──────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save full active memory state."""
        state = {
            "config": self.config.to_dict(),
            "memory_mlp": self.memory.state_dict(),
            "encoder_projection": self.encoder.projection.state_dict(),
            "attention": self.attention.state_dict(),
            "persistent_vectors": self.persistent_vectors.state_dict(),
            "transcript": self.transcript.state_dict(),
            "persistent_text": self.persistent.state_dict(),
            "turn": self._turn,
        }
        torch.save(state, path)
        if self.verbose:
            print(f"[Active] Saved → {path}")

    def load(self, path: str) -> None:
        """Load active memory state."""
        state = torch.load(path, map_location=str(self.device), weights_only=False)
        self.memory.load_state_dict(state["memory_mlp"])
        self.encoder.projection.load_state_dict(state["encoder_projection"])
        self.attention.load_state_dict(state["attention"])
        self.persistent_vectors.load_state_dict(state["persistent_vectors"])
        self.transcript.load_state_dict(state["transcript"])
        self.persistent.load_state_dict(state["persistent_text"])
        self._turn = state.get("turn", 0)
        if self.verbose:
            print(f"[Active] Loaded ← {path}  (turn {self._turn})")

    # ── Diagnostics ──────────────────────────────────────────

    def stats(self) -> dict:
        """Return diagnostic statistics."""
        mem_params = sum(p.numel() for p in self.memory.parameters())
        attn_params = sum(p.numel() for p in self.attention.parameters())
        persist_params = sum(p.numel() for p in self.persistent_vectors.parameters())
        enc_proj_params = sum(p.numel() for p in self.encoder.projection.parameters())
        is_hybrid = self.config.use_hybrid_transcript
        return {
            "architecture": "Active Memory (MAC + FAISS Hybrid)" if is_hybrid else "Active Memory (MAC + Ring Buffer)",
            "turns_seen": self._turn,
            "transcript_entries": len(self.transcript),
            "transcript_backend": "faiss" if is_hybrid else "ring_buffer",
            "transcript_capacity": "unbounded" if is_hybrid else self.config.memory_transcript_size,
            "memory_mlp_params": int(mem_params),
            "attention_params": int(attn_params),
            "persistent_vector_params": int(persist_params),
            "encoder_projection_params": int(enc_proj_params),
            "d_model": self.config.d_model,
            "mlp_layers": self.config.memory_num_layers,
            "attention_heads": self.config.num_attention_heads,
            "persistent_vectors": self.config.num_persistent_vectors,
            "device": str(self.device),
        }
