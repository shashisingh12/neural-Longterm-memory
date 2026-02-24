"""
Titans Memory — MAC (Memory as Context) Architecture
======================================================
Paper §4.1, Figure 2.

Three branches feed into self-attention:
    1. Persistent memory   P = [p1, ..., p_Np]  (learnable, input-independent)
    2. Long-term memory    h_t = M*_{t-1}(q_t)  (neural MLP read)
    3. Current input       S^(t)                 (encoded prompt)

Flow per turn (Eq. 21-25):
    q_t = S^(t) @ W_Q                                 (21) query projection
    h_t = M*_{t-1}(q_t)                               (21) memory read
    S̃^(t) = [p1,...,p_Np || h_t || S^(t)]            (22) concatenate
    y_t = Attn(S̃^(t))                                (23) self-attention
    M_t = M_{t-1}(y_t)                                (24) memory write (update)
    o_t = y_t ⊙ M*_t(y_t)                            (25) output gating

Key properties at test time:
    (i)   Persistent vectors are FIXED (task knowledge)
    (ii)  Attention weights are IN-CONTEXT learners
    (iii) MLP memory continues LEARNING (inner-loop Titans rule)
"""

import time
from typing import Callable, List

import torch
import torch.nn.functional as F

from .config import TitansConfig
from .text_encoder import TextEncoder
from .neural_memory import NeuralMemoryMLP
from .text_decoder import TextDecoder
from .persistent_memory import PersistentMemory
from .attention import MemoryAttention, PersistentMemoryVectors
from .checkpoint import save_checkpoint, load_checkpoint, restore_from_state
from .utils import get_device


class MACMemoryLayer:
    """Memory-as-Context (MAC) architecture wrapper for any LLM.

    Implements the full MAC flow from Paper §4.1:
    - Three-branch input: persistent + memory read + current
    - Self-attention decides relevance across branches
    - Memory updated with attention output (not raw input)
    - Output gating: y_t ⊙ M*(y_t)

    Usage:
        config = TitansConfig(d_model=128, num_persistent_vectors=4)
        layer  = MACMemoryLayer(config)

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

        # Text decoder (nearest-neighbor retrieval for prompt building)
        self.decoder = TextDecoder(
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
            device=self.device,
        )

        # Text-based persistent memory (for prompt building — separate from vectors)
        self.persistent = PersistentMemory(
            tokens=config.persistent_tokens if config.persistent_tokens else None,
            sos_token=config.sos_token,
            eos_token=config.eos_token,
            max_length=config.persistent_max_length,
        )

    def run(self, prompt: str, llm_fn: Callable[[str], str]) -> str:
        """Execute one full MAC turn.

        Implements Eq. 21-25:
            1. Encode prompt → S^(t)
            2. Read memory: h_t = M*(q_t)                     (Eq. 21)
            3. Concat: [persistent || h_t || S^(t)]            (Eq. 22)
            4. Attention: y_t = Attn(S̃)                       (Eq. 23)
            5. Update memory: M_t via write(y_t)               (Eq. 24)
            6. Output gate: o_t = y_t ⊙ M*(y_t)              (Eq. 25)
            7. Use o_t for retrieval, build prompt, call LLM
        """
        self._turn += 1
        t0 = time.time()

        # ── Step 1: Encode prompt → S^(t) ──
        s_t = self.encoder.encode(prompt)  # (d_model,)
        s_t_seq = s_t.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model) — batch=1, seq=1

        # ── Step 2: Memory read (Eq. 21) ──
        # h_t = M*_{t-1}(q_t) where q_t = S^(t) @ W_Q
        h_t = self.memory.read(s_t)  # (d_model,)
        h_t_seq = h_t.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)

        # ── Step 3: Concatenate three branches (Eq. 22) ──
        # S̃^(t) = [p1,...,p_Np || h_t || S^(t)]
        p_vecs = self.persistent_vectors(batch_size=1)  # (1, N_p, d_model)
        s_tilde = torch.cat([p_vecs, h_t_seq, s_t_seq], dim=1)  # (1, N_p+2, d_model)

        if self.verbose:
            N_p = self.config.num_persistent_vectors
            print(f"\n[MAC | Turn {self._turn}]")
            print(f"  Sequence: {N_p} persistent + 1 memory_read + 1 current = {s_tilde.shape[1]} tokens")

        # ── Step 4: Self-attention (Eq. 23) ──
        y_t_full = self.attention(s_tilde)  # (1, N_p+2, d_model)

        # Extract the current-token output (last position)
        y_t = y_t_full[0, -1, :]  # (d_model,) — output for S^(t) position

        # ── Step 5: Update memory (Eq. 24) ──
        # M_t = M_{t-1}(y_t) — write attention output into memory
        surprise = self.memory.write(y_t.detach())

        # ── Step 6: Output gating (Eq. 25) ──
        # o_t = y_t ⊙ M*_t(y_t)  — element-wise product
        mem_gate = self.memory.read(y_t.detach())  # M*_t(y_t) after update
        o_t = y_t * mem_gate  # (d_model,) — gated output

        # ── Step 7: Use gated output for text retrieval + prompt building ──
        # Query the text decoder with the gated output
        retrieved_texts = self.decoder.decode(o_t.detach(), self.config.top_k)

        if self.verbose:
            print(f"  Retrieved {len(retrieved_texts)} snippet(s)")
            for i, t in enumerate(retrieved_texts, 1):
                print(f"    [{i}] {t[:90]}")

        enriched = self._build_prompt(prompt, retrieved_texts)

        if self.verbose:
            extra = len(enriched) - len(prompt)
            print(f"  Enriched prompt: +{extra} chars added")

        # Call the LLM
        response = llm_fn(enriched)

        # Register in text decoder for future retrieval
        combined = f"Q: {prompt}\nA: {response}"
        combined_vec = self.encoder.encode(combined)
        self.decoder.register(combined_vec.detach(), combined)

        if self.verbose:
            elapsed = time.time() - t0
            print(f"  Surprise: {surprise:.4f} | elapsed: {elapsed:.3f}s")

        return response

    def _build_prompt(self, user_prompt: str, memories: List[str]) -> str:
        """Assemble enriched prompt from persistent text + memories + user input."""
        parts: List[str] = []

        p = self.persistent.render()
        if p:
            parts.append(p)

        if memories:
            parts.append("=== RELEVANT LONG-TERM MEMORY ===")
            for i, m in enumerate(memories, 1):
                parts.append(f"[Memory {i}]\n{m}")
            parts.append("=== END MEMORY ===")

        parts.append(f"User: {user_prompt}")
        return "\n\n".join(parts)

    # ── Save / Load ──────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save full MAC system state."""
        import torch as _torch
        state = {
            "config": self.config.to_dict(),
            "memory_mlp": self.memory.state_dict(),
            "encoder_projection": self.encoder.projection.state_dict(),
            "attention": self.attention.state_dict(),
            "persistent_vectors": self.persistent_vectors.state_dict(),
            "decoder_store": self.decoder.state_dict(),
            "persistent_text": self.persistent.state_dict(),
            "turn": self._turn,
        }
        _torch.save(state, path)
        if self.verbose:
            print(f"[MAC] Saved → {path}")

    def load(self, path: str) -> None:
        """Load MAC system state."""
        import torch as _torch
        state = _torch.load(path, map_location=str(self.device), weights_only=False)
        self.memory.load_state_dict(state["memory_mlp"])
        self.encoder.projection.load_state_dict(state["encoder_projection"])
        self.attention.load_state_dict(state["attention"])
        self.persistent_vectors.load_state_dict(state["persistent_vectors"])
        self.decoder.load_state_dict(state["decoder_store"])
        self.persistent.load_state_dict(state["persistent_text"])
        self._turn = state.get("turn", 0)
        if self.verbose:
            print(f"[MAC] Loaded ← {path}  (turn {self._turn})")

    # ── Diagnostics ──────────────────────────────────────────

    def stats(self) -> dict:
        """Return diagnostic statistics."""
        mem_params = sum(p.numel() for p in self.memory.parameters())
        attn_params = sum(p.numel() for p in self.attention.parameters())
        persist_params = sum(p.numel() for p in self.persistent_vectors.parameters())
        enc_proj_params = sum(p.numel() for p in self.encoder.projection.parameters())
        return {
            "architecture": "MAC (Memory as Context)",
            "turns_seen": self._turn,
            "memory_entries": len(self.decoder),
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
