"""
Titans Memory — Memory Layer (Main Orchestrator)
==================================================
Wraps ANY existing LLM / agent with external neural long-term memory.

Per-Turn Flow:
    1. encode(prompt)           → input embedding x_t
    2. memory.read(x_t)         → projects x→q via W_Q, retrieves from MLP (no update)
    3. decoder.decode(vec)      → top-k past text snippets
    4. build enriched prompt:
            [persistent tokens with <SOS>/<EOS>]
            [retrieved memory snippets]
            [current user prompt]
    5. call your LLM with enriched prompt
    6. encode("Q:...A:...") → x_t, memory.write(x_t)
            internally: k_t = x_t @ W_K,  v_t = x_t @ W_V
            loss = ||M(k_t) - v_t||², update MLP via Titans rule
"""

import time
from typing import Callable, List

import torch

from .config import TitansConfig
from .text_encoder import TextEncoder
from .neural_memory import NeuralMemoryMLP
from .text_decoder import TextDecoder
from .persistent_memory import PersistentMemory
from .checkpoint import save_checkpoint, load_checkpoint, restore_from_state
from .utils import get_device


class TitansMemoryLayer:
    """Wraps any LLM callable with Titans-inspired external memory.

    Usage:
        config = TitansConfig(d_model=128, verbose=True)
        layer  = TitansMemoryLayer(config)
        layer.persistent.add("You are a helpful assistant.")

        def my_llm(prompt: str) -> str:
            return your_model.generate(prompt)

        response = layer.run("What is Python?", my_llm)
    """

    def __init__(self, config: TitansConfig):
        self.config = config
        self.device = get_device(config.device)
        self.verbose = config.verbose
        self._turn: int = 0

        # Sub-modules
        self.encoder = TextEncoder(config).to(self.device)
        self.memory = NeuralMemoryMLP(config).to(self.device)
        self.decoder = TextDecoder(
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
            device=self.device,
        )
        self.persistent = PersistentMemory(
            tokens=config.persistent_tokens if config.persistent_tokens else None,
            sos_token=config.sos_token,
            eos_token=config.eos_token,
            max_length=config.persistent_max_length,
        )

    def run(self, prompt: str, llm_fn: Callable[[str], str]) -> str:
        """Execute one full memory-augmented turn.

        Args:
            prompt:  Raw user input.
            llm_fn:  Any callable  str → str  (your LLM / agent).

        Returns:
            LLM response string.
        """
        self._turn += 1
        t0 = time.time()

        # 1. Encode prompt → input embedding x_t
        x_vec = self.encoder.encode(prompt)  # (d_model,)

        # 2. Read from neural memory (projects x → q via W_Q, then MLP forward)
        #    This gives a fuzzy associative recall from the MLP weight-memory.
        retrieved_vec = self.memory.read(x_vec)  # (d_model,)

        # 3. Decode: nearest-neighbour cosine lookup in raw embedding space.
        #    The decoder stores raw BERT embeddings, so we query with x_vec
        #    (not the MLP output which lives in a different projected space).
        retrieved_texts = self.decoder.decode(x_vec, self.config.top_k)

        if self.verbose:
            print(f"\n[Memory | Turn {self._turn}]")
            print(f"  Retrieved {len(retrieved_texts)} snippet(s)")
            for i, t in enumerate(retrieved_texts, 1):
                print(f"    [{i}] {t[:90]}")

        # 4. Build enriched prompt
        enriched = self._build_prompt(prompt, retrieved_texts)

        if self.verbose:
            extra = len(enriched) - len(prompt)
            print(f"  Enriched prompt: +{extra} chars added")

        # 5. Call the LLM
        response = llm_fn(enriched)

        # 6. Write into memory
        # Encode the combined Q+A as x_t; the memory module internally
        # projects x_t → k_t via W_K and x_t → v_t via W_V (Eq. 11)
        combined = f"Q: {prompt}\nA: {response}"
        x_write = self.encoder.encode(combined)  # (d_model,)
        surprise = self.memory.write(x_write)
        self.decoder.register(x_write.detach(), combined)

        if self.verbose:
            elapsed = time.time() - t0
            print(f"  Surprise: {surprise:.4f} | elapsed: {elapsed:.3f}s")

        return response

    def _build_prompt(self, user_prompt: str, memories: List[str]) -> str:
        """Assemble the enriched prompt from persistent context + memories + user input."""
        parts: List[str] = []

        # Persistent context (with <SOS>/<EOS> wrapping)
        p = self.persistent.render()
        if p:
            parts.append(p)

        # Retrieved memory snippets
        if memories:
            parts.append("=== RELEVANT LONG-TERM MEMORY ===")
            for i, m in enumerate(memories, 1):
                parts.append(f"[Memory {i}]\n{m}")
            parts.append("=== END MEMORY ===")

        parts.append(f"User: {user_prompt}")
        return "\n\n".join(parts)

    # ── Batch / Parallel Processing ─────────────────────────

    def run_batch(
        self,
        prompts: List[str],
        llm_fn: Callable[[str], str],
    ) -> List[str]:
        """Process multiple prompts, using parallel chunk writes if enabled.

        When config.use_parallel_memory is True, all prompt embeddings are
        written to memory in parallel chunks (§3.2) instead of one-by-one.

        Args:
            prompts: List of user prompts.
            llm_fn:  LLM callable str → str.

        Returns:
            List of LLM response strings.
        """
        responses = []

        if self.config.use_parallel_memory and len(prompts) > 1:
            # Encode all prompts first
            embeddings = []
            for p in prompts:
                embeddings.append(self.encoder.encode(p))
            x_stack = torch.stack(embeddings, dim=0)  # (N, d_model)

            # Read from memory for each prompt (reads are independent)
            for i, prompt in enumerate(prompts):
                self._turn += 1
                x_vec = embeddings[i]
                retrieved_texts = self.decoder.decode(x_vec, self.config.top_k)
                enriched = self._build_prompt(prompt, retrieved_texts)
                response = llm_fn(enriched)
                responses.append(response)

                # Register Q+A in decoder for future retrieval
                combined = f"Q: {prompt}\nA: {response}"
                combined_vec = self.encoder.encode(combined)
                self.decoder.register(combined_vec.detach(), combined)

            # Parallel chunk write: write all combined Q+A embeddings at once
            write_embeddings = []
            for prompt, response in zip(prompts, responses):
                combined = f"Q: {prompt}\nA: {response}"
                write_embeddings.append(self.encoder.encode(combined))
            write_stack = torch.stack(write_embeddings, dim=0)  # (N, d_model)

            losses = self.memory.write_chunk_parallel(
                write_stack, chunk_size=self.config.chunk_size
            )

            if self.verbose:
                avg_loss = sum(losses) / len(losses)
                print(f"[Parallel Write] {len(prompts)} tokens, "
                      f"chunk_size={self.config.chunk_size}, "
                      f"avg surprise={avg_loss:.4f}")
        else:
            # Fall back to sequential processing
            for prompt in prompts:
                responses.append(self.run(prompt, llm_fn))

        return responses

    # ── Save / Load ──────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save full system state to a .pt checkpoint."""
        save_checkpoint(path, self.config, self)
        if self.verbose:
            print(f"[Memory] Saved → {path}")

    def load(self, path: str) -> None:
        """Load system state from a .pt checkpoint."""
        _, state = load_checkpoint(path, map_location=str(self.device))
        restore_from_state(self, state)
        if self.verbose:
            print(f"[Memory] Loaded ← {path}  (turn {self._turn})")

    # ── Diagnostics ──────────────────────────────────────────

    def stats(self) -> dict:
        """Return diagnostic statistics about the memory system."""
        total_params = sum(p.numel() for p in self.memory.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.projection.parameters())
        return {
            "turns_seen": self._turn,
            "memory_entries": len(self.decoder),
            "memory_mlp_params": int(total_params),
            "encoder_projection_params": int(encoder_params),
            "d_model": self.config.d_model,
            "mlp_layers": self.config.memory_num_layers,
            "device": str(self.device),
        }
