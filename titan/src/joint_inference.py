"""
Phase 2 — Inference with Surprise-Only Updates
=================================================
All weights frozen after Phase 1.

Per turn:
  1. M_t state conditions token embeddings via frozen adapter
  2. Generate response (frozen LLM + LoRA + adapter)
  3. Compute surprise for this turn
  4. alpha_t = alpha_net(surprise) — dynamic, not fixed
  5. M_t.surprise_write(k, v) — M_t weights update
  6. Repeat

The only things that change per turn:
  M_t.W[0], M_t.W[1]   — weight matrices (via surprise write)
  M_t momentum buffers   — internal state
  alpha_t                — per-turn, computed fresh each turn from surprise
"""

import torch
from typing import List

from .memory_llm import MemoryLLM
from .simple_encoder import TiktokenEncoder
from .joint_config import JointTrainingConfig


class Phase2Inference:
    """
    Inference-time memory system.

    After Phase 1 joint training, all weights are frozen.
    Each conversation turn:
      1. Generate response using current M_t state
      2. Compute surprise for (Q+A) → alpha_t
      3. Write into M_t with alpha_t strength
      4. M_t accumulates user-specific information

    alpha_t is dynamic:
      "My name is Alice"       → high surprise → strong write
      "What is 2+2?"           → low surprise  → weak write
      "I have diabetes"        → very high surprise → very strong write

    Critical facts persist. Trivia fades.
    """

    def __init__(self, model: MemoryLLM, config: JointTrainingConfig):
        self.model = model
        self.config = config
        self.encoder = TiktokenEncoder(
            d_model=config.d_mem,
            seed=config.encoder_seed,
        )
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(config.torch_dtype, torch.float32)
        self.turn = 0
        self.alpha_history: List[float] = []
        self.surprise_history: List[float] = []

    def chat(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> str:
        """
        Generate a response with memory-conditioned LLM.

        Args:
            prompt: user input text
            max_new_tokens: max tokens to generate
            temperature: sampling temperature (0 = greedy)
            verbose: print surprise/alpha diagnostics

        Returns:
            Generated response string.
        """
        self.turn += 1
        device = self.model.device

        # generate with current M_t state
        # M_t.W conditions token embeddings via frozen adapter hook
        inputs = self.model.tokenizer(
            f"Q: {prompt}\nA:",
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            output_ids = self.model.base_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.model.tokenizer.eos_token_id,
            )

        # extract only the newly generated tokens
        new_ids = output_ids[0][input_ids.shape[1]:]
        response = self.model.tokenizer.decode(
            new_ids, skip_special_tokens=True
        )

        # compute surprise and write to M_t
        combined = f"Q:{prompt} A:{response}"
        k = self.encoder.encode_tensor(combined, device=str(device), dtype=self.dtype)
        v = self.encoder.encode_tensor(response, device=str(device), dtype=self.dtype)

        # surprise write — M_t updates, alpha_t computed dynamically
        alpha_t, surprise, loss_inner = (
            self.model.memory.surprise_write(k, v)
        )

        self.alpha_history.append(alpha_t)
        self.surprise_history.append(surprise)

        if verbose or self.config.verbose:
            strength = "HIGH" if alpha_t > 0.1 else "low"
            detail = ("written strongly" if alpha_t > 0.1
                      else "written weakly")
            print(
                f"  Turn {self.turn}: "
                f"surprise={surprise:.4f}  "
                f"alpha_t={alpha_t:.4f}  "
                f"{strength} — {detail}"
            )

        return response

    def reset(self):
        """Start a new conversation — reset M_t to blank slate."""
        self.model.memory.reset()
        self.turn = 0
        self.alpha_history = []
        self.surprise_history = []

    def memory_stats(self) -> dict:
        """Return diagnostic info about the current memory state."""
        return {
            "turns": self.turn,
            "alpha_history": list(self.alpha_history),
            "surprise_history": list(self.surprise_history),
            "alpha_mean": (
                sum(self.alpha_history) / len(self.alpha_history)
                if self.alpha_history else 0.0
            ),
            "surprise_mean": (
                sum(self.surprise_history) / len(self.surprise_history)
                if self.surprise_history else 0.0
            ),
        }
