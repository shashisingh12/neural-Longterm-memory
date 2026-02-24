"""
Titans Memory — Outer-Loop Trainer
=====================================
Meta-learning trainer for the outer-loop parameters (W_K, W_V, W_Q)
and optionally the encoder projection head.

The training is episodic (like MAML / Reptile):

    For each episode:
        1. Reset MLP to fresh weights   (clean inner-loop slate)
        2. WRITE phase:  Process N texts through the inner loop
           - Each write returns a loss tensor (graph alive → W_K, W_V)
           - Inner-loop updates MLP weights via Titans rule
        3. EVAL phase:   Read back a subset of texts
           - Check how well M*(W_Q(x)) ≈ W_V(x) after the writes
           - This trains W_Q (and indirectly validates W_K, W_V)
        4. Sum write_losses + eval_losses → outer_loss
        5. outer_loss.backward()  →  grads flow to W_K, W_V, W_Q
        6. outer_optimizer.step()

This is a first-order approximation: we don't differentiate through
the inner-loop weight updates themselves (like FOMAML), but we do
differentiate through the K/V/Q projections that produced the loss.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .config import TitansConfig
from .text_encoder import TextEncoder
from .neural_memory import NeuralMemoryMLP
from .attention import MemoryAttention, PersistentMemoryVectors
from .utils import set_seed, get_device


class OuterLoopTrainer:
    """Trains the outer-loop parameters of the Titans memory system.

    Outer-loop parameters:
        - W_K, W_V, W_Q  (key/value/query projections in NeuralMemoryMLP)
        - Encoder projection head  (nn.Linear from BERT dim → d_model)

    Training data: a list of text strings. Each episode samples a
    subsequence and simulates the inner-loop write/read cycle.
    """

    def __init__(
        self,
        config: TitansConfig,
        encoder: TextEncoder,
        memory: NeuralMemoryMLP,
    ):
        self.config = config
        self.device = get_device(config.device)
        self.encoder = encoder
        self.memory = memory

        # Collect outer-loop parameters
        outer_params = list(memory.outer_loop_parameters())
        if config.train_encoder_projection:
            outer_params += list(encoder.projection.parameters())

        self.optimizer = torch.optim.AdamW(
            outer_params,
            lr=config.outer_lr,
            weight_decay=config.outer_weight_decay,
        )

        self.episode_len = config.episode_len
        self.eval_ratio = config.eval_ratio
        self.grad_clip = config.grad_clip
        self.epochs = config.outer_epochs
        self.verbose = config.verbose

    def load_training_data(self, path: str) -> List[str]:
        """Load training texts from a file (one text per line)."""
        with open(path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        if self.verbose:
            print(f"[Trainer] Loaded {len(texts)} training texts from {path}")
        return texts

    def train(self, texts: List[str]) -> List[float]:
        """Run the full outer-loop training.

        Args:
            texts: List of training strings.

        Returns:
            List of per-epoch average losses.
        """
        if len(texts) < self.episode_len:
            raise ValueError(
                f"Need at least {self.episode_len} texts, got {len(texts)}"
            )

        epoch_losses = []
        num_episodes = max(1, len(texts) // self.episode_len)

        for epoch in range(self.epochs):
            random.shuffle(texts)
            total_loss = 0.0
            n_episodes = 0

            for ep_start in range(0, len(texts) - self.episode_len + 1, self.episode_len):
                episode_texts = texts[ep_start: ep_start + self.episode_len]
                ep_loss = self._train_episode(episode_texts)
                total_loss += ep_loss
                n_episodes += 1

            avg_loss = total_loss / max(n_episodes, 1)
            epoch_losses.append(avg_loss)

            if self.verbose:
                print(f"[Trainer] Epoch {epoch + 1}/{self.epochs}  "
                      f"avg_loss={avg_loss:.6f}  episodes={n_episodes}")

        return epoch_losses

    def _train_episode(self, episode_texts: List[str]) -> float:
        """Run one training episode.

        Steps:
            1. Reset MLP to clean state
            2. Encode all texts
            3. Write phase: feed texts through inner loop (training=True)
            4. Eval phase: read back a subset, compute retrieval loss
            5. Backprop outer loss, step optimizer
        """
        # 1. Reset MLP (fresh inner-loop slate for this episode)
        self.memory.reset_mlp()

        # 2. Encode all episode texts
        with torch.no_grad():
            embeddings = self.encoder.encode(episode_texts)  # (N, d_model)

        # Split into write set and eval set
        n_eval = max(1, int(len(episode_texts) * self.eval_ratio))
        n_write = len(episode_texts) - n_eval

        write_embeds = embeddings[:n_write]   # (n_write, d_model)
        eval_embeds = embeddings[n_write:]    # (n_eval, d_model)

        # 3. Write phase: process each text through inner loop
        #    training=True keeps grad graph alive for W_K, W_V
        write_losses = []
        for i in range(n_write):
            x_t = write_embeds[i]  # (d_model,)
            loss_t = self.memory.write(x_t, training=True)
            write_losses.append(loss_t)

        # 4. Eval phase: read back eval texts and check recall quality
        #    This trains W_Q: how well does M*(W_Q(x)) ≈ W_V(x)?
        eval_losses = []
        for i in range(n_eval):
            x_t = eval_embeds[i]  # (d_model,)
            retrieved = self.memory.read(x_t, training=True)     # W_Q(x) → MLP
            target_v = self.memory.project_value(x_t)            # W_V(x)
            eval_loss = F.mse_loss(retrieved, target_v.detach())
            eval_losses.append(eval_loss)

        # 5. Combine losses and backprop
        outer_loss = sum(write_losses) + sum(eval_losses)

        self.optimizer.zero_grad()
        outer_loss.backward()

        # Gradient clipping
        if self.grad_clip > 0:
            all_params = list(self.memory.outer_loop_parameters())
            if self.config.train_encoder_projection:
                all_params += list(self.encoder.projection.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)

        self.optimizer.step()

        return outer_loss.item()


class ActiveOuterLoopTrainer:
    """Trains outer-loop parameters for the ActiveMemoryLayer.

    Extends the base trainer to include the full MAC pipeline:
        - Attention weights (W_attn_q, W_attn_k, W_attn_v, W_out)
        - Persistent vectors (p_1, ..., p_Np)
        - Memory projections (W_K, W_V, W_Q)
        - Encoder projection (optional)

    Training episode flow:
        1. Reset MLP to fresh weights
        2. Encode all episode texts
        3. WRITE phase: For each text x_t, run full MAC forward:
           h_t = read(x_t) → concat [P || h_t || x_t] → attention → write → gate
        4. EVAL phase: Same MAC forward, compare gated output against target
        5. Backprop outer_loss → optimizer.step()
    """

    def __init__(
        self,
        config: TitansConfig,
        encoder: TextEncoder,
        memory: NeuralMemoryMLP,
        attention: MemoryAttention,
        persistent_vectors: PersistentMemoryVectors,
    ):
        self.config = config
        self.device = get_device(config.device)
        self.encoder = encoder
        self.memory = memory
        self.attention = attention
        self.persistent_vectors = persistent_vectors

        # Collect all outer-loop parameters
        outer_params = list(memory.outer_loop_parameters())

        if config.train_attention:
            outer_params += list(attention.parameters())

        if config.train_persistent_vectors:
            outer_params += list(persistent_vectors.parameters())

        if config.train_encoder_projection:
            outer_params += list(encoder.projection.parameters())

        self._all_outer_params = outer_params

        self.optimizer = torch.optim.AdamW(
            outer_params,
            lr=config.outer_lr,
            weight_decay=config.outer_weight_decay,
        )

        self.episode_len = config.episode_len
        self.eval_ratio = config.eval_ratio
        self.grad_clip = config.grad_clip
        self.epochs = config.outer_epochs
        self.verbose = config.verbose

    def train(self, texts: List[str]) -> List[float]:
        """Run the full outer-loop training.

        Args:
            texts: List of training strings.

        Returns:
            List of per-epoch average losses.
        """
        if len(texts) < self.episode_len:
            raise ValueError(
                f"Need at least {self.episode_len} texts, got {len(texts)}"
            )

        epoch_losses = []

        for epoch in range(self.epochs):
            random.shuffle(texts)
            total_loss = 0.0
            n_episodes = 0

            for ep_start in range(0, len(texts) - self.episode_len + 1,
                                  self.episode_len):
                episode_texts = texts[ep_start: ep_start + self.episode_len]
                ep_loss = self._train_episode(episode_texts)
                total_loss += ep_loss
                n_episodes += 1

            avg_loss = total_loss / max(n_episodes, 1)
            epoch_losses.append(avg_loss)

            if self.verbose:
                print(f"[ActiveTrainer] Epoch {epoch + 1}/{self.epochs}  "
                      f"avg_loss={avg_loss:.6f}  episodes={n_episodes}")

        return epoch_losses

    def _mac_forward_training(
        self, x_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one MAC forward pass in training mode (graph alive).

        Args:
            x_t: (d_model,) — encoded text embedding.

        Returns:
            write_loss: tensor from memory.write(y_t, training=True)
            o_t:        (d_model,) gated output vector
        """
        x_seq = x_t.unsqueeze(0).unsqueeze(0)               # (1, 1, d_model)

        # Memory read (Eq. 21) — training=True keeps W_Q in graph
        h_t = self.memory.read(x_t, training=True)           # (d_model,)
        h_seq = h_t.unsqueeze(0).unsqueeze(0)                # (1, 1, d_model)

        # Concatenate three branches (Eq. 22)
        p_vecs = self.persistent_vectors(batch_size=1)       # (1, N_p, d_model)
        s_tilde = torch.cat([p_vecs, h_seq, x_seq], dim=1)

        # Self-attention (Eq. 23)
        y_full = self.attention(s_tilde)                     # (1, N_p+2, d_model)
        y_t = y_full[0, -1, :]                               # (d_model,)

        # Write to memory (Eq. 24) — training=True keeps W_K/W_V in graph
        write_loss = self.memory.write(y_t, training=True)

        # Output gating (Eq. 25) — training=True keeps W_Q in graph
        mem_gate = self.memory.read(y_t, training=True)
        o_t = y_t * mem_gate                                 # (d_model,)

        return write_loss, o_t

    def _train_episode(self, episode_texts: List[str]) -> float:
        """Run one training episode with full MAC pipeline.

        Steps:
            1. Reset MLP
            2. Encode all texts
            3. Write phase: MAC forward for each text
            4. Eval phase: MAC forward + compare gated output vs target value
            5. Backprop, clip, step
        """
        # 1. Reset MLP (fresh inner-loop slate)
        self.memory.reset_mlp()

        # 2. Encode all episode texts
        with torch.no_grad():
            embeddings = self.encoder.encode(episode_texts)  # (N, d_model)

        # Split into write and eval sets
        n_eval = max(1, int(len(episode_texts) * self.eval_ratio))
        n_write = len(episode_texts) - n_eval

        write_embeds = embeddings[:n_write]
        eval_embeds = embeddings[n_write:]

        # 3. Write phase: full MAC forward for each text
        write_losses = []
        for i in range(n_write):
            x_t = write_embeds[i]
            write_loss, _ = self._mac_forward_training(x_t)
            write_losses.append(write_loss)

        # 4. Eval phase: MAC forward + retrieval quality check
        eval_losses = []
        for i in range(n_eval):
            x_t = eval_embeds[i]
            _, o_t = self._mac_forward_training(x_t)
            # How well does the gated output approximate the target value?
            target_v = self.memory.project_value(x_t)
            eval_loss = F.mse_loss(o_t, target_v.detach())
            eval_losses.append(eval_loss)

        # 5. Backprop
        outer_loss = sum(write_losses) + sum(eval_losses)

        self.optimizer.zero_grad()
        outer_loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self._all_outer_params, self.grad_clip
            )

        self.optimizer.step()

        return outer_loss.item()
