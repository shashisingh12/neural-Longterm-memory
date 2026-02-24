"""
Titans Memory — Neural Memory MLP
====================================
An MLP whose weights ARE the long-term memory.

Two-Loop Architecture:
    OUTER LOOP (training):  Learns W_K, W_V, W_Q via standard optimizer
    INNER LOOP (on-the-go): Updates MLP weights via Titans surprise rule

Paper Eq. 11 — Learned K/V/Q projections:
    k_t = x_t @ W_K
    v_t = x_t @ W_V
    q_t = x_t @ W_Q

Paper Eq. 12 — Associative memory loss:
    loss = || M_{t-1}(k_t) - v_t ||^2

Paper Section 3.1 — Inner-loop update rule:
    S_t = eta_t * S_{t-1} - theta_t * grad(loss)     (momentum over surprise)
    M_t = (1 - alpha_t) * M_{t-1} + S_t              (forgetting gate)

READ  (Eq. 15):  y_t = M*(q_t)  — inference, no weight update
WRITE:            project x_t → (k_t, v_t), then one step of the update rule
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
from .config import TitansConfig
from .parallel_memory import (
    ChunkedMLPMemoryUpdate,
    parallel_associative_scan,
    compute_cumulative_decay,
    compute_beta_ratios,
)


class NeuralMemoryMLP(nn.Module):
    """MLP whose weights serve as compressed long-term memory.

    Contains three learned projection matrices (Eq. 11):
        W_K: projects input x_t → key   k_t  (for writing)
        W_V: projects input x_t → value v_t  (for writing)
        W_Q: projects input x_t → query q_t  (for reading)

    These projections are outer-loop parameters trained with a standard
    optimizer. The MLP layers themselves are inner-loop memory updated
    by the Titans rule at inference time.
    """

    def __init__(self, config: TitansConfig):
        super().__init__()

        self.d_model = config.d_model
        self.num_layers = config.memory_num_layers
        hidden_dim = config.memory_hidden_dim
        self.lr = config.memory_lr
        self.momentum_decay = config.memory_momentum_decay
        self.forget_gate = config.memory_forget_gate

        # ── Eq. 11: Learned projections (OUTER-LOOP parameters) ──
        self.W_K = nn.Linear(self.d_model, self.d_model)
        self.W_V = nn.Linear(self.d_model, self.d_model)
        self.W_Q = nn.Linear(self.d_model, self.d_model)

        nn.init.xavier_normal_(self.W_K.weight)
        nn.init.xavier_normal_(self.W_V.weight)
        nn.init.xavier_normal_(self.W_Q.weight)
        nn.init.zeros_(self.W_K.bias)
        nn.init.zeros_(self.W_V.bias)
        nn.init.zeros_(self.W_Q.bias)

        # ── MLP layers: the actual memory (INNER-LOOP parameters) ──
        mlp_layers = []
        in_dim = self.d_model
        for i in range(self.num_layers):
            out_dim = self.d_model if i == self.num_layers - 1 else hidden_dim
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.mlp_layers = nn.ModuleList(mlp_layers)

        for i, layer in enumerate(self.mlp_layers):
            if i < self.num_layers - 1:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
            else:
                nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # ── Momentum buffers for inner-loop MLP params ──
        for name, param in self.mlp_layers.named_parameters():
            buf_name = "momentum_mlp_layers_" + name.replace(".", "_")
            self.register_buffer(buf_name, torch.zeros_like(param))

    # ── Projection helpers ───────────────────────────────────

    def project_key(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. 11: k_t = x_t @ W_K"""
        return self.W_K(x)

    def project_value(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. 11: v_t = x_t @ W_V"""
        return self.W_V(x)

    def project_query(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. 11: q_t = x_t @ W_Q"""
        return self.W_Q(x)

    # ── Outer-loop parameter helpers ─────────────────────────

    def outer_loop_parameters(self):
        """Return only the outer-loop parameters (W_K, W_V, W_Q).

        Use this to build the outer optimizer:
            optimizer = Adam(memory.outer_loop_parameters(), lr=...)
        """
        return list(self.W_K.parameters()) + \
               list(self.W_V.parameters()) + \
               list(self.W_Q.parameters())

    # ── MLP snapshot (for training episodes) ─────────────────

    def snapshot_mlp(self) -> Dict[str, torch.Tensor]:
        """Save a copy of MLP weights + momentum buffers.

        Call before a training episode so you can restore after.
        """
        state = {}
        for name, param in self.mlp_layers.named_parameters():
            state[name] = param.data.clone()
            buf_name = "momentum_mlp_layers_" + name.replace(".", "_")
            state[buf_name] = getattr(self, buf_name).clone()
        return state

    def restore_mlp(self, snapshot: Dict[str, torch.Tensor]) -> None:
        """Restore MLP weights + momentum buffers from a snapshot."""
        for name, param in self.mlp_layers.named_parameters():
            param.data.copy_(snapshot[name])
            buf_name = "momentum_mlp_layers_" + name.replace(".", "_")
            getattr(self, buf_name).copy_(snapshot[buf_name])

    def reset_mlp(self) -> None:
        """Re-initialize MLP weights and zero momentum buffers.

        Called at the start of each training episode so the inner loop
        starts from a clean slate.
        """
        for i, layer in enumerate(self.mlp_layers):
            if i < self.num_layers - 1:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
            else:
                nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        for name, param in self.mlp_layers.named_parameters():
            buf_name = "momentum_mlp_layers_" + name.replace(".", "_")
            getattr(self, buf_name).zero_()

    # ── Momentum buffer access ───────────────────────────────

    def _get_momentum_buffer(self, mlp_param_name: str) -> torch.Tensor:
        buf_name = "momentum_mlp_layers_" + mlp_param_name.replace(".", "_")
        return getattr(self, buf_name)

    def _set_momentum_buffer(self, mlp_param_name: str, value: torch.Tensor) -> None:
        buf_name = "momentum_mlp_layers_" + mlp_param_name.replace(".", "_")
        setattr(self, buf_name, value)

    # ── MLP forward ──────────────────────────────────────────

    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the memory MLP."""
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        h = x
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if i < self.num_layers - 1:
                h = F.silu(h)

        if squeeze:
            h = h.squeeze(0)
        return h

    # ── INFERENCE API (inner-loop only, on-the-go) ───────────

    def read(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Read from memory: x → W_Q → MLP forward.

        Paper Eq. 15: y_t = M*(q_t) where q_t = x_t @ W_Q.

        Args:
            x:        (d_model,) or (batch, d_model)
            training: If True, keeps grad graph alive for W_Q.
        """
        if training:
            q = self.project_query(x)
            return self._mlp_forward(q)
        else:
            with torch.no_grad():
                q = self.project_query(x)
                return self._mlp_forward(q)

    def write(self, x: torch.Tensor, training: bool = False) -> Union[float, torch.Tensor]:
        """Write to memory: x → (W_K, W_V) → MLP loss → Titans update.

        Args:
            x:        (d_model,) or (batch, d_model) — raw input embedding
            training: If True, returns loss tensor (graph alive for outer backward).
                      If False, returns loss as float (default inference mode).

        Returns:
            training=True:  loss tensor connected to W_K, W_V computation graph
            training=False: surprise as float
        """
        if training:
            return self._write_training(x)
        else:
            return self._write_inference(x)

    def _write_inference(self, x: torch.Tensor) -> float:
        """Inference-mode write: detach K/V, update MLP, return float."""
        k = self.project_key(x).detach()
        v = self.project_value(x).detach()

        output = self._mlp_forward(k)
        loss = F.mse_loss(output, v)
        loss.backward()

        with torch.no_grad():
            for name, param in self.mlp_layers.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad
                momentum = self._get_momentum_buffer(name)
                new_momentum = self.momentum_decay * momentum - self.lr * grad
                param.data.mul_(1.0 - self.forget_gate).add_(new_momentum)
                self._set_momentum_buffer(name, new_momentum)
                param.grad = None

        return loss.item()

    def _write_training(self, x: torch.Tensor) -> torch.Tensor:
        """Training-mode write: keep K/V grad graph alive for outer loop.

        Uses torch.autograd.grad to compute MLP gradients WITHOUT calling
        loss.backward(), so the computation graph for W_K/W_V stays alive.
        The outer optimizer can later backprop through the returned loss.

        Flow:
            k = W_K(x)             ← grad graph alive
            v = W_V(x)             ← grad graph alive
            output = MLP(k)
            loss = ||output - v||² ← returned to outer loop
            mlp_grads = autograd.grad(loss, mlp_params, retain_graph=True)
            apply Titans update to MLP weights (in-place, no_grad)
        """
        k = self.project_key(x)       # graph alive → W_K
        v = self.project_value(x)     # graph alive → W_V

        output = self._mlp_forward(k)
        loss = F.mse_loss(output, v)

        # Get MLP gradients without destroying the graph (retain_graph=True)
        # create_graph=False → first-order approximation (no higher-order grads)
        mlp_params = [p for p in self.mlp_layers.parameters()]
        mlp_grads = torch.autograd.grad(
            loss, mlp_params,
            retain_graph=True,
            create_graph=False,
        )

        # Apply Titans update to MLP weights (inner loop, in-place)
        with torch.no_grad():
            for (name, param), grad in zip(
                self.mlp_layers.named_parameters(), mlp_grads
            ):
                momentum = self._get_momentum_buffer(name)
                new_momentum = self.momentum_decay * momentum - self.lr * grad
                param.data.mul_(1.0 - self.forget_gate).add_(new_momentum)
                self._set_momentum_buffer(name, new_momentum)

        # Return loss tensor — outer loop calls .backward() on sum of these
        return loss

    # ── Convenience ──────────────────────────────────────────

    def write_batch(self, x: torch.Tensor, training: bool = False) -> Union[float, torch.Tensor]:
        """Batch write: x is (batch, d_model)."""
        return self.write(x, training=training)

    def get_surprise(self, x: torch.Tensor) -> float:
        """Compute surprise without updating weights."""
        with torch.no_grad():
            k = self.project_key(x)
            v = self.project_value(x)
            output = self._mlp_forward(k)
            loss = F.mse_loss(output, v)
        return loss.item()

    # ── PARALLEL WRITE (§3.2) ─────────────────────────────────

    def write_chunk_parallel(
        self,
        xs: torch.Tensor,
        chunk_size: int = 16,
    ) -> List[float]:
        """Write a sequence of tokens using chunked parallel processing (§3.2).

        Splits the input sequence into chunks of size b. Within each chunk,
        all gradients are computed w.r.t. the SAME starting weights (Eq. 17),
        enabling matmul parallelism. Momentum is updated via parallel
        associative scan (Eq. 18).

        Args:
            xs:         (seq_len, d_model) — sequence of input embeddings
            chunk_size: b — number of tokens per chunk

        Returns:
            List of per-token surprise (loss) values
        """
        seq_len = xs.shape[0]
        all_losses = []

        # Project all keys and values at once (detached for inference mode)
        keys_all = self.project_key(xs).detach()    # (seq_len, d_model)
        values_all = self.project_value(xs).detach()  # (seq_len, d_model)

        # Build momentum buffers dict for the chunked updater
        momentum_buffers = {}
        for name, param in self.mlp_layers.named_parameters():
            momentum_buffers[name] = self._get_momentum_buffer(name).clone()

        chunked_updater = ChunkedMLPMemoryUpdate(
            d_model=self.d_model,
            lr=self.lr,
            momentum_decay=self.momentum_decay,
            forget_gate=self.forget_gate,
        )

        # Process sequence in chunks of size b
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_keys = keys_all[start:end]      # (b, d_model)
            chunk_values = values_all[start:end]  # (b, d_model)

            # Enable grads on MLP params for this chunk's backward
            for param in self.mlp_layers.parameters():
                param.requires_grad_(True)

            chunk_losses, new_momentum_buffers = chunked_updater(
                mlp_layers=self.mlp_layers,
                keys=chunk_keys,
                values=chunk_values,
                momentum_buffers=momentum_buffers,
                num_layers=self.num_layers,
            )

            all_losses.extend(chunk_losses.tolist())
            momentum_buffers = new_momentum_buffers

        # Write back updated momentum buffers
        with torch.no_grad():
            for name, _ in self.mlp_layers.named_parameters():
                if name in momentum_buffers:
                    self._set_momentum_buffer(name, momentum_buffers[name])

        return all_losses
