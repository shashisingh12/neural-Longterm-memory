"""
Titans Memory — Parallel Memory Training (§3.2)
=================================================
Parallelizes the inner-loop memory update using:

1. Chunk-based mini-batch gradient descent (Eq. 16-17):
   Split sequence into chunks of size b, compute all gradients
   within a chunk in parallel via matmuls.

2. Parallel associative scan (Eq. 18):
   The momentum recurrence S_t = η_t·S_{t-1} − θ_t·u_t is a
   linear recurrence computable in O(log b) parallel steps.

Key insight (Eq. 17): For a chunk of b tokens with the SAME
starting weights M_0, all gradients can be computed simultaneously:

    ∇ℓ(W_0; x_i) = (W_0·k_i − v_i)·k_i^T

Batched over b tokens:
    Grads = (W_0 @ K^T − V^T) · diag(Θ) · diag(B) · K

where K = (b, d), V = (b, d) are stacked keys/values,
Θ = diag(θ_1,...,θ_b), B = diag(β_b/β_1,...,β_b/β_b).

The cumulative weight decay: β_t = ∏_{j=1}^{t} (1 − α_j).

For MLP memory (N_p ≥ 2 layers), we compute batched forward/backward
through the MLP for all tokens in the chunk simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def parallel_associative_scan(
    gates: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Parallel associative scan for linear recurrence.

    Computes: S_t = gate_t * S_{t-1} + input_t
    for all t in parallel using O(log T) sequential steps.

    This solves the momentum recurrence (Eq. 18):
        S_t = η_t * S_{t-1} − θ_t * u_t
    where gate_t = η_t and input_t = −θ_t * u_t.

    Args:
        gates:  (T, ...) — multiplicative gates (η_t values)
        inputs: (T, ...) — additive inputs (−θ_t * u_t values)

    Returns:
        (T, ...) — the full sequence of S_1, S_2, ..., S_T
    """
    T = gates.shape[0]
    if T == 1:
        return inputs

    # Work on pairs: combine (gate, input) tuples via the associative operator
    # (a1, b1) ∘ (a2, b2) = (a1*a2, a2*b1 + b2)
    a = gates.clone()   # (T, ...)
    b = inputs.clone()  # (T, ...)

    # Up-sweep (reduce)
    # We use an iterative approach for the parallel scan
    result = torch.zeros_like(inputs)
    result[0] = inputs[0]

    # For GPU efficiency, use the sequential-parallel hybrid:
    # Process in blocks, parallel within blocks
    # For small T (typical chunk sizes 4-64), sequential is fine and avoids overhead
    if T <= 64:
        # Sequential scan (efficient for small chunks)
        h = inputs[0]
        result[0] = h
        for t in range(1, T):
            h = gates[t] * h + inputs[t]
            result[t] = h
        return result

    # For larger T, use the Blelloch parallel scan algorithm
    # This gives O(log T) depth with O(T) work
    num_levels = int(torch.tensor(T, dtype=torch.float32).log2().ceil().item())

    # Store intermediate (gate, value) pairs
    a_pairs = a.clone()
    b_pairs = b.clone()

    # Up-sweep phase
    for d in range(num_levels):
        stride = 2 ** (d + 1)
        indices = torch.arange(stride - 1, T, stride)
        prev_indices = indices - 2 ** d

        valid = (prev_indices >= 0) & (indices < T)
        idx = indices[valid]
        prev_idx = prev_indices[valid]

        if len(idx) > 0:
            # (a1, b1) ∘ (a2, b2) = (a1*a2, a2*b1 + b2)
            new_b = a_pairs[idx] * b_pairs[prev_idx] + b_pairs[idx]
            new_a = a_pairs[idx] * a_pairs[prev_idx]
            a_pairs[idx] = new_a
            b_pairs[idx] = new_b

    # Down-sweep phase: extract all prefix sums
    result = b_pairs.clone()

    # Correct intermediate positions
    for d in range(num_levels - 2, -1, -1):
        stride = 2 ** (d + 1)
        half_stride = 2 ** d
        indices = torch.arange(stride + half_stride - 1, T, stride)
        prev_indices = indices - half_stride

        valid = (prev_indices >= 0) & (indices < T)
        idx = indices[valid]
        prev_idx = prev_indices[valid]

        if len(idx) > 0:
            result[idx] = a_pairs[idx] * result[prev_idx] + b_pairs[idx]

    return result


def compute_cumulative_decay(
    alphas: torch.Tensor,
) -> torch.Tensor:
    """Compute cumulative weight decay: β_t = ∏_{j=1}^{t} (1 − α_j).

    Args:
        alphas: (T,) — per-token forgetting gates α_t

    Returns:
        betas: (T,) — cumulative decay factors β_t
    """
    log_decay = torch.log1p(-alphas)            # log(1 - α_t)
    cum_log_decay = torch.cumsum(log_decay, dim=0)
    return torch.exp(cum_log_decay)


def compute_beta_ratios(betas: torch.Tensor) -> torch.Tensor:
    """Compute β_T / β_i for scaling gradients in Eq. 17.

    Args:
        betas: (T,) — cumulative decay factors

    Returns:
        ratios: (T,) — β_T / β_i for each position
    """
    beta_T = betas[-1]
    return beta_T / betas.clamp(min=1e-10)


class ChunkedLinearMemoryUpdate(nn.Module):
    """Parallelized memory update for LINEAR memory (single layer W).

    Implements Eq. 16-17 exactly:
        M_b = β_b·M_0 − Θ_b·B_b·(W_0·K^T − V^T)·K

    All gradients in the chunk are computed w.r.t. the SAME starting
    weights W_0 (start-of-chunk), enabling full matmul parallelism.

    Then applies momentum via parallel associative scan (Eq. 18).
    """

    def __init__(self, d_model: int, lr: float, momentum_decay: float, forget_gate: float):
        super().__init__()
        self.d_model = d_model
        self.lr = lr
        self.momentum_decay = momentum_decay
        self.forget_gate = forget_gate

    def compute_chunk_gradients(
        self,
        W_0: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute all gradients in a chunk via matmul (Eq. 17).

        For linear memory: ∇ℓ(W_0; x_i) = (W_0 @ k_i − v_i) @ k_i^T

        Args:
            W_0:    (d_out, d_in) — starting weights
            keys:   (b, d_in)  — chunk of key vectors
            values: (b, d_out) — chunk of value vectors

        Returns:
            grads: (b, d_out, d_in) — per-token gradient matrices
        """
        b = keys.shape[0]
        # W_0 @ k_i for all i: (d_out, d_in) @ (d_in, b) = (d_out, b)
        predictions = W_0 @ keys.T              # (d_out, b)
        errors = predictions - values.T          # (d_out, b)
        # Outer product for each token: errors[:, i] @ keys[i, :]
        # = (b, d_out, 1) * (b, 1, d_in) → (b, d_out, d_in)
        grads = errors.T.unsqueeze(2) * keys.unsqueeze(1)  # (b, d_out, d_in)
        return 2.0 * grads  # Factor of 2 from MSE derivative

    def forward(
        self,
        W: torch.Tensor,
        b_param: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        momentum_W: torch.Tensor,
        momentum_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process one chunk with parallel gradient computation + scan.

        Args:
            W:          (d_out, d_in) — current weight matrix
            b_param:    (d_out,) — current bias
            keys:       (chunk_size, d_in) — key vectors
            values:     (chunk_size, d_out) — value vectors
            momentum_W: (d_out, d_in) — momentum buffer for W
            momentum_b: (d_out,) — momentum buffer for bias

        Returns:
            W_new, b_new, momentum_W_new, momentum_b_new, chunk_losses
        """
        chunk_size = keys.shape[0]
        device = keys.device

        # ── Step 1: Compute all gradients in parallel (Eq. 17) ──
        # All use W_0 (start-of-chunk weights)
        W_0 = W.detach()

        # Weight gradients: (chunk_size, d_out, d_in)
        W_grads = self.compute_chunk_gradients(W_0, keys, values)

        # Bias gradients: ∂loss/∂b = 2*(W@k + b - v)
        predictions = (W_0 @ keys.T).T + b_param.detach()  # (chunk_size, d_out)
        b_grads = 2.0 * (predictions - values)               # (chunk_size, d_out)

        # Per-token losses
        losses = ((predictions - values) ** 2).mean(dim=1)   # (chunk_size,)

        # ── Step 2: Compute scaling factors ──
        # α (forgetting) — constant per chunk for simplicity (paper §3.2 last para)
        alphas = torch.full((chunk_size,), self.forget_gate, device=device)
        betas = compute_cumulative_decay(alphas)             # (chunk_size,)
        beta_ratios = compute_beta_ratios(betas)             # (chunk_size,)

        # θ (learning rate) — constant per chunk
        thetas = torch.full((chunk_size,), self.lr, device=device)

        # Scale gradients: θ_i * (β_b / β_i) * grad_i
        scale = thetas * beta_ratios                         # (chunk_size,)

        scaled_W_grads = scale.view(-1, 1, 1) * W_grads     # (chunk_size, d_out, d_in)
        scaled_b_grads = scale.view(-1, 1) * b_grads        # (chunk_size, d_out)

        # ── Step 3: Parallel associative scan for momentum (Eq. 18) ──
        # S_t = η_t * S_{t-1} − θ_t * u_t
        # gates = η (momentum_decay), inputs = −scaled_grads

        eta = torch.full((chunk_size,), self.momentum_decay, device=device)

        # For W momentum: flatten grads, scan, reshape
        W_flat = scaled_W_grads.view(chunk_size, -1)             # (chunk_size, d_out*d_in)
        W_inputs = -W_flat                                        # negative because S = η*S - θ*grad
        gates_W = eta.unsqueeze(1).expand_as(W_inputs)           # (chunk_size, d_out*d_in)

        # Prepend momentum from previous chunk
        W_inputs[0] = gates_W[0, 0] * momentum_W.view(-1) + W_inputs[0]

        W_momentum_seq = parallel_associative_scan(gates_W, W_inputs)  # (chunk_size, d_out*d_in)
        new_momentum_W = W_momentum_seq[-1].view_as(W)                 # final momentum

        # Same for bias
        b_inputs = -scaled_b_grads                                     # (chunk_size, d_out)
        gates_b = eta.unsqueeze(1).expand_as(b_inputs)
        b_inputs[0] = gates_b[0, 0] * momentum_b + b_inputs[0]

        b_momentum_seq = parallel_associative_scan(gates_b, b_inputs)
        new_momentum_b = b_momentum_seq[-1]                            # (d_out,)

        # ── Step 4: Apply cumulative update to weights ──
        # M_b = β_b * M_0 + S_b  (Eq. 16 combined)
        beta_total = betas[-1]
        W_new = beta_total * W_0 + new_momentum_W
        b_new = beta_total * b_param.detach() + new_momentum_b

        return W_new, b_new, new_momentum_W, new_momentum_b, losses


class ChunkedMLPMemoryUpdate(nn.Module):
    """Parallelized memory update for MLP memory (multiple layers).

    For MLP with N_p layers, we:
    1. Batch-forward all chunk tokens through the MLP simultaneously
    2. Batch-backward to get per-token gradients for all layers
    3. Apply chunk-parallel momentum scan + weight decay per layer

    This extends the linear case (Eq. 17) to deep networks.
    """

    def __init__(self, d_model: int, lr: float, momentum_decay: float, forget_gate: float):
        super().__init__()
        self.d_model = d_model
        self.lr = lr
        self.momentum_decay = momentum_decay
        self.forget_gate = forget_gate

    def forward(
        self,
        mlp_layers: nn.ModuleList,
        keys: torch.Tensor,
        values: torch.Tensor,
        momentum_buffers: dict,
        num_layers: int,
    ) -> Tuple[torch.Tensor, dict]:
        """Process one chunk through MLP with parallel gradient computation.

        Args:
            mlp_layers:       The MLP nn.ModuleList
            keys:             (chunk_size, d_model) — projected keys
            values:           (chunk_size, d_model) — projected values
            momentum_buffers: dict of momentum tensors per parameter
            num_layers:       number of MLP layers

        Returns:
            losses:            (chunk_size,) per-token losses
            new_momentum_bufs: updated momentum buffers
        """
        chunk_size = keys.shape[0]
        device = keys.device

        # ── Batched forward through MLP (all tokens at once) ──
        h = keys
        for i, layer in enumerate(mlp_layers):
            h = layer(h)
            if i < num_layers - 1:
                h = F.silu(h)
        outputs = h  # (chunk_size, d_model)

        # Per-token losses
        per_token_loss = ((outputs - values.detach()) ** 2).mean(dim=1)  # (chunk_size,)
        total_loss = F.mse_loss(outputs, values.detach())

        # ── Get per-parameter gradients via backward ──
        # We use the total loss to get gradients, then distribute per chunk
        total_loss.backward()

        # ── Apply chunk-parallel update per parameter ──
        chunk_alpha = self.forget_gate
        chunk_beta = (1.0 - chunk_alpha) ** chunk_size
        eta = self.momentum_decay
        theta = self.lr

        new_momentum_bufs = {}

        with torch.no_grad():
            for name, param in mlp_layers.named_parameters():
                if param.grad is None:
                    continue

                grad = param.grad
                momentum = momentum_buffers[name]

                # For chunk processing: apply the cumulative update
                # S_b = η^b * S_0 − θ * Σ(η^(b-i) * grad_i)
                # Since all grads are computed at W_0, and we have
                # the mean grad from backward(), we scale appropriately
                # Geometric sum of η: (1 - η^b) / (1 - η) if η != 1
                if abs(eta - 1.0) > 1e-8:
                    eta_sum = (1.0 - eta ** chunk_size) / (1.0 - eta)
                else:
                    eta_sum = float(chunk_size)

                new_momentum = (eta ** chunk_size) * momentum - theta * eta_sum * grad
                param.data.mul_(chunk_beta).add_(new_momentum)

                new_momentum_bufs[name] = new_momentum.clone()
                param.grad = None

        return per_token_loss.detach(), new_momentum_bufs
