"""
Differentiable Neural Memory (M_t)
====================================
A PyTorch nn.Module whose weights ARE the long-term memory.

Two separate update mechanisms:

MECHANISM 1 — Outer loop (training only)
    Loss flows from LLM task loss backward through adapter.read(M_t)
    into M_t weights. M_t learns: store associations in a way the
    adapter can successfully read.

MECHANISM 2 — Inner loop (training + inference)
    loss_inner = ||M_t(k) - v||²
    gradient of this loss = surprise signal
    alpha_t = alpha_net(gradient magnitude) = dynamic write strength
    M_t.W += alpha_t * momentum_update
    M_t learns: store this specific association.
    At inference, alpha_t is the ONLY thing that varies per turn.

The alpha_net is a small MLP that maps surprise magnitude to write
strength. Trained jointly during Phase 1, frozen during Phase 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .joint_config import JointTrainingConfig


class DifferentiableNeuralMemory(nn.Module):
    """
    M_t as a proper differentiable PyTorch module.

    PATH 1 — Task gradient (outer loop, training only):
        loss_task = cross_entropy(LLM output, target)
        grad flows back through adapter.read(M_t) into M_t weights
        teaches M_t to store in a way the adapter can read

    PATH 2 — Surprise update (inner loop, training + inference):
        loss_inner = ||M_t(key) - value||²
        grad → surprise signal → alpha_t → momentum update
        teaches M_t to store this specific (key, value) pair

    Both paths active during Phase 1 training.
    Only PATH 2 active during Phase 2 inference.
    """

    def __init__(self, config: JointTrainingConfig):
        super().__init__()
        d = config.d_mem
        self.d_model = d
        self.n_layers = config.mem_num_layers

        # --- M_t weights as nn.Parameters (outer-loop gradient flows here) ---
        self.W = nn.ParameterList([
            nn.Parameter(
                torch.randn(d, d) * (2.0 / d) ** 0.5
            )
            for _ in range(self.n_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(d))
            for _ in range(self.n_layers)
        ])

        # --- Momentum buffers (inner-loop state, NOT parameters) ---
        for i in range(self.n_layers):
            self.register_buffer(f'mom_W{i}', torch.zeros(d, d))
            self.register_buffer(f'mom_b{i}', torch.zeros(d))

        # --- Hyperparameters ---
        self.momentum_decay = config.mem_momentum_decay
        self.forget = config.mem_forget

        # --- Alpha network ---
        # Learned mapping: surprise_magnitude → write_strength (alpha_t)
        # Trained jointly in Phase 1; at Phase 2 inference it runs
        # inside surprise_write (no_grad context, uses .item())
        self.alpha_net = nn.Sequential(
            nn.Linear(1, config.alpha_hidden),
            nn.SiLU(),
            nn.Linear(config.alpha_hidden, 1),
            nn.Sigmoid(),
        )
        self.alpha_scale = nn.Parameter(
            torch.tensor(config.alpha_scale_init)
        )

    # ── READ (differentiable forward pass) ──────────────

    def read(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through M_t.
        Differentiable — outer-loop gradient flows through W and b.

        Args:
            x: (d_mem,) or (batch, d_mem)

        Returns:
            Same shape as input.
        """
        h = x
        for i in range(self.n_layers):
            z = h @ self.W[i].T + self.b[i]
            h = F.silu(z) if i < self.n_layers - 1 else z
        return h

    # ── SURPRISE COMPUTATION (differentiable) ───────────

    def compute_surprise(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple:
        """
        Compute surprise for a (key, value) pair.

        Surprise = how wrong M_t is about this association.
        High surprise → new information → write strongly.
        Low surprise → already known → write weakly.

        Returns:
            loss_inner : scalar tensor (differentiable, for outer loop)
            surprise   : float (gradient magnitude, detached)
            alpha_t    : scalar tensor (dynamic write strength)
        """
        prediction = self.read(key)
        loss_inner = ((prediction - value) ** 2).mean()

        # surprise = gradient magnitude (direction of prediction error)
        grad = 2.0 * (prediction - value)
        surprise = grad.norm().detach()

        # alpha_t = learned function of surprise
        s_in = surprise.unsqueeze(0).unsqueeze(0)   # (1, 1)
        alpha_t = self.alpha_net(s_in).squeeze() * self.alpha_scale

        return loss_inner, surprise.item(), alpha_t

    # ── SURPRISE WRITE (inner-loop update) ──────────────

    @torch.no_grad()
    def surprise_write(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple:
        """
        Inner-loop write: update M_t weights via surprise-driven gradient.

        Used:
          Phase 1: after each simulated conversation turn
          Phase 2: after each real conversation turn (ONLY update path)

        alpha_t computed dynamically from surprise magnitude.
        High surprise = strong write = important new information.
        Low surprise  = weak write   = already known, skip.

        Args:
            key:   (d_mem,) tensor
            value: (d_mem,) tensor

        Returns:
            (alpha_val, surprise_val, loss_val) — all floats for logging
        """
        k = key.detach()
        v = value.detach()

        # --- forward through M_t ---
        prediction = self.read(k)
        loss_inner = ((prediction - v) ** 2).mean()

        # --- gradient of inner loss w.r.t. output ---
        grad_out = 2.0 * (prediction - v)

        # clip gradient to prevent momentum explosion
        grad_norm = grad_out.norm()
        max_grad_norm = 10.0
        if grad_norm > max_grad_norm:
            grad_out = grad_out * (max_grad_norm / grad_norm)

        # --- surprise → alpha_t (uses clipped norm) ---
        surprise = grad_out.norm()
        s_in = surprise.unsqueeze(0).unsqueeze(0)
        alpha_t = self.alpha_net(s_in).squeeze() * self.alpha_scale
        alpha_val = alpha_t.item()

        # --- manual backprop through the 2-layer MLP ---
        # Layer 0: z0 = k @ W[0].T + b[0], h1 = silu(z0)
        # Layer 1: z1 = h1 @ W[1].T + b[1], output = z1
        h0 = k
        z0 = h0 @ self.W[0].T + self.b[0]
        h1 = F.silu(z0)

        # grad w.r.t. W[1] and b[1]
        gW1 = torch.outer(grad_out, h1)
        gb1 = grad_out.clone()

        # grad w.r.t. h1
        grad_h1 = grad_out @ self.W[1]

        # grad through silu: silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
        sig_z0 = torch.sigmoid(z0)
        silu_deriv = sig_z0 * (1.0 + z0 * (1.0 - sig_z0))
        grad_z0 = grad_h1 * silu_deriv

        # grad w.r.t. W[0] and b[0]
        gW0 = torch.outer(grad_z0, h0)
        gb0 = grad_z0.clone()

        # --- momentum update with dynamic alpha_t ---
        self.mom_W0.mul_(self.momentum_decay).sub_(alpha_val * gW0)
        self.mom_W1.mul_(self.momentum_decay).sub_(alpha_val * gW1)
        self.mom_b0.mul_(self.momentum_decay).sub_(alpha_val * gb0)
        self.mom_b1.mul_(self.momentum_decay).sub_(alpha_val * gb1)

        # clip momentum to prevent drift
        max_mom = 1.0
        self.mom_W0.clamp_(-max_mom, max_mom)
        self.mom_W1.clamp_(-max_mom, max_mom)
        self.mom_b0.clamp_(-max_mom, max_mom)
        self.mom_b1.clamp_(-max_mom, max_mom)

        # --- weight decay + momentum step ---
        self.W[0].data.mul_(1.0 - self.forget).add_(self.mom_W0)
        self.W[1].data.mul_(1.0 - self.forget).add_(self.mom_W1)

        # clamp weights to prevent unbounded growth
        max_weight = 5.0
        self.W[0].data.clamp_(-max_weight, max_weight)
        self.W[1].data.clamp_(-max_weight, max_weight)
        self.b[0].data.mul_(1.0 - self.forget).add_(self.mom_b0)
        self.b[1].data.mul_(1.0 - self.forget).add_(self.mom_b1)

        return alpha_val, surprise.item(), loss_inner.item()

    # ── STATE ACCESS ────────────────────────────────────

    def get_state(self) -> torch.Tensor:
        """Return W[0] as the memory state tensor for adapter conditioning."""
        return self.W[0]

    # ── RESET ───────────────────────────────────────────

    def reset(self):
        """Reset M_t to blank slate (new conversation at inference)."""
        d = self.d_model
        for i in range(self.n_layers):
            self.W[i].data.copy_(
                torch.randn(d, d, device=self.W[i].device) * (2.0/d)**0.5
            )
            self.b[i].data.zero_()
        for i in range(self.n_layers):
            getattr(self, f'mom_W{i}').zero_()
            getattr(self, f'mom_b{i}').zero_()
