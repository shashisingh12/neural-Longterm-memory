"""
Memory Adapter
================
Reads from M_t and conditions each token embedding in the LLM.

Trains jointly with M_t during Phase 1.
Frozen during Phase 2.

The adapter and M_t co-evolve during training:
  M_t learns to store in a way adapter can read.
  Adapter learns to read what M_t stores.
  LoRA learns to use what adapter produces.

Architecture:
  1. Project token embeddings to M_t query space via W_Q
  2. Attend over M_t weight matrix rows (content-based addressing)
  3. Per-token gating (learn which tokens benefit from memory)
  4. Project back to LLM space via W_V
  5. Residual addition with learnable strength alpha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .joint_config import JointTrainingConfig
from .differentiable_memory import DifferentiableNeuralMemory


class MemoryAdapter(nn.Module):
    """
    Conditions LLM token embeddings on M_t state.

    Given token embeddings (batch, seq, d_llm) and a NeuralMemory instance,
    produces conditioned embeddings that carry memory information.

    The conditioning is additive:
        output = token_embeds + alpha * gate * project_up(attend(project_down(token_embeds), M_t.W))
    """

    def __init__(self, config: JointTrainingConfig, d_llm: int):
        super().__init__()
        d_mem = config.d_mem
        self.d_mem = d_mem
        self.d_llm = d_llm

        # project from LLM token space → M_t query space
        self.W_Q = nn.Linear(d_llm, d_mem, bias=False)

        # project from M_t read space → LLM token space
        self.W_V = nn.Linear(d_mem, d_llm, bias=False)

        # per-token gate: should this token use memory?
        self.W_gate = nn.Sequential(
            nn.Linear(d_mem, config.adapter_gate_hidden),
            nn.SiLU(),
            nn.Linear(config.adapter_gate_hidden, 1),
            nn.Sigmoid(),
        )

        # learnable conditioning strength
        self.alpha = nn.Parameter(
            torch.tensor(config.adapter_alpha_init)
        )

        # layer norm for stability
        self.norm = nn.LayerNorm(d_llm)

        # init small so adapter starts near-identity
        nn.init.normal_(self.W_Q.weight, std=0.001)
        nn.init.normal_(self.W_V.weight, std=0.001)

    def forward(
        self,
        token_embeds: torch.Tensor,
        memory: DifferentiableNeuralMemory,
    ) -> tuple:
        """
        Condition token embeddings on M_t state.

        Args:
            token_embeds: (batch, seq, d_llm)
            memory:       DifferentiableNeuralMemory instance

        Returns:
            conditioned: (batch, seq, d_llm) — memory-conditioned embeddings
            gates:       (batch, seq, 1)     — per-token gate values
        """
        # current M_t weight state (d_mem, d_mem)
        mem_state = memory.get_state()

        # project tokens to M_t query space
        queries = self.W_Q(token_embeds)           # (batch, seq, d_mem)

        # content-based addressing: attend over M_t weight rows
        # scores[i,j,k] = how relevant is M_t row k to token j?
        scores = torch.matmul(
            queries, mem_state.T
        ) / (self.d_mem ** 0.5)                    # (batch, seq, d_mem)
        weights = torch.softmax(scores, dim=-1)    # (batch, seq, d_mem)

        # weighted blend of M_t weight rows → memory readout
        mem_out = torch.matmul(weights, mem_state)  # (batch, seq, d_mem)

        # per-token gate: which tokens benefit from memory?
        gates = self.W_gate(mem_out)               # (batch, seq, 1)

        # project back to LLM space + normalize
        signal = self.norm(self.W_V(mem_out))      # (batch, seq, d_llm)

        # additive conditioning with learned strength
        conditioned = token_embeds + self.alpha * gates * signal

        return conditioned, gates
