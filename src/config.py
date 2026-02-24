"""
Titans Memory — Configuration
==============================
Centralizes every hyperparameter into a single dataclass.
Provides an argparse builder so all params can be set from CLI.
"""

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import List


@dataclass
class TitansConfig:
    """All hyperparameters for the Titans memory system."""

    # ── Encoder ──────────────────────────────────────────────
    tokenizer_name: str = "bert-base-uncased"
    d_model: int = 128
    max_seq_len: int = 512
    freeze_backbone: bool = True
    pooling_strategy: str = "mean"  # "mean", "cls", "max"

    # ── Neural Memory MLP ────────────────────────────────────
    memory_num_layers: int = 2
    memory_hidden_dim: int = 128
    memory_lr: float = 0.01          # theta_t: inner-loop learning rate
    memory_momentum_decay: float = 0.9  # eta_t: surprise momentum decay
    memory_forget_gate: float = 0.02    # alpha_t: forgetting / weight decay

    # ── Retrieval ────────────────────────────────────────────
    top_k: int = 3
    similarity_threshold: float = 0.05

    # ── Persistent Memory ────────────────────────────────────
    persistent_tokens: List[str] = field(default_factory=list)
    sos_token: str = "<SOS>"
    eos_token: str = "<EOS>"
    persistent_max_length: int = 0  # 0 = unlimited

    # ── MAC Architecture (Memory as Context, §4.1) ─────────
    num_persistent_vectors: int = 4      # N_p: learnable persistent param vectors
    num_attention_heads: int = 4         # heads in the MAC attention module
    attention_dropout: float = 0.0       # dropout in attention
    segment_size: int = 1               # tokens per segment (1 = per-turn)

    # ── Active Memory Layer ──────────────────────────────
    memory_transcript_size: int = 32     # ring buffer capacity (replaces unbounded RAG store)
    train_attention: bool = True         # train attention params in outer loop
    train_persistent_vectors: bool = True  # train persistent vectors in outer loop

    # ── Parallel Memory (§3.2) ────────────────────────────
    chunk_size: int = 16                 # b: tokens per chunk for parallel gradient computation
    use_parallel_memory: bool = False    # enable chunked parallel memory updates

    # ── Outer-Loop Training ─────────────────────────────────
    outer_lr: float = 1e-4               # learning rate for W_K, W_V, W_Q
    outer_epochs: int = 10               # training epochs
    episode_len: int = 8                 # texts per training episode
    eval_ratio: float = 0.25             # fraction of episode used for eval
    train_data_path: str = ""            # path to training text file (one per line)
    outer_weight_decay: float = 0.0      # AdamW weight decay
    grad_clip: float = 1.0              # max gradient norm (0 = no clipping)
    train_encoder_projection: bool = True  # also train encoder's projection head

    # ── Runtime ──────────────────────────────────────────────
    device: str = "cpu"
    verbose: bool = False
    checkpoint_path: str = ""
    seed: int = 42

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TitansConfig":
        """Build config from parsed CLI arguments."""
        kwargs = {}
        for f in dataclasses.fields(cls):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TitansConfig":
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


def build_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser with every Titans hyperparameter."""
    p = argparse.ArgumentParser(
        description="Titans-Inspired External Memory Layer (PyTorch)"
    )

    # Encoder
    p.add_argument("--tokenizer-name", type=str, default="bert-base-uncased",
                    help="HuggingFace model/tokenizer ID (default: bert-base-uncased)")
    p.add_argument("--d-model", type=int, default=128,
                    help="Latent dimension for memory vectors (default: 128)")
    p.add_argument("--max-seq-len", type=int, default=512,
                    help="Max token length for the tokenizer (default: 512)")
    p.add_argument("--freeze-backbone", action="store_true", default=True,
                    help="Freeze the HuggingFace backbone weights (default: True)")
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone",
                    action="store_false",
                    help="Unfreeze the HuggingFace backbone for fine-tuning")
    p.add_argument("--pooling-strategy", type=str, default="mean",
                    choices=["mean", "cls", "max"],
                    help="Pooling over token embeddings (default: mean)")

    # Neural Memory MLP
    p.add_argument("--memory-num-layers", type=int, default=2,
                    help="Number of layers in the memory MLP (default: 2)")
    p.add_argument("--memory-hidden-dim", type=int, default=128,
                    help="Hidden dim of memory MLP (default: 128)")
    p.add_argument("--memory-lr", type=float, default=0.01,
                    help="theta_t: inner-loop learning rate (default: 0.01)")
    p.add_argument("--memory-momentum-decay", type=float, default=0.9,
                    help="eta_t: surprise momentum decay (default: 0.9)")
    p.add_argument("--memory-forget-gate", type=float, default=0.02,
                    help="alpha_t: forgetting gate / weight decay (default: 0.02)")

    # Retrieval
    p.add_argument("--top-k", type=int, default=3,
                    help="Number of memory snippets to retrieve (default: 3)")
    p.add_argument("--similarity-threshold", type=float, default=0.05,
                    help="Min cosine similarity for retrieval (default: 0.05)")

    # Persistent Memory
    p.add_argument("--persistent-tokens", type=str, nargs="*", default=[],
                    help="Fixed context tokens prepended to every prompt")
    p.add_argument("--sos-token", type=str, default="<SOS>",
                    help="Start-of-sequence delimiter (default: <SOS>)")
    p.add_argument("--eos-token", type=str, default="<EOS>",
                    help="End-of-sequence delimiter (default: <EOS>)")
    p.add_argument("--persistent-max-length", type=int, default=0,
                    help="Max char length for persistent context, 0=unlimited (default: 0)")

    # MAC Architecture
    p.add_argument("--num-persistent-vectors", type=int, default=4,
                    help="N_p: number of learnable persistent memory vectors (default: 4)")
    p.add_argument("--num-attention-heads", type=int, default=4,
                    help="Number of heads in MAC attention module (default: 4)")
    p.add_argument("--attention-dropout", type=float, default=0.0,
                    help="Dropout rate in attention (default: 0.0)")
    p.add_argument("--segment-size", type=int, default=1,
                    help="Tokens per segment for chunking (default: 1)")

    # Active Memory Layer
    p.add_argument("--memory-transcript-size", type=int, default=32,
                    help="Ring buffer capacity for active memory transcript (default: 32)")
    p.add_argument("--train-attention", action="store_true", default=True,
                    help="Train attention params in outer loop (default: True)")
    p.add_argument("--no-train-attention", dest="train_attention",
                    action="store_false",
                    help="Freeze attention during outer-loop training")
    p.add_argument("--train-persistent-vectors", action="store_true", default=True,
                    help="Train persistent vectors in outer loop (default: True)")
    p.add_argument("--no-train-persistent-vectors", dest="train_persistent_vectors",
                    action="store_false",
                    help="Freeze persistent vectors during outer-loop training")

    # Parallel Memory (§3.2)
    p.add_argument("--chunk-size", type=int, default=16,
                    help="b: chunk size for parallel gradient computation (default: 16)")
    p.add_argument("--use-parallel-memory", action="store_true", default=False,
                    help="Enable chunked parallel memory updates (§3.2)")

    # Outer-Loop Training
    p.add_argument("--outer-lr", type=float, default=1e-4,
                    help="Learning rate for outer-loop params W_K,W_V,W_Q (default: 1e-4)")
    p.add_argument("--outer-epochs", type=int, default=10,
                    help="Number of outer-loop training epochs (default: 10)")
    p.add_argument("--episode-len", type=int, default=8,
                    help="Number of texts per training episode (default: 8)")
    p.add_argument("--eval-ratio", type=float, default=0.25,
                    help="Fraction of episode texts used for evaluation (default: 0.25)")
    p.add_argument("--train-data-path", type=str, default="",
                    help="Path to training data file (one text per line)")
    p.add_argument("--outer-weight-decay", type=float, default=0.0,
                    help="AdamW weight decay for outer loop (default: 0.0)")
    p.add_argument("--grad-clip", type=float, default=1.0,
                    help="Max gradient norm for clipping, 0=no clip (default: 1.0)")
    p.add_argument("--train-encoder-projection", action="store_true", default=True,
                    help="Also train encoder projection head (default: True)")
    p.add_argument("--no-train-encoder-projection", dest="train_encoder_projection",
                    action="store_false",
                    help="Freeze encoder projection during training")

    # Runtime
    p.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda", "mps"],
                    help="Compute device (default: cpu)")
    p.add_argument("--verbose", action="store_true", default=False,
                    help="Print debug info each turn")
    p.add_argument("--checkpoint-path", type=str, default="",
                    help="Path for saving/loading checkpoints")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")

    return p
