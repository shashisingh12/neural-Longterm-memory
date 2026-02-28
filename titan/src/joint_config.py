"""
Joint Training Configuration
==============================
Extends TitansConfig with parameters for the joint training system:
  - LLM backbone selection
  - LoRA hyperparameters
  - Memory adapter settings
  - Dynamic alpha network config
  - Phase 1 / Phase 2 training params
"""

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class JointTrainingConfig:
    """All hyperparameters for the joint M_t + adapter + LoRA system."""

    # ── LLM Backbone ──────────────────────────────────────
    model_name: str = "microsoft/bitnet-b1.58-2B-4T-bf16"
    d_llm: int = 0             # auto-detected from model config
    torch_dtype: str = "bfloat16"

    # ── Memory M_t ────────────────────────────────────────
    d_mem: int = 64            # memory vector dimension
    mem_num_layers: int = 2    # layers in M_t MLP
    mem_momentum_decay: float = 0.90
    mem_forget: float = 0.02

    # ── Alpha Network (surprise → write strength) ────────
    alpha_hidden: int = 16     # hidden dim in alpha_net MLP
    alpha_scale_init: float = 0.5  # initial max alpha output

    # ── Adapter ───────────────────────────────────────────
    adapter_gate_hidden: int = 16
    adapter_alpha_init: float = 0.1  # initial conditioning strength

    # ── LoRA ──────────────────────────────────────────────
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_layers: List[int] = field(default_factory=lambda: [0, 1, 14, 15, 28, 29])
    lora_targets: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # ── Phase 1 Training ─────────────────────────────────
    lr_lora: float = 2e-5
    lr_memory: float = 1e-4
    lr_adapter: float = 1e-4
    weight_decay_lora: float = 0.01
    weight_decay_memory: float = 0.001
    weight_decay_adapter: float = 0.01
    n_epochs: int = 1
    max_seq_len: int = 256
    grad_clip: float = 1.0
    inner_loss_weight: float = 0.1   # weight for M_t inner loss in total

    # ── Text Encoder (for key/value encoding) ────────────
    encoder_d_model: int = 64  # should match d_mem
    encoder_seed: int = 0

    # ── Runtime ───────────────────────────────────────────
    device: str = "cpu"
    seed: int = 42
    verbose: bool = False
    checkpoint_path: str = ""
    trust_remote_code: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JointTrainingConfig":
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "JointTrainingConfig":
        """Load config from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "JointTrainingConfig":
        kwargs = {}
        for f in dataclasses.fields(cls):
            key = f.name
            if hasattr(args, key):
                kwargs[key] = getattr(args, key)
        return cls(**kwargs)

    @classmethod
    def from_yaml_with_overrides(
        cls, yaml_path: str, args: argparse.Namespace, parser: argparse.ArgumentParser
    ) -> "JointTrainingConfig":
        """Load YAML config, then override with any explicitly-passed CLI args."""
        # start from YAML
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # find which CLI args were explicitly provided (not just defaults)
        defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
        for f in dataclasses.fields(cls):
            arg_val = getattr(args, f.name, None)
            if arg_val is None:
                continue
            # override only if the user explicitly passed it on the CLI
            if f.name in defaults and arg_val != defaults[f.name]:
                cfg[f.name] = arg_val
            elif f.name not in cfg:
                cfg[f.name] = arg_val

        return cls.from_dict(cfg)


def build_joint_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser for joint training."""
    p = argparse.ArgumentParser(
        description="Joint Training: M_t + Adapter + LoRA"
    )

    # Config file
    p.add_argument("--config", type=str, default="",
                    help="Path to YAML config file (CLI args override YAML values)")

    # LLM
    p.add_argument("--model-name", type=str,
                    default="microsoft/bitnet-b1.58-2B-4T-bf16",
                    help="HuggingFace model ID")
    p.add_argument("--torch-dtype", type=str, default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])

    # Memory
    p.add_argument("--d-mem", type=int, default=64,
                    help="Memory vector dimension (default: 64)")
    p.add_argument("--mem-num-layers", type=int, default=2)
    p.add_argument("--mem-momentum-decay", type=float, default=0.90)
    p.add_argument("--mem-forget", type=float, default=0.02)

    # Alpha network
    p.add_argument("--alpha-hidden", type=int, default=16)
    p.add_argument("--alpha-scale-init", type=float, default=0.5)

    # Adapter
    p.add_argument("--adapter-gate-hidden", type=int, default=16)
    p.add_argument("--adapter-alpha-init", type=float, default=0.1)

    # LoRA
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-layers", type=int, nargs="+",
                    default=[0, 1, 14, 15, 28, 29],
                    help="Transformer layer indices for LoRA injection")
    p.add_argument("--lora-targets", type=str, nargs="+",
                    default=["q_proj", "v_proj"],
                    help="Attention projection names to apply LoRA")

    # Phase 1 training
    p.add_argument("--lr-lora", type=float, default=2e-5)
    p.add_argument("--lr-memory", type=float, default=1e-4)
    p.add_argument("--lr-adapter", type=float, default=1e-4)
    p.add_argument("--weight-decay-lora", type=float, default=0.01)
    p.add_argument("--weight-decay-memory", type=float, default=0.001)
    p.add_argument("--weight-decay-adapter", type=float, default=0.01)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--inner-loss-weight", type=float, default=0.1)

    # Text encoder
    p.add_argument("--encoder-d-model", type=int, default=64)
    p.add_argument("--encoder-seed", type=int, default=0)

    # Runtime
    p.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--checkpoint-path", type=str, default="")
    p.add_argument("--trust-remote-code", action="store_true", default=False,
                    help="Trust remote code for custom HF models")

    return p
