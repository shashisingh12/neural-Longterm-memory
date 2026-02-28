"""
MemoryLLM — Full Joint System
================================
Wires together:
  - Pretrained LLM (frozen base weights)
  - LoRA via HuggingFace peft library
  - DifferentiableNeuralMemory (M_t)
  - MemoryAdapter (conditions embeddings on M_t state)

TRAINING (Phase 1):
    M_t + adapter + LoRA all train together via joint optimizer.
    M_t receives gradients from two paths:
      1. Inner loop: surprise-driven self-update (association learning)
      2. Outer loop: task loss backward through adapter (readability learning)

INFERENCE (Phase 2):
    Everything frozen. Only M_t surprise writes + dynamic alpha update.
    Adapter and LoRA weights locked — they apply their learned transforms
    to whatever M_t has accumulated.
"""

import torch
import torch._dynamo
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .joint_config import JointTrainingConfig
from .differentiable_memory import DifferentiableNeuralMemory
from .memory_adapter import MemoryAdapter
from .utils import get_device


class MemoryLLM(nn.Module):
    """
    Complete memory-augmented LLM.

    Architecture:
        input_ids → embed_tokens → [MemoryAdapter hook] → transformer layers (with LoRA) → logits

    The MemoryAdapter hook intercepts token embeddings after the embedding
    layer and conditions them on M_t state before they enter the transformer.

    LoRA is injected via HuggingFace peft — handles all model architectures
    automatically (LLaMA, Mistral, GPT-NeoX, BitNet, etc.).
    """

    def __init__(self, config: JointTrainingConfig):
        super().__init__()
        self.config = config
        self.device = get_device(config.device)

        # --- Load pretrained LLM ---
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        resolved_dtype = dtype_map.get(config.torch_dtype, torch.float32)

        trust_remote = getattr(config, 'trust_remote_code', False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=trust_remote,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_str = str(self.device)

        # BitNet uses torch.compile (inductor) for weight quantization,
        # but inductor doesn't support MPS — disable dynamo to use eager mode.
        if device_str == "mps":
            torch._dynamo.disable()

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=resolved_dtype,
            device_map=device_str if device_str != "cpu" else None,
            trust_remote_code=trust_remote,
        )
        if device_str == "cpu":
            base_model = base_model.to(self.device)

        # auto-detect d_llm
        d_llm = base_model.config.hidden_size
        self.d_llm = d_llm

        # --- Apply LoRA via peft ---
        # peft's LoraConfig.layers_to_transform selects which layers get LoRA.
        # target_modules selects which linear projections inside each layer.
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_targets,
            layers_to_transform=config.lora_layers,
            bias="none",
            lora_dropout=0.0,
        )
        self.base_model = get_peft_model(base_model, lora_config)
        self.base_model.print_trainable_parameters()

        # --- M_t and Adapter (train jointly) ---
        self.memory = DifferentiableNeuralMemory(config).to(device=self.device, dtype=resolved_dtype)
        self.adapter = MemoryAdapter(config, d_llm).to(device=self.device, dtype=resolved_dtype)

        # --- Embedding hook ---
        self._hook_handle = self._attach_hook()

        # --- Print parameter summary ---
        self._print_params()

    def _get_embed_tokens(self):
        """Get the embedding layer (handles peft wrapping + different architectures)."""
        # peft wraps the model: self.base_model.base_model.model is the
        # actual transformer; self.base_model.model is also accessible.
        # Walk through common paths.
        inner = self.base_model.base_model
        if hasattr(inner, 'model') and hasattr(inner.model, 'embed_tokens'):
            return inner.model.embed_tokens
        # BitNet and similar: model.model.embed_tokens
        if hasattr(inner, 'model') and hasattr(inner.model, 'model') \
                and hasattr(inner.model.model, 'embed_tokens'):
            return inner.model.model.embed_tokens
        if hasattr(inner, 'transformer') and hasattr(inner.transformer, 'wte'):
            return inner.transformer.wte
        # fallback: try direct
        if hasattr(inner, 'embed_tokens'):
            return inner.embed_tokens
        raise AttributeError(
            "Cannot find embedding layer in the peft-wrapped model. "
            "Unsupported architecture."
        )

    def _attach_hook(self):
        """Register a forward hook on the embedding layer."""
        memory = self.memory
        adapter = self.adapter

        def hook_fn(module, input, output):
            # output shape: (batch, seq, d_llm)
            conditioned, _ = adapter(output, memory)
            return conditioned

        embed_layer = self._get_embed_tokens()
        return embed_layer.register_forward_hook(hook_fn)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        """
        Standard forward pass through the memory-augmented LLM.

        The embedding hook automatically conditions tokens on M_t.
        """
        return self.base_model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    # ── Parameter groups ──────────────────────────────────

    def training_parameters(self) -> list:
        """All parameters that update during Phase 1 training."""
        params = []
        # M_t weights + alpha_net + alpha_scale
        params += list(self.memory.parameters())
        # adapter weights
        params += list(self.adapter.parameters())
        # LoRA weights (managed by peft — only these have requires_grad)
        params += [p for p in self.base_model.parameters()
                   if p.requires_grad]
        return params

    def lora_parameters(self) -> list:
        """Only LoRA parameters (managed by peft)."""
        return [p for p in self.base_model.parameters()
                if p.requires_grad]

    def memory_parameters(self) -> list:
        """Only M_t parameters (W, b, alpha_net, alpha_scale)."""
        return list(self.memory.parameters())

    def adapter_parameters(self) -> list:
        """Only adapter parameters (W_Q, W_V, gate, alpha, norm)."""
        return list(self.adapter.parameters())

    # ── Phase transition ──────────────────────────────────

    def freeze_for_inference(self):
        """
        Freeze everything after Phase 1 training.

        Only M_t.surprise_write() will update weights at inference.
        That method runs under torch.no_grad() and modifies .data directly.
        """
        for p in self.adapter.parameters():
            p.requires_grad = False
        for p in self.base_model.parameters():
            p.requires_grad = False
        for p in self.memory.parameters():
            p.requires_grad = False
        print("  [MemoryLLM] Frozen for inference. "
              "Only M_t surprise writes active.")

    # ── Save / Load ───────────────────────────────────────

    def save_checkpoint(self, path: str):
        """Save trainable state (M_t + adapter + LoRA) to disk."""
        state = {
            "config": self.config.to_dict(),
            "memory": self.memory.state_dict(),
            "adapter": self.adapter.state_dict(),
        }
        torch.save(state, path)
        # also save peft adapter weights alongside
        peft_dir = path.replace(".pt", "_peft")
        self.base_model.save_pretrained(peft_dir)
        print(f"  [MemoryLLM] Checkpoint saved → {path}")
        print(f"  [MemoryLLM] LoRA weights  → {peft_dir}/")

    def load_checkpoint(self, path: str):
        """Load trainable state from disk."""
        state = torch.load(
            path,
            map_location=str(self.device),
            weights_only=False,
        )
        self.memory.load_state_dict(state["memory"])
        self.adapter.load_state_dict(state["adapter"])
        # load peft adapter weights
        peft_dir = path.replace(".pt", "_peft")
        self.base_model.load_adapter(peft_dir, adapter_name="default")
        print(f"  [MemoryLLM] Checkpoint loaded ← {path}")

    # ── Diagnostics ───────────────────────────────────────

    def _print_params(self):
        """Print parameter counts."""
        # peft model: trainable are LoRA params, rest are frozen
        base_p = sum(
            p.numel() for p in self.base_model.parameters()
            if not p.requires_grad
        )
        lora_p = sum(
            p.numel() for p in self.base_model.parameters()
            if p.requires_grad
        )
        mem_p = sum(p.numel() for p in self.memory.parameters())
        adap_p = sum(p.numel() for p in self.adapter.parameters())
        total = base_p + lora_p + mem_p + adap_p
        trainable = lora_p + mem_p + adap_p

        print(f"\n  Parameter Summary:")
        print(f"    Base (frozen): {base_p:>12,}")
        print(f"    LoRA (peft):   {lora_p:>12,}  trainable")
        print(f"    Memory M_t:    {mem_p:>12,}  trainable")
        print(f"    Adapter:       {adap_p:>12,}  trainable")
        print(f"    Total train:   {trainable:>12,}  "
              f"({100*trainable/max(total,1):.3f}%)")
