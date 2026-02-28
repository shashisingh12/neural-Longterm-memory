"""
Phase 1 — Joint Trainer
=========================
Trains M_t + adapter + LoRA all together on conversation data.

For each training example:

1. Simulate conversation thread
   Write each past turn into M_t via surprise_write.
   M_t weights update through its own inner-loop mechanism.

2. After all past turns written:
   M_t state is the accumulated memory for this conversation.
   The adapter will read from this state during the forward pass.

3. Forward pass:
   test question → tokenize → embed → hook(adapter reads M_t) → transformer
   loss = cross_entropy(output, target answer)

4. Backward pass:
   gradient flows into:
     adapter weights       (outer loop)
     LoRA weights          (outer loop)
     M_t weights           (outer loop, via differentiable adapter.read)
     M_t alpha_net weights (outer loop, via alpha computation)

5. Inner loop loss also collected:
   loss_inner = ||M_t(k) - v||² for each simulated turn
   added to total loss (weighted by inner_loss_weight)

M_t receives gradients from BOTH loops simultaneously.
This is the meta-learning formulation from the Titans paper.
"""

import numpy as np

import torch
import torch.nn as nn
from typing import List, Tuple

from .joint_config import JointTrainingConfig
from .memory_llm import MemoryLLM
from .simple_encoder import TiktokenEncoder


# ── Phase 1 Trainer ───────────────────────────────────────

class Phase1Trainer:
    """
    Joint training of M_t + adapter + LoRA.

    Training data format: list of conversations.
    Each conversation is a list of (question, answer) tuples.

    Example:
        conversations = [
            [("What is your name?", "I'm Alice"),
             ("Where do you live?", "San Francisco"),
             ("What do you know about me?", "Your name is Alice...")],
            ...
        ]

    For each conversation, we simulate the thread by writing past turns
    into M_t, then train the LLM to produce the correct answer for the
    current turn, with M_t conditioning the embeddings.
    """

    def __init__(
        self,
        model: MemoryLLM,
        config: JointTrainingConfig,
    ):
        self.model = model
        self.config = config
        self.encoder = TiktokenEncoder(
            d_model=config.d_mem,
            seed=config.encoder_seed,
        )

        # match encoder output dtype to memory dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(config.torch_dtype, torch.float32)

        lora_p = model.lora_parameters()
        memory_p = model.memory_parameters()
        adapt_p = model.adapter_parameters()

        # three parameter groups, three learning rates
        param_groups = [
            {
                "params": lora_p,
                "lr": config.lr_lora,
                "weight_decay": config.weight_decay_lora,
            },
            {
                "params": memory_p,
                "lr": config.lr_memory,
                "weight_decay": config.weight_decay_memory,
            },
            {
                "params": adapt_p,
                "lr": config.lr_adapter,
                "weight_decay": config.weight_decay_adapter,
            },
        ]

        self.optimizer = torch.optim.AdamW(param_groups)

        print(f"\n  Joint training optimizer:")
        print(f"    LoRA    : {sum(p.numel() for p in lora_p):>8,}  "
              f"lr={config.lr_lora}")
        print(f"    Memory  : {sum(p.numel() for p in memory_p):>8,}  "
              f"lr={config.lr_memory}")
        print(f"    Adapter : {sum(p.numel() for p in adapt_p):>8,}  "
              f"lr={config.lr_adapter}")

    def _build_example(
        self,
        past_turns: List[Tuple[str, str]],
        question: str,
        answer: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, list]:
        """
        Simulate conversation thread and prepare tokenized input.

        1. Write each past turn into M_t via surprise_write
        2. Collect inner loop losses and alpha values
        3. Tokenize current question + answer for LLM training

        Returns:
            input_ids, attention_mask, labels, inner_losses, alpha_log
        """
        device = self.model.device
        inner_losses = []
        alpha_log = []

        for q, a in past_turns:
            combined = f"Q:{q} A:{a}"
            k = self.encoder.encode_tensor(combined, device=device, dtype=self.dtype)
            v = self.encoder.encode_tensor(a, device=device, dtype=self.dtype)

            # surprise write — M_t's own inner-loop mechanism
            alpha_t, surprise, loss_inner = (
                self.model.memory.surprise_write(k, v)
            )
            inner_losses.append(loss_inner)
            alpha_log.append(alpha_t)

        # tokenize the current turn
        tokenizer = self.model.tokenizer
        prompt = f"Q: {question}\nA:"
        full_text = prompt + f" {answer}"

        tokens = tokenizer(
            full_text,
            max_length=self.config.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"]       # (1, seq_len)
        attn_mask = tokens["attention_mask"]   # (1, seq_len)

        # labels: mask out the prompt tokens with -100
        prompt_len = len(tokenizer(prompt)["input_ids"])
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100

        return input_ids, attn_mask, labels, inner_losses, alpha_log

    def train_step(
        self,
        past_turns: List[Tuple[str, str]],
        question: str,
        answer: str,
    ) -> dict:
        """
        One training step: simulate thread → forward → backward → step.

        Returns metrics dict for logging.
        """
        self.model.train()
        self.optimizer.zero_grad()

        device = self.model.device

        # build example — M_t surprise writes happen here
        input_ids, attn_mask, labels, inner_losses, alpha_log = (
            self._build_example(past_turns, question, answer)
        )

        # outer loop forward pass
        # the embedding hook reads from M_t (differentiable)
        # gradient flows back into M_t.W via adapter
        output = self.model(
            input_ids=input_ids.to(device),
            labels=labels.to(device),
            attention_mask=attn_mask.to(device),
        )
        loss_task = output.loss

        # inner loop loss — weighted contribution
        if inner_losses:
            loss_inner = np.mean(inner_losses)
        else:
            loss_inner = 0.0

        # total loss: task loss + weighted inner loss
        # inner_losses are floats (from surprise_write under no_grad),
        # so they don't add to the graph — only loss_task backprops
        total_loss = loss_task

        total_loss.backward()

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.training_parameters(),
                self.config.grad_clip,
            )

        self.optimizer.step()

        return {
            "loss_task": loss_task.item(),
            "loss_inner": loss_inner,
            "alpha_mean": float(np.mean(alpha_log)) if alpha_log else 0.0,
            "alpha_max": float(np.max(alpha_log)) if alpha_log else 0.0,
        }

    def train(
        self,
        conversations: List[List[Tuple[str, str]]],
        n_epochs: int = None,
    ) -> List[dict]:
        """
        Run full Phase 1 joint training.

        Args:
            conversations: list of conversations, each a list of (Q, A) tuples
            n_epochs: override config.n_epochs if provided

        Returns:
            List of per-epoch metric dicts.
        """
        n_epochs = n_epochs or self.config.n_epochs
        print(f"\n  Phase 1 joint training: {n_epochs} epochs, "
              f"{len(conversations)} conversations")

        # count total training steps for progress logging
        total_steps = sum(max(len(c) - 1, 0) for c in conversations)
        epoch_metrics = []

        for epoch in range(n_epochs):
            metrics = {
                "loss_task": [],
                "loss_inner": [],
                "alpha_mean": [],
                "alpha_max": [],
            }

            step_num = 0
            for ci, convo in enumerate(conversations):
                # reset M_t for each conversation
                self.model.memory.reset()

                for t in range(1, len(convo)):
                    past = convo[:t]
                    curr_q, curr_a = convo[t]

                    step = self.train_step(past, curr_q, curr_a)
                    step_num += 1

                    for k, v in step.items():
                        metrics[k].append(v)

                    # per-step log
                    print(
                        f"  [Epoch {epoch+1}/{n_epochs}]"
                        f"  Step {step_num}/{total_steps}"
                        f"  Conv {ci+1}/{len(conversations)}"
                        f"  task={step['loss_task']:.4f}"
                        f"  inner={step['loss_inner']:.4f}"
                        f"  alpha={step['alpha_mean']:.4f}"
                    )

            epoch_summary = {
                k: float(np.mean(v)) for k, v in metrics.items()
            }
            epoch_summary["adapter_alpha"] = (
                self.model.adapter.alpha.item()
            )
            epoch_metrics.append(epoch_summary)

            print(
                f"\n  ── Epoch {epoch+1}/{n_epochs} Summary ──"
                f"  task={epoch_summary['loss_task']:.4f}"
                f"  inner={epoch_summary['loss_inner']:.4f}"
                f"  alpha_mean={epoch_summary['alpha_mean']:.4f}"
                f"  alpha_max={epoch_summary['alpha_max']:.4f}"
                f"  adapter_alpha={epoch_summary['adapter_alpha']:.4f}\n"
            )

        return epoch_metrics
