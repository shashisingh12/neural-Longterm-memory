"""
Train Outer Loop from big.txt → Save Checkpoint
=================================================

Reads your big.txt file (one text/sentence per line), trains the
outer-loop parameters (W_K, W_V, W_Q + encoder projection) so the
inner-loop MLP can memorise and retrieve effectively.

The saved checkpoint can then be loaded in Phase 2 (memory retrieval)
by any script that builds a TitansMemoryLayer and calls layer.load().

Usage:
    # Train on your corpus (one sentence per line in big.txt):
    python train_from_file.py --data big.txt

    # Customise hyperparams:
    python train_from_file.py --data big.txt \
        --outer-epochs 20 --d-model 128 --episode-len 16 --outer-lr 3e-4

    # Use GPU:
    python train_from_file.py --data big.txt --device cuda --verbose
"""

import argparse
import os
import time

from src.config import TitansConfig
from src.memory_layer import TitansMemoryLayer
from src.trainer import OuterLoopTrainer
from src.utils import set_seed


def load_corpus(path: str, min_len: int = 10) -> list[str]:
    """Load a text file — one meaningful sentence/paragraph per line.

    Lines shorter than `min_len` characters are dropped (headings, blanks).
    """
    with open(path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if len(line.strip()) >= min_len]
    return texts


def run_train(config: TitansConfig, data_path: str, checkpoint_path: str):
    """Train the outer-loop (W_K, W_V, W_Q) on a text corpus and save."""

    texts = load_corpus(data_path)
    if len(texts) < config.episode_len:
        raise ValueError(
            f"Corpus has only {len(texts)} usable lines but episode_len="
            f"{config.episode_len}.  Add more text or lower --episode-len."
        )

    print("=" * 65)
    print("  OUTER-LOOP TRAINING")
    print("=" * 65)
    print(f"  Corpus:             {data_path}  ({len(texts)} lines)")
    print(f"  d_model:            {config.d_model}")
    print(f"  MLP layers:         {config.memory_num_layers}")
    print(f"  Outer LR:           {config.outer_lr}")
    print(f"  Epochs:             {config.outer_epochs}")
    print(f"  Episode len:        {config.episode_len}")
    print(f"  Eval ratio:         {config.eval_ratio}")
    print(f"  Grad clip:          {config.grad_clip}")
    print(f"  Train encoder proj: {config.train_encoder_projection}")
    print(f"  Device:             {config.device}")
    print("=" * 65)

    # Build memory layer (encoder + memory module)
    layer = TitansMemoryLayer(config)

    # Create outer-loop trainer
    trainer = OuterLoopTrainer(
        config=config,
        encoder=layer.encoder,
        memory=layer.memory,
    )

    # Train
    t0 = time.time()
    epoch_losses = trainer.train(texts)
    elapsed = time.time() - t0

    # Summary
    print(f"\n{'─' * 55}")
    print("Training Summary:")
    print(f"{'─' * 55}")
    for i, loss in enumerate(epoch_losses):
        bar = "█" * int(min(loss * 100, 40))
        print(f"  Epoch {i + 1:3d}:  loss = {loss:.6f}  {bar}")

    improvement = ((epoch_losses[0] - epoch_losses[-1])
                   / max(epoch_losses[0], 1e-12) * 100)
    print(f"{'─' * 55}")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Loss improvement: {improvement:.1f}%")
    print(f"  Start loss:       {epoch_losses[0]:.6f}")
    print(f"  Final loss:       {epoch_losses[-1]:.6f}")

    # Save checkpoint
    layer.save(checkpoint_path)
    print(f"\n  Checkpoint saved -> {checkpoint_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Titans outer loop (W_K, W_V, W_Q) from a text corpus"
    )

    p.add_argument("--data", type=str, default="big.txt",
                   help="Path to training corpus, one text per line (default: big.txt)")
    p.add_argument("--checkpoint", type=str, default="trained_outer.pt",
                   help="Output checkpoint file path (default: trained_outer.pt)")

    # Model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--memory-num-layers", type=int, default=2)
    p.add_argument("--memory-hidden-dim", type=int, default=128)
    p.add_argument("--memory-lr", type=float, default=0.01,
                   help="Inner-loop LR theta_t (default: 0.01)")
    p.add_argument("--memory-momentum-decay", type=float, default=0.9)
    p.add_argument("--memory-forget-gate", type=float, default=0.02)

    # Outer-loop training
    p.add_argument("--outer-lr", type=float, default=3e-4,
                   help="Outer-loop LR for W_K, W_V, W_Q (default: 3e-4)")
    p.add_argument("--outer-epochs", type=int, default=10)
    p.add_argument("--episode-len", type=int, default=8,
                   help="Texts per training episode (default: 8)")
    p.add_argument("--eval-ratio", type=float, default=0.25)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Runtime
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Training corpus not found: {args.data}\n"
            f"Create a text file with one sentence per line."
        )

    config = TitansConfig(
        d_model=args.d_model,
        memory_num_layers=args.memory_num_layers,
        memory_hidden_dim=args.memory_hidden_dim,
        memory_lr=args.memory_lr,
        memory_momentum_decay=args.memory_momentum_decay,
        memory_forget_gate=args.memory_forget_gate,
        outer_lr=args.outer_lr,
        outer_epochs=args.outer_epochs,
        episode_len=args.episode_len,
        eval_ratio=args.eval_ratio,
        grad_clip=args.grad_clip,
        train_encoder_projection=True,
        device=args.device,
        verbose=args.verbose,
        seed=args.seed,
    )

    run_train(config, args.data, args.checkpoint)
    print("\nDone.")


if __name__ == "__main__":
    main()
