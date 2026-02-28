"""
Chat Joint — Phase 2 Interactive Inference
=============================================
Load a jointly-trained checkpoint and run interactive conversation.

Only M_t surprise writes + dynamic alpha are active.
Everything else (LLM, LoRA, adapter) is frozen.

Usage:
    python chat_joint.py --checkpoint joint_checkpoint.pt
    python chat_joint.py --checkpoint joint_checkpoint.pt --verbose
    python chat_joint.py --checkpoint joint_checkpoint.pt --device cuda

Commands during chat:
    /reset    — clear M_t, start fresh conversation
    /stats    — print memory diagnostics
    /quit     — exit
"""

import argparse
import sys
import time

import torch

from src.joint_config import JointTrainingConfig
from src.memory_llm import MemoryLLM
from src.joint_inference import Phase2Inference
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 — Interactive inference with memory"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to jointly-trained checkpoint (.pt)"
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Override model name (default: from checkpoint config)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # load config from checkpoint
    state = torch.load(
        args.checkpoint,
        map_location=args.device,
        weights_only=False,
    )
    config = JointTrainingConfig.from_dict(state["config"])
    config.device = args.device
    config.verbose = args.verbose

    if args.model_name:
        config.model_name = args.model_name

    print("=" * 65)
    print("  PHASE 2 — INFERENCE WITH MEMORY")
    print("=" * 65)
    print(f"  Model:       {config.model_name}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Device:      {config.device}")
    print(f"  d_mem:       {config.d_mem}")
    print(f"  LoRA rank:   {config.lora_rank}")
    print("=" * 65)

    # build model and load weights
    print("\n  Loading model...")
    t0 = time.time()
    model = MemoryLLM(config)
    model.load_checkpoint(args.checkpoint)
    model.freeze_for_inference()
    print(f"  Ready in {time.time() - t0:.1f}s")

    # create inference session
    session = Phase2Inference(model, config)

    print("\n  Chat started. Type /quit to exit, /reset to clear memory.")
    print(f"  Verbose: {args.verbose}")
    print(f"  Temperature: {args.temperature}")
    print("-" * 65)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("  Goodbye.")
            break

        if user_input.lower() == "/reset":
            session.reset()
            print("  [Memory cleared. Starting fresh conversation.]")
            continue

        if user_input.lower() == "/stats":
            stats = session.memory_stats()
            print(f"  Turns: {stats['turns']}")
            print(f"  Alpha mean:    {stats['alpha_mean']:.4f}")
            print(f"  Surprise mean: {stats['surprise_mean']:.4f}")
            if stats['alpha_history']:
                print(f"  Alpha history:   {stats['alpha_history']}")
                print(f"  Surprise history: {stats['surprise_history']}")
            continue

        # generate response
        t0 = time.time()
        response = session.chat(
            user_input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0

        print(f"\n  AI: {response}")
        if args.verbose:
            print(f"  [{elapsed:.2f}s]")


if __name__ == "__main__":
    main()
