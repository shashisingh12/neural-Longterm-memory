"""
Interactive Memory Chat — Phase 2
===================================

Mimics a chat loop where:
    1. You type a question
    2. The memory layer retrieves relevant past context
    3. The enriched prompt (persistent + recalled memory + your question) is shown
    4. You manually type the answer (you are the LLM for now)
    5. The Q+A pair is written into the inner-loop MLP → memory updates
    6. Next question benefits from everything stored so far

Optionally pre-seeds big.txt into the MLP before chatting so the
memory already "knows" the corpus from turn 1.

Usage:
    # Chat with trained checkpoint (memory builds as you talk):
    python chat_memory.py --checkpoint trained_outer.pt

    # Pre-seed corpus into MLP first, then chat:
    python chat_memory.py --checkpoint trained_outer.pt --preseed big.txt

    # Verbose mode (shows surprise scores, timings):
    python chat_memory.py --checkpoint trained_outer.pt --preseed big.txt --verbose
"""

import argparse
import json
import os
import time

from src.config import TitansConfig
from src.memory_layer import TitansMemoryLayer
from src.utils import set_seed


def preseed_corpus(layer: TitansMemoryLayer, path: str, min_len: int = 10):
    """Write every line from a text file into the inner-loop MLP.

    After this, the MLP weights contain the corpus as memory and the
    decoder store has every line for cosine retrieval.
    """
    with open(path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if len(line.strip()) >= min_len]

    print(f"  Pre-seeding {len(texts)} lines into memory...")
    t0 = time.time()

    for i, text in enumerate(texts):
        x = layer.encoder.encode(text)
        surprise = layer.memory.write(x)          # inner-loop MLP update
        layer.decoder.register(x.detach(), text)   # store for cosine lookup

        if layer.verbose and i % 50 == 0:
            print(f"    [{i+1}/{len(texts)}]  surprise={surprise:.4f}  "
                  f"'{text[:60]}...'")

    elapsed = time.time() - t0
    print(f"  Pre-seeded {len(texts)} entries in {elapsed:.1f}s")
    print(f"  Decoder store: {len(layer.decoder)} entries\n")


def chat_loop(layer: TitansMemoryLayer):
    """Interactive chat with manual LLM input.

    Each turn:
        1. You enter a question.
        2. Memory retrieves context → enriched prompt is printed.
        3. You type the answer (acting as the LLM).
        4. Q+A is written into memory.
    """

    print("─" * 55)
    print("  MEMORY CHAT  (you provide both question AND answer)")
    print("─" * 55)
    print("  Commands:")
    print("    quit   — exit and save memory")
    print("    stats  — show memory diagnostics")
    print("    dump   — show full enriched prompt for last query")
    print()

    last_enriched = ""

    def manual_llm(enriched_prompt: str) -> str:
        """Instead of calling an LLM, show the context and ask the user."""
        nonlocal last_enriched
        last_enriched = enriched_prompt

        # Show what the memory layer injected
        print()
        print("┌─ ENRICHED PROMPT (sent to LLM) ─────────────────────")

        # Show persistent block (shortened)
        lines = enriched_prompt.split("\n")
        for line in lines:
            if line.startswith("[PERSISTENT]") or line.startswith("<SOS>"):
                print(f"│  {line[:80]}")

        # Show memory block
        in_memory = False
        for line in lines:
            if "RELEVANT LONG-TERM MEMORY" in line:
                in_memory = True
                print(f"│  {line}")
                continue
            if "END MEMORY" in line:
                print(f"│  {line}")
                in_memory = False
                continue
            if in_memory:
                print(f"│  {line[:90]}")

        # Show the user query line
        for line in lines:
            if line.startswith("User:"):
                print(f"│  {line}")

        print("└─────────────────────────────────────────────────────")
        print()

        # User manually provides the answer
        print("  Type your answer (the LLM response).  Press Enter to submit.")
        try:
            answer = input("  Answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        return answer if answer else "[no answer provided]"

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() == "quit":
            break

        if query.lower() == "stats":
            print(json.dumps(layer.stats(), indent=2))
            continue

        if query.lower() == "dump":
            if last_enriched:
                print("\n=== FULL ENRICHED PROMPT ===")
                print(last_enriched)
                print("=== END ===\n")
            else:
                print("  (no prompt yet — ask a question first)")
            continue

        # Run through the memory layer → calls manual_llm internally
        t0 = time.time()
        response = layer.run(query, manual_llm)
        elapsed = time.time() - t0

        print(f"\n  [Memory updated | turn {layer._turn} | "
              f"surprise logged | {elapsed:.2f}s]")
        print()


def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive memory chat — Phase 2 retrieval with manual LLM input"
    )

    p.add_argument("--checkpoint", type=str, default="trained_outer.pt",
                   help="Trained checkpoint from train_from_file.py (default: trained_outer.pt)")
    p.add_argument("--preseed", type=str, default="",
                   help="Path to corpus file to pre-seed into MLP before chatting (e.g. big.txt)")
    p.add_argument("--save-on-exit", type=str, default="",
                   help="Save updated memory on exit (default: <checkpoint>_chat.pt)")

    # Model (must match the checkpoint)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--memory-num-layers", type=int, default=2)
    p.add_argument("--memory-hidden-dim", type=int, default=128)
    p.add_argument("--memory-lr", type=float, default=0.01)
    p.add_argument("--memory-momentum-decay", type=float, default=0.9)
    p.add_argument("--memory-forget-gate", type=float, default=0.02)
    p.add_argument("--top-k", type=int, default=3,
                   help="Memory snippets to retrieve per query (default: 3)")

    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    config = TitansConfig(
        d_model=args.d_model,
        memory_num_layers=args.memory_num_layers,
        memory_hidden_dim=args.memory_hidden_dim,
        memory_lr=args.memory_lr,
        memory_momentum_decay=args.memory_momentum_decay,
        memory_forget_gate=args.memory_forget_gate,
        top_k=args.top_k,
        device=args.device,
        verbose=args.verbose,
        seed=args.seed,
    )

    print("=" * 55)
    print("  Titans Memory Chat — Phase 2")
    print("=" * 55)

    # Build layer and load trained outer-loop weights
    layer = TitansMemoryLayer(config)

    if os.path.exists(args.checkpoint):
        layer.load(args.checkpoint)
        print(f"  Loaded checkpoint <- {args.checkpoint}")
    else:
        print(f"  WARNING: {args.checkpoint} not found, using untrained weights")

    # Persistent context
    layer.persistent.add("You are a knowledgeable assistant with perfect recall.")
    layer.persistent.add("Use the retrieved memory context to answer accurately.")

    # Pre-seed corpus if provided
    if args.preseed:
        if os.path.exists(args.preseed):
            preseed_corpus(layer, args.preseed)
        else:
            print(f"  WARNING: preseed file {args.preseed} not found, skipping")

    # Run interactive chat
    chat_loop(layer)

    # Save updated memory
    save_path = args.save_on_exit
    if not save_path:
        save_path = args.checkpoint.replace(".pt", "_chat.pt")
    layer.save(save_path)
    print(f"  Memory saved -> {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()
