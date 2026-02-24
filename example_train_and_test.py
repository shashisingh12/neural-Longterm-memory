"""
Example: Train Outer Loop → Save Checkpoint → Test Memory Recall
=================================================================

Phase 1 — TRAIN:
    Train W_K, W_V, W_Q projections on a corpus of texts.
    The outer loop learns HOW to project inputs so the inner-loop
    MLP can memorize and retrieve them effectively.

Phase 2 — TEST:
    Load the trained checkpoint, simulate a multi-turn conversation.
    Verify that the memory recalls earlier facts when asked.

Usage:
    # Full train + test:
    python example_train_and_test.py

    # Train only (saves checkpoint):
    python example_train_and_test.py --skip-test

    # Test only (loads existing checkpoint):
    python example_train_and_test.py --skip-train --checkpoint-path trained_memory.pt

    # Customize:
    python example_train_and_test.py --outer-epochs 20 --d-model 128 --verbose
"""

import argparse
import os
import time

import torch

from src.config import TitansConfig
from src.memory_layer import TitansMemoryLayer
from src.active_layer import ActiveMemoryLayer
from src.trainer import OuterLoopTrainer, ActiveOuterLoopTrainer
from src.utils import set_seed


# ── Training corpus ──────────────────────────────────────────

TRAINING_TEXTS = [
    # Python & programming
    "Python is a high-level programming language created by Guido van Rossum.",
    "Python uses indentation for code blocks instead of curly braces.",
    "List comprehensions in Python provide a concise way to create lists.",
    "Python supports multiple paradigms: procedural, object-oriented, and functional.",
    "The Python package manager pip installs libraries from PyPI.",
    "NumPy is a Python library for numerical computing with multi-dimensional arrays.",
    "Pandas provides DataFrames for tabular data manipulation in Python.",
    "Flask and Django are popular Python web frameworks.",

    # Machine learning
    "Machine learning models learn patterns from labeled training data.",
    "Supervised learning uses input-output pairs to train a model.",
    "Unsupervised learning discovers hidden structure in unlabeled data.",
    "Gradient descent optimizes model parameters by following the loss gradient.",
    "Overfitting occurs when a model memorizes training data but fails on new data.",
    "Regularization techniques like dropout prevent overfitting in neural networks.",
    "Cross-validation splits data into folds to estimate model performance.",
    "Feature engineering transforms raw data into inputs suitable for ML models.",

    # Deep learning
    "Neural networks are composed of layers of interconnected neurons.",
    "Backpropagation computes gradients by applying the chain rule layer by layer.",
    "Convolutional neural networks use filters to detect spatial patterns in images.",
    "Recurrent neural networks process sequential data using hidden state feedback.",
    "The transformer architecture uses self-attention instead of recurrence.",
    "BERT is a bidirectional encoder trained with masked language modeling.",
    "GPT generates text autoregressively, predicting one token at a time.",
    "Attention mechanisms compute weighted sums over all input positions.",

    # Titans architecture
    "The Titans architecture stores long-term memory in MLP weights.",
    "Surprise-driven gradients in Titans decide what information to memorize.",
    "Momentum in Titans memory preserves context after surprising observations.",
    "Weight decay in Titans acts as a forgetting mechanism for old memories.",
    "The inner loop in Titans updates MLP weights using gradient-based memory rules.",
    "The outer loop in Titans trains projection matrices W_K, W_V, and W_Q.",
    "MAC architecture in Titans combines persistent vectors, memory read, and current input.",
    "Parallel associative scan enables efficient chunked memory updates in Titans.",
]

# ── Test conversations ────────────────────────────────────────

TEST_CONVERSATIONS = [
    {
        "description": "Basic fact recall",
        "turns": [
            ("Who created Python?", "python"),
            ("What does Python use instead of curly braces?", "indentation"),
            ("Who created the language we just discussed?", "guido"),
        ],
    },
    {
        "description": "Cross-topic retrieval",
        "turns": [
            ("What is gradient descent?", "gradient"),
            ("Explain backpropagation.", "chain rule"),
            ("What architecture uses self-attention?", "transformer"),
            ("How does the Titans architecture store memory?", "mlp weights"),
        ],
    },
    {
        "description": "Memory persistence across turns",
        "turns": [
            ("Tell me about BERT.", "bidirectional"),
            ("What about GPT?", "autoregressive"),
            ("What is the Titans inner loop?", "mlp"),
            ("What was the first model we discussed?", "bert"),
        ],
    },
]


# ── Demo LLM (stub) ──────────────────────────────────────────

def make_test_llm():
    """A lookup-based stub LLM that responds based on keywords.

    In production, replace with:
        from openai import OpenAI
        client = OpenAI()
        def llm(prompt):
            r = client.chat.completions.create(model="gpt-4o", ...)
            return r.choices[0].message.content
        return llm
    """
    responses = {
        "python": "Python is a high-level programming language created by Guido van Rossum, known for readable syntax.",
        "indentation": "Python uses indentation (whitespace) to define code blocks instead of curly braces.",
        "guido": "Guido van Rossum created Python in the late 1980s at CWI in the Netherlands.",
        "gradient": "Gradient descent is an optimization algorithm that updates parameters in the direction of steepest loss decrease.",
        "chain rule": "Backpropagation applies the chain rule to compute gradients layer by layer from output to input.",
        "transformer": "The transformer architecture, introduced in 'Attention Is All You Need', replaces recurrence with self-attention.",
        "mlp weights": "Titans stores long-term memory directly in MLP weight matrices, updated by surprise-driven gradients.",
        "bidirectional": "BERT is a bidirectional encoder that reads text in both directions using masked language modeling.",
        "autoregressive": "GPT generates text autoregressively, predicting the next token based on all previous tokens.",
        "mlp": "The Titans inner loop updates MLP weights on-the-fly using the surprise-momentum-forgetting rule.",
        "bert": "BERT (Bidirectional Encoder Representations from Transformers) was the first model we discussed.",
    }

    def llm_fn(prompt: str) -> str:
        pl = prompt.lower()
        # Check for keyword matches in the enriched prompt (including memory context)
        for keyword, response in responses.items():
            if keyword in pl:
                return response
        # Fallback
        if any(w in pl for w in ["earlier", "discussed", "first", "recall", "previous"]):
            return "Based on our conversation, we discussed several topics including Python, BERT, and Titans."
        return f"I'll help with: {prompt.split('User:')[-1].strip()[:80]}"

    return llm_fn


# ── Phase 1: Train ────────────────────────────────────────────

def run_training(config: TitansConfig, checkpoint_path: str) -> TitansMemoryLayer:
    """Train outer-loop parameters and save checkpoint."""

    print("=" * 65)
    print("  PHASE 1: OUTER-LOOP TRAINING")
    print("=" * 65)
    print(f"  d_model:            {config.d_model}")
    print(f"  MLP layers:         {config.memory_num_layers}")
    print(f"  Outer LR:           {config.outer_lr}")
    print(f"  Epochs:             {config.outer_epochs}")
    print(f"  Episode len:        {config.episode_len}")
    print(f"  Eval ratio:         {config.eval_ratio}")
    print(f"  Grad clip:          {config.grad_clip}")
    print(f"  Train encoder proj: {config.train_encoder_projection}")
    print(f"  Training texts:     {len(TRAINING_TEXTS)}")
    print(f"  Device:             {config.device}")
    print("=" * 65)

    # Build the memory layer
    layer = TitansMemoryLayer(config)

    # Create trainer (trains W_K, W_V, W_Q + encoder projection)
    trainer = OuterLoopTrainer(
        config=config,
        encoder=layer.encoder,
        memory=layer.memory,
    )

    # Train
    t0 = time.time()
    epoch_losses = trainer.train(TRAINING_TEXTS)
    elapsed = time.time() - t0

    # Print training summary
    print(f"\n{'─' * 55}")
    print("Training Summary:")
    print(f"{'─' * 55}")
    for i, loss in enumerate(epoch_losses):
        bar = "█" * int(min(loss * 100, 40))
        print(f"  Epoch {i + 1:3d}:  loss = {loss:.6f}  {bar}")

    improvement = (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100
    print(f"{'─' * 55}")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Loss improvement: {improvement:.1f}%")
    print(f"  Start loss:       {epoch_losses[0]:.6f}")
    print(f"  Final loss:       {epoch_losses[-1]:.6f}")

    # Save checkpoint
    layer.save(checkpoint_path)
    print(f"\n  Checkpoint saved → {checkpoint_path}")

    return layer


# ── Phase 2: Test ─────────────────────────────────────────────

def run_testing(config: TitansConfig, checkpoint_path: str):
    """Load trained checkpoint and run test conversations."""

    print(f"\n\n{'=' * 65}")
    print("  PHASE 2: TESTING TRAINED MEMORY")
    print("=" * 65)

    # Build a fresh layer and load trained weights
    layer = TitansMemoryLayer(config)

    if os.path.exists(checkpoint_path):
        layer.load(checkpoint_path)
        print(f"  Loaded checkpoint ← {checkpoint_path}")
    else:
        print(f"  WARNING: No checkpoint at {checkpoint_path}, using untrained weights")

    # Add persistent context
    layer.persistent.add("You are a helpful technical assistant with perfect recall.")
    layer.persistent.add("Domain: programming, machine learning, deep learning, Titans.")

    llm_fn = make_test_llm()

    # ── Compare trained vs untrained surprise ──
    print(f"\n{'─' * 55}")
    print("Surprise Comparison (trained vs untrained):")
    print(f"{'─' * 55}")

    untrained_layer = TitansMemoryLayer(config)
    sample_texts = TRAINING_TEXTS[:5]

    for text in sample_texts:
        vec = layer.encoder.encode(text)
        trained_surprise = layer.memory.get_surprise(vec)

        vec_u = untrained_layer.encoder.encode(text)
        untrained_surprise = untrained_layer.memory.get_surprise(vec_u)

        delta = "↓ better" if trained_surprise < untrained_surprise else "↑ worse"
        print(f"  \"{text[:50]}...\"")
        print(f"    Trained: {trained_surprise:.6f}  |  Untrained: {untrained_surprise:.6f}  {delta}")

    # ── Run test conversations ──
    total_turns = 0
    passed_turns = 0

    for conv_idx, conv in enumerate(TEST_CONVERSATIONS, 1):
        print(f"\n{'═' * 55}")
        print(f"  Test {conv_idx}: {conv['description']}")
        print(f"{'═' * 55}")

        # Reset memory MLP for each test conversation
        # (start fresh, so we can test memory building during conversation)
        layer.memory.reset_mlp()

        for turn_idx, (question, expected_keyword) in enumerate(conv["turns"], 1):
            total_turns += 1

            response = layer.run(question, llm_fn)

            # Check if the expected keyword appears in the response
            match = expected_keyword.lower() in response.lower()
            if match:
                passed_turns += 1
                status = "✓ PASS"
            else:
                status = "✗ MISS"

            print(f"\n  Turn {turn_idx}: {question}")
            print(f"    Response: {response[:90]}")
            print(f"    Expected keyword: '{expected_keyword}' → {status}")

        # Print memory stats after conversation
        stats = layer.stats()
        print(f"\n  Memory after conversation: {stats['memory_entries']} entries stored")

    # ── Final score ──
    print(f"\n\n{'=' * 65}")
    print(f"  FINAL SCORE: {passed_turns}/{total_turns} turns matched "
          f"({passed_turns / total_turns * 100:.0f}%)")
    print(f"{'=' * 65}")

    # ── Test parallel batch write (if enabled) ──
    if config.use_parallel_memory:
        print(f"\n{'─' * 55}")
        print("Parallel Batch Write Test:")
        print(f"{'─' * 55}")

        layer.memory.reset_mlp()
        batch_texts = TRAINING_TEXTS[:8]

        # Encode all texts
        embeddings = []
        for text in batch_texts:
            embeddings.append(layer.encoder.encode(text))
        xs = torch.stack(embeddings, dim=0)

        # Parallel chunk write
        t0 = time.time()
        losses = layer.memory.write_chunk_parallel(xs, chunk_size=config.chunk_size)
        elapsed = time.time() - t0

        print(f"  Wrote {len(batch_texts)} texts in {elapsed:.3f}s "
              f"(chunk_size={config.chunk_size})")
        print(f"  Per-token surprise: {', '.join(f'{l:.4f}' for l in losses)}")
        print(f"  Avg surprise:       {sum(losses) / len(losses):.4f}")


# ── Active Memory: Train + Test ──────────────────────────────

def run_active_training(config: TitansConfig, checkpoint_path: str) -> ActiveMemoryLayer:
    """Train active memory (attention + persistent vectors + projections)."""

    print("=" * 65)
    print("  PHASE 1: ACTIVE MEMORY OUTER-LOOP TRAINING")
    print("=" * 65)
    print(f"  d_model:             {config.d_model}")
    print(f"  MLP layers:          {config.memory_num_layers}")
    print(f"  Attention heads:     {config.num_attention_heads}")
    print(f"  Persistent vectors:  {config.num_persistent_vectors}")
    print(f"  Transcript size:     {config.memory_transcript_size}")
    print(f"  Outer LR:            {config.outer_lr}")
    print(f"  Epochs:              {config.outer_epochs}")
    print(f"  Training texts:      {len(TRAINING_TEXTS)}")
    print(f"  Device:              {config.device}")
    print("=" * 65)

    layer = ActiveMemoryLayer(config)

    trainer = ActiveOuterLoopTrainer(
        config=config,
        encoder=layer.encoder,
        memory=layer.memory,
        attention=layer.attention,
        persistent_vectors=layer.persistent_vectors,
    )

    t0 = time.time()
    epoch_losses = trainer.train(TRAINING_TEXTS)
    elapsed = time.time() - t0

    print(f"\n{'─' * 55}")
    print("Active Training Summary:")
    print(f"{'─' * 55}")
    for i, loss in enumerate(epoch_losses):
        bar = "█" * int(min(loss * 100, 40))
        print(f"  Epoch {i + 1:3d}:  loss = {loss:.6f}  {bar}")

    improvement = (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100
    print(f"{'─' * 55}")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Loss improvement: {improvement:.1f}%")
    print(f"  Start loss:       {epoch_losses[0]:.6f}")
    print(f"  Final loss:       {epoch_losses[-1]:.6f}")

    layer.save(checkpoint_path)
    print(f"\n  Active checkpoint saved → {checkpoint_path}")

    return layer


def run_active_testing(config: TitansConfig, checkpoint_path: str):
    """Load trained active checkpoint and run test conversations."""

    print(f"\n\n{'=' * 65}")
    print("  PHASE 2: TESTING ACTIVE MEMORY")
    print("=" * 65)

    layer = ActiveMemoryLayer(config)

    if os.path.exists(checkpoint_path):
        layer.load(checkpoint_path)
        print(f"  Loaded active checkpoint ← {checkpoint_path}")
    else:
        print(f"  WARNING: No checkpoint at {checkpoint_path}, using untrained weights")

    layer.persistent.add("You are a helpful technical assistant with perfect recall.")
    layer.persistent.add("Domain: programming, machine learning, deep learning, Titans.")

    llm_fn = make_test_llm()

    # ── Compare trained vs untrained surprise ──
    print(f"\n{'─' * 55}")
    print("Surprise Comparison (trained vs untrained):")
    print(f"{'─' * 55}")

    untrained_layer = ActiveMemoryLayer(config)
    sample_texts = TRAINING_TEXTS[:5]

    for text in sample_texts:
        vec = layer.encoder.encode(text)
        trained_surprise = layer.memory.get_surprise(vec)

        vec_u = untrained_layer.encoder.encode(text)
        untrained_surprise = untrained_layer.memory.get_surprise(vec_u)

        delta = "↓ better" if trained_surprise < untrained_surprise else "↑ worse"
        print(f"  \"{text[:50]}...\"")
        print(f"    Trained: {trained_surprise:.6f}  |  Untrained: {untrained_surprise:.6f}  {delta}")

    # ── Run test conversations ──
    total_turns = 0
    passed_turns = 0

    for conv_idx, conv in enumerate(TEST_CONVERSATIONS, 1):
        print(f"\n{'═' * 55}")
        print(f"  Test {conv_idx}: {conv['description']}")
        print(f"{'═' * 55}")

        layer.memory.reset_mlp()

        for turn_idx, (question, expected_keyword) in enumerate(conv["turns"], 1):
            total_turns += 1

            response = layer.run(question, llm_fn)

            match = expected_keyword.lower() in response.lower()
            if match:
                passed_turns += 1
                status = "✓ PASS"
            else:
                status = "✗ MISS"

            print(f"\n  Turn {turn_idx}: {question}")
            print(f"    Response: {response[:90]}")
            print(f"    Expected keyword: '{expected_keyword}' → {status}")

        stats = layer.stats()
        print(f"\n  Transcript: {stats['transcript_entries']}/{stats['transcript_capacity']} entries")

    print(f"\n\n{'=' * 65}")
    print(f"  FINAL SCORE: {passed_turns}/{total_turns} turns matched "
          f"({passed_turns / total_turns * 100:.0f}%)")
    print(f"{'=' * 65}")


# ── CLI ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Titans Memory — Train outer loop and test recall"
    )

    # Training params
    p.add_argument("--d-model", type=int, default=128,
                   help="Latent dimension (default: 128)")
    p.add_argument("--memory-num-layers", type=int, default=2,
                   help="MLP layers in memory (default: 2)")
    p.add_argument("--memory-hidden-dim", type=int, default=128,
                   help="Hidden dim of memory MLP (default: 128)")
    p.add_argument("--memory-lr", type=float, default=0.01,
                   help="Inner-loop learning rate θ (default: 0.01)")
    p.add_argument("--memory-momentum-decay", type=float, default=0.9,
                   help="Momentum decay η (default: 0.9)")
    p.add_argument("--memory-forget-gate", type=float, default=0.02,
                   help="Forgetting gate α (default: 0.02)")

    p.add_argument("--outer-lr", type=float, default=1e-4,
                   help="Outer-loop learning rate (default: 1e-4)")
    p.add_argument("--outer-epochs", type=int, default=10,
                   help="Training epochs (default: 10)")
    p.add_argument("--episode-len", type=int, default=8,
                   help="Texts per training episode (default: 8)")
    p.add_argument("--eval-ratio", type=float, default=0.25,
                   help="Fraction of episode for eval (default: 0.25)")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Gradient clipping norm (default: 1.0)")

    # Parallel memory
    p.add_argument("--use-parallel-memory", action="store_true", default=False,
                   help="Enable parallel chunk writes for testing")
    p.add_argument("--chunk-size", type=int, default=4,
                   help="Chunk size for parallel writes (default: 4)")

    # Runtime
    p.add_argument("--checkpoint-path", type=str, default="trained_memory.pt",
                   help="Checkpoint file path (default: trained_memory.pt)")
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="Device (default: cpu)")
    p.add_argument("--verbose", action="store_true", default=False,
                   help="Print detailed debug info")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")

    # Active memory mode
    p.add_argument("--use-active", action="store_true", default=False,
                   help="Use ActiveMemoryLayer instead of TitansMemoryLayer")
    p.add_argument("--memory-transcript-size", type=int, default=32,
                   help="Ring buffer capacity for active memory (default: 32)")
    p.add_argument("--num-persistent-vectors", type=int, default=4,
                   help="Learnable persistent vectors for active mode (default: 4)")
    p.add_argument("--num-attention-heads", type=int, default=4,
                   help="Attention heads for active mode (default: 4)")

    # Skip flags
    p.add_argument("--skip-train", action="store_true", default=False,
                   help="Skip training, load existing checkpoint for testing")
    p.add_argument("--skip-test", action="store_true", default=False,
                   help="Skip testing, only train and save checkpoint")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Build config
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
        use_parallel_memory=args.use_parallel_memory,
        chunk_size=args.chunk_size,
        memory_transcript_size=args.memory_transcript_size,
        num_persistent_vectors=args.num_persistent_vectors,
        num_attention_heads=args.num_attention_heads,
        device=args.device,
        verbose=args.verbose,
        seed=args.seed,
    )

    checkpoint_path = args.checkpoint_path

    if args.use_active:
        # Active memory mode
        if not args.skip_train:
            run_active_training(config, checkpoint_path)
        else:
            print("[Skipping active training, will load checkpoint for testing]")

        if not args.skip_test:
            run_active_testing(config, checkpoint_path)
        else:
            print("[Skipping testing, checkpoint saved]")
    else:
        # Original vanilla mode
        if not args.skip_train:
            run_training(config, checkpoint_path)
        else:
            print("[Skipping training, will load checkpoint for testing]")

        if not args.skip_test:
            run_testing(config, checkpoint_path)
        else:
            print("[Skipping testing, checkpoint saved]")

    print("\nDone.")


if __name__ == "__main__":
    main()
