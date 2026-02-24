"""
Titans Memory — CLI Entry Point
=================================
Five modes:

    # Train outer-loop (W_K, W_V, W_Q) on a text corpus:
    python main.py --mode train --train-data-path data.txt --outer-epochs 10 --verbose

    # Simple inference demo (inner-loop only):
    python main.py --mode demo --verbose --checkpoint-path checkpoint.pt

    # MAC architecture demo (Memory as Context, Paper §4.1):
    python main.py --mode mac --verbose --num-persistent-vectors 4 --num-attention-heads 4

    # Active memory: train (attention + persistent vectors + projections):
    python main.py --mode train-active --outer-epochs 10 --verbose

    # Active memory: inference demo (MLP-driven context, no RAG):
    python main.py --mode active --verbose --checkpoint-path active_checkpoint.pt

All hyperparameters are configurable via CLI. See:
    python main.py --help
"""

import os
import json

from src.config import TitansConfig, build_parser
from src.memory_layer import TitansMemoryLayer
from src.mac_layer import MACMemoryLayer
from src.active_layer import ActiveMemoryLayer
from src.trainer import OuterLoopTrainer, ActiveOuterLoopTrainer
from src.utils import set_seed


# ── Argument setup ────────────────────────────────────────

def make_parser():
    """Extend the base parser with a --mode flag."""
    parser = build_parser()
    parser.add_argument("--mode", type=str, default="demo",
                        choices=["train", "demo", "mac", "active", "train-active"],
                        help="'train' = outer-loop training, 'demo' = simple inference, "
                             "'mac' = MAC architecture, 'active' = active memory inference, "
                             "'train-active' = train active memory")
    return parser


# ── Training mode ─────────────────────────────────────────

def generate_synthetic_train_data() -> list:
    """Generate synthetic training texts for demo purposes.

    In production, replace with --train-data-path pointing to a real corpus.
    """
    return [
        "Python is a high-level programming language used in data science.",
        "Machine learning models learn patterns from data.",
        "Neural networks are inspired by biological neurons.",
        "Gradient descent optimizes model parameters by minimizing loss.",
        "Transformers use self-attention for sequence modeling.",
        "BERT is a bidirectional encoder for language understanding.",
        "GPT generates text using autoregressive decoding.",
        "Reinforcement learning trains agents through reward signals.",
        "Convolutional neural networks excel at image recognition.",
        "Recurrent neural networks handle sequential data.",
        "The Titans architecture uses MLP weights as long-term memory.",
        "Surprise-driven gradients decide what to memorize.",
        "Momentum in neural memory preserves context after surprises.",
        "Weight decay in memory acts as a forgetting mechanism.",
        "Associative memory maps keys to values for fast retrieval.",
        "Meta-learning trains models to learn quickly from few examples.",
        "The inner loop updates task-specific parameters.",
        "The outer loop learns shared initialization across tasks.",
        "Attention mechanisms weigh the importance of input tokens.",
        "Embeddings represent discrete tokens as continuous vectors.",
        "Transfer learning applies pretrained models to new tasks.",
        "Fine-tuning adapts a pretrained model with domain-specific data.",
        "Tokenizers convert text into numerical sequences for models.",
        "Loss functions measure the gap between prediction and target.",
        "Batch normalization stabilizes training by normalizing activations.",
        "Dropout regularization randomly zeroes activations during training.",
        "Learning rate schedules adjust the step size during optimization.",
        "Adam optimizer combines momentum with adaptive learning rates.",
        "Backpropagation computes gradients by applying the chain rule.",
        "The softmax function converts logits into probabilities.",
        "Cross-entropy loss is standard for classification tasks.",
        "Mean squared error loss is used for regression problems.",
    ]


def run_train(config: TitansConfig):
    """Run outer-loop training."""
    print("=" * 60)
    print("  Titans Memory  |  Outer-Loop Training")
    print("=" * 60)
    print(f"  Outer LR:       {config.outer_lr}")
    print(f"  Epochs:         {config.outer_epochs}")
    print(f"  Episode len:    {config.episode_len}")
    print(f"  Eval ratio:     {config.eval_ratio}")
    print(f"  Grad clip:      {config.grad_clip}")
    print(f"  Train encoder:  {config.train_encoder_projection}")
    print(f"  Device:         {config.device}")
    print("=" * 60)

    # Build memory layer (encoder + memory module)
    layer = TitansMemoryLayer(config)

    # Load training data
    if config.train_data_path and os.path.exists(config.train_data_path):
        with open(config.train_data_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"[Loaded {len(texts)} texts from {config.train_data_path}]")
    else:
        texts = generate_synthetic_train_data()
        print(f"[Using {len(texts)} synthetic training texts]")

    # Create trainer
    trainer = OuterLoopTrainer(
        config=config,
        encoder=layer.encoder,
        memory=layer.memory,
    )

    # Train
    epoch_losses = trainer.train(texts)

    # Print summary
    print(f"\n{'=' * 55}")
    print("Training Complete:")
    for i, loss in enumerate(epoch_losses):
        print(f"  Epoch {i+1:3d}:  loss = {loss:.6f}")

    # Save checkpoint with trained W_K, W_V, W_Q
    if config.checkpoint_path:
        layer.save(config.checkpoint_path)
        print(f"\n[Saved trained checkpoint to {config.checkpoint_path}]")
    else:
        print("\n[No --checkpoint-path provided, trained weights not saved]")


# ── Demo / Inference mode ─────────────────────────────────

def make_demo_llm():
    """Return a simple stub LLM for demonstration.

    Replace this with your real LLM, e.g.:

        # OpenAI
        from openai import OpenAI
        client = OpenAI()
        def llm(prompt):
            r = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content
        return llm
    """

    def demo_llm(prompt: str) -> str:
        pl = prompt.lower()
        if "python" in pl:
            return "Python is a high-level, readable programming language, popular in AI/ML."
        if "titans" in pl:
            return ("Titans is a neural architecture using a deep MLP as long-term "
                    "memory, updated by surprise-driven gradients.")
        if "memory" in pl and "titans" not in pl:
            return "Computer memory stores data; RAM is fast and volatile, disk is slow and persistent."
        if any(w in pl for w in ["earlier", "mentioned", "recall", "discussed", "past"]):
            return "Based on the context provided, we discussed Python and the Titans architecture."
        return f"I understand your question about: {prompt[:60]}"

    return demo_llm


def run_demo(config: TitansConfig):
    """Run inference demo."""
    print("=" * 60)
    print("  Titans External Memory Layer (PyTorch)  |  Demo")
    print("=" * 60)
    print(f"  Device:       {config.device}")
    print(f"  d_model:      {config.d_model}")
    print(f"  MLP layers:   {config.memory_num_layers}")
    print(f"  Tokenizer:    {config.tokenizer_name}")
    print(f"  Pooling:      {config.pooling_strategy}")
    print(f"  Memory LR:    {config.memory_lr}")
    print(f"  Momentum:     {config.memory_momentum_decay}")
    print(f"  Forget gate:  {config.memory_forget_gate}")
    print(f"  Top-K:        {config.top_k}")
    print("=" * 60)

    layer = TitansMemoryLayer(config)

    # Add persistent context
    if config.persistent_tokens:
        for token in config.persistent_tokens:
            layer.persistent.add(token)
    else:
        layer.persistent.add("You are a helpful technical assistant.")
        layer.persistent.add("Domain: AI, software engineering, computer science.")

    # Load checkpoint (includes trained W_K, W_V, W_Q if trained)
    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        layer.load(config.checkpoint_path)
        print(f"[Loaded checkpoint from {config.checkpoint_path}]")

    llm_fn = make_demo_llm()

    questions = [
        "What is Python?",
        "Explain the Titans architecture briefly.",
        "How does long-term memory work in computers?",
        "What programming language did we talk about earlier?",
        "Which architecture uses gradient-based memory updates?",
    ]

    for q in questions:
        print(f"\n{'─' * 55}")
        print(f"USER: {q}")
        r = layer.run(q, llm_fn)
        print(f"LLM : {r}")

    print(f"\n{'=' * 55}")
    print("Memory Stats:")
    print(json.dumps(layer.stats(), indent=2))

    if config.checkpoint_path:
        layer.save(config.checkpoint_path)
        print(f"\n[Saved checkpoint to {config.checkpoint_path}]")


# ── MAC (Memory as Context) mode ──────────────────────────

def run_mac(config: TitansConfig):
    """Run MAC architecture demo (Paper §4.1)."""
    print("=" * 60)
    print("  Titans MAC (Memory as Context)  |  Demo")
    print("=" * 60)
    print(f"  Device:             {config.device}")
    print(f"  d_model:            {config.d_model}")
    print(f"  MLP layers:         {config.memory_num_layers}")
    print(f"  Attention heads:    {config.num_attention_heads}")
    print(f"  Persistent vectors: {config.num_persistent_vectors}")
    print(f"  Memory LR:          {config.memory_lr}")
    print(f"  Forget gate:        {config.memory_forget_gate}")
    print(f"  Top-K:              {config.top_k}")
    print("=" * 60)

    layer = MACMemoryLayer(config)

    # Text persistent context (for prompt building)
    if config.persistent_tokens:
        for token in config.persistent_tokens:
            layer.persistent.add(token)
    else:
        layer.persistent.add("You are a helpful technical assistant.")
        layer.persistent.add("Domain: AI, software engineering, computer science.")

    # Load checkpoint if exists
    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        layer.load(config.checkpoint_path)
        print(f"[Loaded MAC checkpoint from {config.checkpoint_path}]")

    llm_fn = make_demo_llm()

    questions = [
        "What is Python?",
        "Explain the Titans architecture briefly.",
        "How does long-term memory work in computers?",
        "What programming language did we talk about earlier?",
        "Which architecture uses gradient-based memory updates?",
    ]

    for q in questions:
        print(f"\n{'─' * 55}")
        print(f"USER: {q}")
        r = layer.run(q, llm_fn)
        print(f"LLM : {r}")

    print(f"\n{'=' * 55}")
    print("MAC Stats:")
    print(json.dumps(layer.stats(), indent=2))

    if config.checkpoint_path:
        layer.save(config.checkpoint_path)
        print(f"\n[Saved MAC checkpoint to {config.checkpoint_path}]")


# ── Active Memory mode ────────────────────────────────────

def run_train_active(config: TitansConfig):
    """Train the active memory system (attention + persistent vectors + projections)."""
    print("=" * 60)
    print("  Titans Active Memory  |  Outer-Loop Training")
    print("=" * 60)
    print(f"  Outer LR:            {config.outer_lr}")
    print(f"  Epochs:              {config.outer_epochs}")
    print(f"  Episode len:         {config.episode_len}")
    print(f"  Eval ratio:          {config.eval_ratio}")
    print(f"  Grad clip:           {config.grad_clip}")
    print(f"  Train encoder:       {config.train_encoder_projection}")
    print(f"  Train attention:     {config.train_attention}")
    print(f"  Train persistent:    {config.train_persistent_vectors}")
    print(f"  Attention heads:     {config.num_attention_heads}")
    print(f"  Persistent vectors:  {config.num_persistent_vectors}")
    print(f"  Transcript size:     {config.memory_transcript_size}")
    print(f"  Device:              {config.device}")
    print("=" * 60)

    # Build active memory layer
    layer = ActiveMemoryLayer(config)

    # Load training data
    if config.train_data_path and os.path.exists(config.train_data_path):
        with open(config.train_data_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"[Loaded {len(texts)} texts from {config.train_data_path}]")
    else:
        texts = generate_synthetic_train_data()
        print(f"[Using {len(texts)} synthetic training texts]")

    # Create active trainer (trains attention + persistent + projections)
    trainer = ActiveOuterLoopTrainer(
        config=config,
        encoder=layer.encoder,
        memory=layer.memory,
        attention=layer.attention,
        persistent_vectors=layer.persistent_vectors,
    )

    # Train
    epoch_losses = trainer.train(texts)

    # Print summary
    print(f"\n{'=' * 55}")
    print("Active Training Complete:")
    for i, loss in enumerate(epoch_losses):
        print(f"  Epoch {i+1:3d}:  loss = {loss:.6f}")

    # Save checkpoint
    if config.checkpoint_path:
        layer.save(config.checkpoint_path)
        print(f"\n[Saved active checkpoint to {config.checkpoint_path}]")
    else:
        print("\n[No --checkpoint-path provided, trained weights not saved]")


def run_active(config: TitansConfig):
    """Run active memory inference demo (MLP-driven context, no RAG)."""
    print("=" * 60)
    print("  Titans Active Memory  |  Demo")
    print("=" * 60)
    print(f"  Device:              {config.device}")
    print(f"  d_model:             {config.d_model}")
    print(f"  MLP layers:          {config.memory_num_layers}")
    print(f"  Attention heads:     {config.num_attention_heads}")
    print(f"  Persistent vectors:  {config.num_persistent_vectors}")
    print(f"  Transcript size:     {config.memory_transcript_size}")
    print(f"  Memory LR:           {config.memory_lr}")
    print(f"  Forget gate:         {config.memory_forget_gate}")
    print(f"  Top-K:               {config.top_k}")
    print("=" * 60)

    layer = ActiveMemoryLayer(config)

    # Persistent text context
    if config.persistent_tokens:
        for token in config.persistent_tokens:
            layer.persistent.add(token)
    else:
        layer.persistent.add("You are a helpful technical assistant.")
        layer.persistent.add("Domain: AI, software engineering, computer science.")

    # Load checkpoint if exists
    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        layer.load(config.checkpoint_path)
        print(f"[Loaded active checkpoint from {config.checkpoint_path}]")

    llm_fn = make_demo_llm()

    questions = [
        "What is Python?",
        "Explain the Titans architecture briefly.",
        "How does long-term memory work in computers?",
        "What programming language did we talk about earlier?",
        "Which architecture uses gradient-based memory updates?",
    ]

    for q in questions:
        print(f"\n{'─' * 55}")
        print(f"USER: {q}")
        r = layer.run(q, llm_fn)
        print(f"LLM : {r}")

    print(f"\n{'=' * 55}")
    print("Active Memory Stats:")
    print(json.dumps(layer.stats(), indent=2))

    if config.checkpoint_path:
        layer.save(config.checkpoint_path)
        print(f"\n[Saved active checkpoint to {config.checkpoint_path}]")


# ── Entry point ───────────────────────────────────────────

def main():
    parser = make_parser()
    args = parser.parse_args()
    config = TitansConfig.from_args(args)

    set_seed(config.seed)

    if args.mode == "train":
        run_train(config)
    elif args.mode == "mac":
        run_mac(config)
    elif args.mode == "train-active":
        run_train_active(config)
    elif args.mode == "active":
        run_active(config)
    else:
        run_demo(config)


if __name__ == "__main__":
    main()
