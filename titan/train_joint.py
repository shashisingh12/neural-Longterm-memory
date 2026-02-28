"""
Train Joint — Phase 1 CLI Entry Point
========================================
Joint training of M_t + Adapter + LoRA on conversation data.

Usage:
    # Train with default settings:
    python train_joint.py --data conversations.json

    # Custom model + hyperparams:
    python train_joint.py \
        --model-name microsoft/bitnet_b1_58-large \
        --d-mem 64 --lora-rank 8 \
        --n-epochs 5 --lr-lora 2e-5 --lr-memory 1e-4 \
        --checkpoint-path joint_checkpoint.pt

    # Use GPU:
    python train_joint.py --data conversations.json --device cuda

Data format (JSON):
    [
        [
            ["What is your name?", "I'm Alice"],
            ["Where do you live?", "San Francisco"],
            ["What do you know about me?", "You are Alice from San Francisco"]
        ],
        [
            ["My favorite color is blue", "I'll remember that"],
            ["What color do I like?", "Your favorite color is blue"]
        ]
    ]

Each conversation is a list of [question, answer] pairs.
"""

import argparse
import json
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

import yaml

from src.joint_config import JointTrainingConfig, build_joint_parser
from src.memory_llm import MemoryLLM
from src.joint_trainer import Phase1Trainer
from src.utils import set_seed, get_device


def load_conversations(path: str) -> list:
    """Load conversation data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data should be list of conversations
    # each conversation is list of [question, answer] pairs
    conversations = []
    for convo in data:
        pairs = [(turn[0], turn[1]) for turn in convo]
        conversations.append(pairs)

    return conversations


def create_sample_data(path: str):
    """Create sample conversation data for testing."""
    conversations = [
        [
            ["What is your name?", "I'm a helpful AI assistant."],
            ["My name is Alice.", "Nice to meet you, Alice!"],
            ["I live in San Francisco.", "San Francisco is a great city!"],
            ["What is my name?", "Your name is Alice."],
            ["Where do I live?", "You live in San Francisco."],
        ],
        [
            ["I have a dog named Max.", "That's a lovely name for a dog!"],
            ["Max is a golden retriever.", "Golden retrievers are wonderful dogs."],
            ["What kind of dog do I have?", "You have a golden retriever named Max."],
        ],
        [
            ["My favorite programming language is Python.",
             "Python is a versatile and popular language."],
            ["I work as a data scientist.",
             "Data science is a fascinating field."],
            ["What do I do for work?",
             "You work as a data scientist."],
            ["What language do I prefer?",
             "Your favorite programming language is Python."],
        ],
        [
            ["I was born in 1990.", "So you're in your thirties."],
            ["I have diabetes.", "That's important health information to remember."],
            ["My doctor is Dr. Smith.", "I'll remember Dr. Smith is your doctor."],
            ["What health conditions do I have?",
             "You have diabetes, and your doctor is Dr. Smith."],
        ],
        [
            ["The weather is nice today.", "Glad to hear it's nice out!"],
            ["What is 2 plus 2?", "2 plus 2 equals 4."],
            ["I like reading science fiction.", "Science fiction is a great genre!"],
            ["What do I enjoy reading?", "You enjoy reading science fiction."],
        ],
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2)

    print(f"  Created sample data: {path}")
    print(f"  {len(conversations)} conversations, "
          f"{sum(len(c) for c in conversations)} total turns")
    return conversations


def main():
    parser = build_joint_parser()
    parser.add_argument(
        "--data", type=str, default="conversations.json",
        help="Path to conversation data JSON (default: conversations.json)"
    )
    parser.add_argument(
        "--create-sample-data", action="store_true",
        help="Create sample conversation data file and train on it"
    )
    args = parser.parse_args()

    # build config: YAML (if given) with CLI overrides, else pure CLI
    if args.config:
        print(f"  Loading config from: {args.config}")
        config = JointTrainingConfig.from_yaml_with_overrides(
            args.config, args, parser
        )
        # also read data/create_sample_data from YAML if not overridden on CLI
        with open(args.config, "r", encoding="utf-8") as _f:
            _yaml_cfg = yaml.safe_load(_f) or {}
        if args.data == "conversations.json" and "data" in _yaml_cfg:
            args.data = _yaml_cfg["data"]
        if not args.create_sample_data and _yaml_cfg.get("create_sample_data"):
            args.create_sample_data = True
    else:
        config = JointTrainingConfig.from_args(args)

    set_seed(config.seed)

    # load or create data
    data_path = args.data
    if args.create_sample_data or not os.path.exists(data_path):
        if not os.path.exists(data_path):
            print(f"  Data file not found: {data_path}")
            print(f"  Creating sample data...")
        conversations = create_sample_data(data_path)
    else:
        conversations = load_conversations(data_path)

    # Resolve device early so the banner shows the actual device
    resolved_device = get_device(config.device)
    config.device = str(resolved_device)

    print("=" * 65)
    print("  PHASE 1 — JOINT TRAINING: M_t + Adapter + LoRA (peft)")
    print("=" * 65)
    print(f"  Model:          {config.model_name}")
    print(f"  d_mem:          {config.d_mem}")
    print(f"  LoRA rank:      {config.lora_rank}")
    print(f"  LoRA targets:   {config.lora_targets}")
    print(f"  LoRA layers:    {config.lora_layers}")
    print(f"  Epochs:         {config.n_epochs}")
    print(f"  LR (LoRA):      {config.lr_lora}")
    print(f"  LR (Memory):    {config.lr_memory}")
    print(f"  LR (Adapter):   {config.lr_adapter}")
    print(f"  Device:         {config.device}")
    print(f"  Conversations:  {len(conversations)}")
    print(f"  Total turns:    {sum(len(c) for c in conversations)}")
    print("=" * 65)

    # build model
    print("\n  Loading model...")
    t0 = time.time()
    model = MemoryLLM(config)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # build trainer
    trainer = Phase1Trainer(model, config)

    # train
    t0 = time.time()
    epoch_metrics = trainer.train(conversations)
    elapsed = time.time() - t0

    # summary
    print(f"\n{'=' * 65}")
    print("  Training Summary")
    print(f"{'=' * 65}")
    for i, m in enumerate(epoch_metrics):
        print(f"  Epoch {i+1}: "
              f"task={m['loss_task']:.4f}  "
              f"inner={m['loss_inner']:.4f}  "
              f"alpha_mean={m['alpha_mean']:.4f}")
    print(f"{'─' * 65}")
    print(f"  Total time: {elapsed:.1f}s")

    if epoch_metrics:
        improvement = (
            (epoch_metrics[0]["loss_task"] - epoch_metrics[-1]["loss_task"])
            / max(epoch_metrics[0]["loss_task"], 1e-12)
            * 100
        )
        print(f"  Task loss improvement: {improvement:.1f}%")

    # save checkpoint
    checkpoint_path = (
        config.checkpoint_path or "joint_checkpoint.pt"
    )
    model.save_checkpoint(checkpoint_path)

    # freeze for inference
    model.freeze_for_inference()

    print(f"\n  Done. Checkpoint: {checkpoint_path}")
    print("  To run Phase 2 inference:")
    print(f"    python chat_joint.py --checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
