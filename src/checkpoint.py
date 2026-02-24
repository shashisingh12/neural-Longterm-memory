"""
Titans Memory — Checkpoint Save / Load
========================================
Uses torch.save / torch.load for native tensor serialization.
Stores full system state: config, memory weights, momentum buffers,
encoder projection, decoder store, persistent tokens, turn counter.
"""

import torch
from typing import Tuple

from .config import TitansConfig


def save_checkpoint(path: str, config: TitansConfig, memory_layer) -> None:
    """Save complete memory system state to a .pt file.

    Args:
        path:         File path to save (e.g. "checkpoint.pt").
        config:       The TitansConfig used to create the system.
        memory_layer: A TitansMemoryLayer instance.
    """
    state = {
        "config": config.to_dict(),
        "memory_mlp": memory_layer.memory.state_dict(),
        "encoder_projection": memory_layer.encoder.projection.state_dict(),
        "decoder_store": memory_layer.decoder.state_dict(),
        "persistent": memory_layer.persistent.state_dict(),
        "turn": memory_layer._turn,
    }
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Tuple[TitansConfig, dict]:
    """Load a checkpoint and return (config, raw_state).

    The caller is responsible for reconstructing the TitansMemoryLayer
    from the config and then calling restore_from_state().

    Args:
        path:         File path to load.
        map_location: Device to map tensors to (default: "cpu").

    Returns:
        (config, state_dict) tuple.
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    config = TitansConfig.from_dict(state["config"])
    return config, state


def restore_from_state(memory_layer, state: dict) -> None:
    """Restore a TitansMemoryLayer's internal state from a loaded checkpoint.

    Args:
        memory_layer: A freshly constructed TitansMemoryLayer.
        state:        The raw state dict from load_checkpoint().
    """
    memory_layer.memory.load_state_dict(state["memory_mlp"])
    memory_layer.encoder.projection.load_state_dict(state["encoder_projection"])
    memory_layer.decoder.load_state_dict(state["decoder_store"])
    memory_layer.persistent.load_state_dict(state["persistent"])
    memory_layer._turn = state.get("turn", 0)
