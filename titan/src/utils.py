"""
Titans Memory — Utilities
==========================
Shared helpers: seeding, device resolution, parameter counting.
"""

import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across torch and cuda."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    """Validate and return a torch.device for the requested string."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("[Warning] MPS requested but not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def count_parameters(module: nn.Module) -> dict:
    """Count total and trainable parameters in a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
