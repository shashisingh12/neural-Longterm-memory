"""
Titans — External Neural Memory Layer for LLMs (PyTorch)
==========================================================
Based on Behrouz et al. 2501.00663 §3.

Public API:
    from src import TitansMemoryLayer, TitansConfig, build_parser
"""

from .config import TitansConfig, build_parser
from .memory_layer import TitansMemoryLayer
from .neural_memory import NeuralMemoryMLP
from .text_encoder import TextEncoder
from .text_decoder import TextDecoder
from .persistent_memory import PersistentMemory
from .trainer import OuterLoopTrainer, ActiveOuterLoopTrainer
from .mac_layer import MACMemoryLayer
from .active_layer import ActiveMemoryLayer
from .memory_transcript import MemoryTranscript
from .attention import MemoryAttention, PersistentMemoryVectors
from .parallel_memory import (
    ChunkedLinearMemoryUpdate,
    ChunkedMLPMemoryUpdate,
    parallel_associative_scan,
)

__all__ = [
    "TitansMemoryLayer",
    "MACMemoryLayer",
    "TitansConfig",
    "build_parser",
    "NeuralMemoryMLP",
    "TextEncoder",
    "TextDecoder",
    "PersistentMemory",
    "PersistentMemoryVectors",
    "MemoryAttention",
    "ActiveMemoryLayer",
    "MemoryTranscript",
    "OuterLoopTrainer",
    "ActiveOuterLoopTrainer",
    "ChunkedLinearMemoryUpdate",
    "ChunkedMLPMemoryUpdate",
    "parallel_associative_scan",
]
