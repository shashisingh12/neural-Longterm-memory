"""
Titans — Joint BitNet + Neural Memory System (PyTorch)
=======================================================
Based on Behrouz et al. 2501.00663 §3.

Public API:
    from src import MemoryLLM, Phase1Trainer, Phase2Inference, JointTrainingConfig
"""

from .joint_config import JointTrainingConfig, build_joint_parser
from .differentiable_memory import DifferentiableNeuralMemory
from .memory_adapter import MemoryAdapter
from .simple_encoder import TiktokenEncoder
from .memory_llm import MemoryLLM
from .joint_trainer import Phase1Trainer
from .joint_inference import Phase2Inference
from .utils import set_seed, get_device, count_parameters

__all__ = [
    "JointTrainingConfig",
    "build_joint_parser",
    "DifferentiableNeuralMemory",
    "MemoryAdapter",
    "MemoryLLM",
    "Phase1Trainer",
    "Phase2Inference",
    "TiktokenEncoder",
    "set_seed",
    "get_device",
    "count_parameters",
]
