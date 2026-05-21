"""NNUE evaluator (HalfKP features, two-perspective accumulators, small dense head)."""

from .features import (
    FEATURES_PER_PERSPECTIVE,
    NUM_PIECE_KINDS,
    NUM_SQUARES,
    active_features,
    both_perspectives,
)
from .model import DEFAULT_HIDDEN, NNUE, ClippedReLU, pack_batch

__all__ = [
    "FEATURES_PER_PERSPECTIVE",
    "NUM_PIECE_KINDS",
    "NUM_SQUARES",
    "active_features",
    "both_perspectives",
    "DEFAULT_HIDDEN",
    "NNUE",
    "ClippedReLU",
    "pack_batch",
]

# NNUEEvaluator imports torch and the rest of the bot package, kept off the top
# level so `from bot.nnue import active_features` still works in torch-less envs.
def get_evaluator_cls():
    from .evaluator import NNUEEvaluator
    return NNUEEvaluator
