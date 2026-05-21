"""NNUE model definition (HalfKP, two-perspective, small dense head).

Architecture:
    For each perspective p in {side_to_move, other}:
        sparse 40960-d feature vector  -->  Linear(40960 -> H)
                                            (this is the "accumulator")
    concat(acc_stm, acc_other)         -->  2H
        ClippedReLU                    -->  2H
        Linear(2H -> 32) + ClippedReLU
        Linear(32 -> 32) + ClippedReLU
        Linear(32 -> 1)                -->  scalar eval (centipawn-ish)

The first layer is implemented via nn.EmbeddingBag(mode='sum'): for a sparse
input we sum the columns of the weight matrix corresponding to the active
feature indices. This is the same math as a sparse Linear, but is the form
that supports the incremental-update trick at inference time (you sum/subtract
columns as features toggle on/off after each move).

Output convention: the network always predicts eval FROM THE SIDE TO MOVE'S
PERSPECTIVE. The adapter in evaluator.py converts that to whichever
perspective the caller asked for.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import FEATURES_PER_PERSPECTIVE

DEFAULT_HIDDEN = 256


class ClippedReLU(nn.Module):
    """ReLU clamped to [0, 1]. Standard NNUE activation; plays well with int8 quantization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0)


class NNUE(nn.Module):
    def __init__(self, hidden: int = DEFAULT_HIDDEN) -> None:
        super().__init__()
        self.hidden = hidden
        # Shared feature transformer: one EmbeddingBag, used for both perspectives.
        # Shared weights matter — without sharing, white/black-perspective accumulators
        # would learn different feature embeddings, doubling parameters and hurting generalization.
        self.feature_transformer = nn.EmbeddingBag(
            num_embeddings=FEATURES_PER_PERSPECTIVE,
            embedding_dim=hidden,
            mode="sum",
        )
        self.feature_bias = nn.Parameter(torch.zeros(hidden))

        self.act = ClippedReLU()
        self.fc1 = nn.Linear(2 * hidden, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc_out = nn.Linear(32, 1)

    def accumulator(self, feature_indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Run the feature transformer for one perspective.

        feature_indices: 1-D LongTensor, the concatenated active features across a batch
        offsets: 1-D LongTensor, length = batch_size, where each perspective's features start
        """
        return self.feature_transformer(feature_indices, offsets) + self.feature_bias

    def forward(
        self,
        stm_features: torch.Tensor,
        stm_offsets: torch.Tensor,
        other_features: torch.Tensor,
        other_offsets: torch.Tensor,
    ) -> torch.Tensor:
        acc_stm = self.accumulator(stm_features, stm_offsets)
        acc_other = self.accumulator(other_features, other_offsets)
        x = torch.cat([acc_stm, acc_other], dim=1)
        x = self.act(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc_out(x).squeeze(-1)


def pack_batch(
    feature_lists: list[list[int]],
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a batch of variable-length feature lists into (indices, offsets) for EmbeddingBag."""
    offsets = [0]
    flat: list[int] = []
    for feats in feature_lists:
        flat.extend(feats)
        offsets.append(len(flat))
    indices = torch.tensor(flat, dtype=torch.long, device=device)
    offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
    return indices, offsets_tensor
