"""Inference-side adapter: turns the NNUE model into a drop-in replacement for evaluate().

This deliberately mirrors the API of bot.evaluation.evaluate so it can be A/B'd:
    evaluate(board, perspective) -> int

Mate/stalemate are still handled by the engine, not the network — the NN can
approximate those, but the search benefits from exact, deterministic terminal
scores. We only delegate non-terminal positions to the network.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..board import Board
from ..enums import Color
from ..evaluation import CHECKMATE_SCORE, STALEMATE_SCORE
from .features import active_features
from .model import NNUE, pack_batch

# Output scaling: the network is trained to predict sigmoid(eval / SCALE).
# At inference we invert that to recover centipawns.
EVAL_SCALE = 400.0


class NNUEEvaluator:
    """Wraps a trained NNUE for use as the engine's evaluation function."""

    def __init__(self, model: NNUE, device: torch.device | str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, hidden: int | None = None) -> "NNUEEvaluator":
        checkpoint = torch.load(path, map_location="cpu")
        h = hidden if hidden is not None else checkpoint.get("hidden", 256)
        model = NNUE(hidden=h)
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model)

    @torch.no_grad()
    def _raw_eval_stm(self, board: Board) -> float:
        """Network output in centipawns, from the side-to-move's perspective."""
        stm = board.state.side_to_move
        other = stm.opposite
        stm_feats = active_features(board, stm)
        other_feats = active_features(board, other)

        stm_idx, stm_off = pack_batch([stm_feats], device=self.device)
        oth_idx, oth_off = pack_batch([other_feats], device=self.device)

        raw = self.model(stm_idx, stm_off, oth_idx, oth_off).item()
        # The model emits a logit; convert sigmoid(logit) back to centipawns.
        # We use the inverse of the training target: target = sigmoid(cp / SCALE),
        # so cp = logit * SCALE keeps the conversion linear in the model's output.
        return raw * EVAL_SCALE

    def evaluate(self, board: Board, perspective: Color) -> int:
        if board.is_checkmate():
            return -CHECKMATE_SCORE if perspective is board.state.side_to_move else CHECKMATE_SCORE
        if board.is_stalemate():
            return STALEMATE_SCORE

        score_stm = self._raw_eval_stm(board)
        score = score_stm if perspective is board.state.side_to_move else -score_stm
        return int(round(score))

    def evaluate_for_side_to_move(self, board: Board) -> int:
        return self.evaluate(board, board.state.side_to_move)
