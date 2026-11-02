from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


class PairConditionedMolecularGate(nn.Module):
    """Pair-conditioned reliability fusion for TrimNet and MoLFormer views.

    A single pair-level reliability vector is inferred from a permutation-
    invariant summary of both drugs.  The same modality weights are then
    applied to the two members of that pair, which preserves pair symmetry.

    The pair-conditioned representation is introduced as a learnable residual
    around the global per-drug molecular representation.
    """

    def __init__(
        self,
        dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.10,
        strength_init: float = 0.10,
    ):
        super().__init__()
        self.dim = int(dim)
        self.gate_net = nn.Sequential(
            nn.Linear(self.dim * 5, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2),
        )

        # Start with equal modality reliability.  Pair-specific deviations are
        # learned from data rather than hard-coded.
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

        self.strength_logit = nn.Parameter(
            torch.tensor(_logit(strength_init), dtype=torch.float32)
        )

    def _summary(
        self,
        trim_a: torch.Tensor,
        trim_b: torch.Tensor,
        smiles_a: torch.Tensor,
        smiles_b: torch.Tensor,
    ) -> torch.Tensor:
        trim_mean = 0.5 * (trim_a + trim_b)
        trim_diff = torch.abs(trim_a - trim_b)
        smiles_mean = 0.5 * (smiles_a + smiles_b)
        smiles_diff = torch.abs(smiles_a - smiles_b)
        cross_disagreement = torch.abs(trim_mean - smiles_mean)
        return torch.cat(
            (
                trim_mean,
                trim_diff,
                smiles_mean,
                smiles_diff,
                cross_disagreement,
            ),
            dim=-1,
        )

    def forward(
        self,
        trim_a: torch.Tensor,
        trim_b: torch.Tensor,
        smiles_a: torch.Tensor,
        smiles_b: torch.Tensor,
        base_a: torch.Tensor,
        base_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary = self._summary(trim_a, trim_b, smiles_a, smiles_b)
        reliability = torch.softmax(self.gate_net(summary), dim=-1)

        w_trim = reliability[:, 0:1]
        w_smiles = reliability[:, 1:2]

        pair_a = w_trim * trim_a + w_smiles * smiles_a
        pair_b = w_trim * trim_b + w_smiles * smiles_b

        strength = torch.sigmoid(self.strength_logit)
        fused_a = base_a + strength * (pair_a - base_a)
        fused_b = base_b + strength * (pair_b - base_b)
        return fused_a, fused_b, reliability

    def stats(self) -> Dict[str, float]:
        return {
            'pair_gate_strength': float(
                torch.sigmoid(self.strength_logit).detach().cpu()
            ),
        }


class MonotonicOrdinalHead(nn.Module):
    """CORAL-style monotonic ordinal head for Minor < Moderate < Major.

    It predicts a scalar severity score and two ordered thresholds.  The two
    logits correspond to P(y >= Moderate) and P(y >= Major).  Ordered
    thresholds guarantee the cumulative probabilities are monotonic.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.10,
        min_threshold_gap: float = 0.05,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.threshold_low = nn.Parameter(torch.tensor(-0.5))
        # softplus(0.5) ~= 0.974; thresholds start near -0.5 and +0.52.
        self.threshold_gap_raw = nn.Parameter(torch.tensor(0.5))
        self.min_threshold_gap = float(min_threshold_gap)

    def thresholds(self) -> torch.Tensor:
        gap = F.softplus(self.threshold_gap_raw) + self.min_threshold_gap
        high = self.threshold_low + gap
        return torch.stack((self.threshold_low, high))

    def forward(self, pair_feature: torch.Tensor) -> torch.Tensor:
        score = self.encoder(pair_feature)
        thresholds = self.thresholds().to(
            device=score.device,
            dtype=score.dtype,
        )
        return score - thresholds.view(1, 2)

    @staticmethod
    def targets(labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long().view(-1)
        return torch.stack(
            ((labels >= 1).float(), (labels >= 2).float()),
            dim=1,
        )

    @staticmethod
    def class_probabilities(ordinal_logits: torch.Tensor) -> torch.Tensor:
        q1 = torch.sigmoid(ordinal_logits[:, 0])
        q2 = torch.sigmoid(ordinal_logits[:, 1])
        p0 = 1.0 - q1
        p1 = q1 - q2
        p2 = q2
        probs = torch.stack((p0, p1, p2), dim=1)
        probs = probs.clamp_min(1e-8)
        return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def blend_class_and_ordinal_logits(
    class_logits: torch.Tensor,
    ordinal_logits: torch.Tensor,
    blend: float,
) -> torch.Tensor:
    """Blend ordinary 3-class and monotonic ordinal probabilities."""
    alpha = min(max(float(blend), 0.0), 1.0)
    if alpha <= 0.0:
        return class_logits
    class_probs = torch.softmax(class_logits, dim=-1)
    ordinal_probs = MonotonicOrdinalHead.class_probabilities(ordinal_logits)
    probs = (1.0 - alpha) * class_probs + alpha * ordinal_probs
    return torch.log(probs.clamp_min(1e-8))
