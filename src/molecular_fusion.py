"""Trainable fusion of frozen TrimNet and MoLFormer drug representations."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class ResidualTrimNetAdapter(nn.Module):
    """Near-identity adapter for the 128-dimensional TrimNet view."""

    def __init__(self, dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class MoLFormerAdapter(nn.Module):
    """Project frozen MoLFormer vectors into the shared molecular space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class MolecularFeatureFusion(nn.Module):
    """Fuse structural and SMILES-derived molecular representations.

    The structural view is adapted with a residual near-identity mapping. The
    MoLFormer view is projected to the same dimension and introduced through a
    trainable feature-wise gate and a bounded residual scale.
    """

    def __init__(
        self,
        molformer_dim: int,
        trimnet_dim: int = 128,
        common_dim: int = 128,
        hidden_dim: int = 256,
        molformer_scale_init: float = 0.10,
    ):
        super().__init__()
        if int(trimnet_dim) != int(common_dim):
            raise ValueError("trimnet_dim must equal common_dim")

        self.trimnet_adapter = ResidualTrimNetAdapter(
            dim=int(trimnet_dim),
            hidden_dim=int(hidden_dim),
        )
        self.molformer_adapter = MoLFormerAdapter(
            input_dim=int(molformer_dim),
            output_dim=int(common_dim),
            hidden_dim=int(hidden_dim),
        )
        self.fusion_gate = nn.Linear(int(common_dim) * 2, int(common_dim))
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -2.0)

        initial_scale = min(max(float(molformer_scale_init), 1e-4), 1.0 - 1e-4)
        self.molformer_scale_logit = nn.Parameter(
            torch.tensor(math.log(initial_scale / (1.0 - initial_scale)))
        )

    def forward(
        self,
        raw_trimnet: torch.Tensor,
        raw_molformer: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trimnet_view = self.trimnet_adapter(raw_trimnet)
        molformer_view = self.molformer_adapter(raw_molformer)
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((trimnet_view, molformer_view), dim=-1))
        )
        scale = torch.sigmoid(self.molformer_scale_logit)
        fused = trimnet_view + scale * gate * molformer_view
        return fused, trimnet_view, molformer_view

    def stats(self) -> dict[str, float]:
        return {
            "molformer_residual_scale": float(
                torch.sigmoid(self.molformer_scale_logit).detach().cpu()
            )
        }
