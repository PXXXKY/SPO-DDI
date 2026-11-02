"""Dependency-light smoke tests for SPO-DDI's new components."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molecular_fusion import MolecularFeatureFusion  # noqa: E402
from pair_conditioned_gate_ordinal import (  # noqa: E402
    MonotonicOrdinalHead,
    PairConditionedMolecularGate,
    blend_class_and_ordinal_logits,
)


def test_pair_gate() -> None:
    gate = PairConditionedMolecularGate(
        dim=128,
        hidden_dim=64,
        dropout=0.0,
        strength_init=0.1,
    )
    trim = torch.randn(8, 128)
    smiles = torch.randn(8, 128)
    fused_a, fused_b, reliability = gate(
        trim[:4], trim[4:], smiles[:4], smiles[4:], trim[:4], trim[4:]
    )
    assert fused_a.shape == (4, 128)
    assert fused_b.shape == (4, 128)
    assert reliability.shape == (4, 2)
    assert torch.allclose(
        reliability.sum(dim=-1),
        torch.ones(4),
        atol=1e-6,
    )


def test_ordinal_head_and_blend() -> None:
    head = MonotonicOrdinalHead(input_dim=512, hidden_dim=64, dropout=0.0)
    features = torch.randn(6, 512)
    ordinal_logits = head(features)
    assert ordinal_logits.shape == (6, 2)

    class_logits = torch.randn(6, 3)
    blended_logits = blend_class_and_ordinal_logits(
        class_logits,
        ordinal_logits,
        blend=0.15,
    )
    probabilities = blended_logits.exp()
    assert probabilities.shape == (6, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(6), atol=1e-6)


def test_molecular_fusion() -> None:
    module = MolecularFeatureFusion(
        molformer_dim=64,
        trimnet_dim=128,
        common_dim=128,
        hidden_dim=64,
    )
    trim = torch.randn(10, 128)
    mol = torch.randn(10, 64)
    fused, trim_view, smiles_view = module(trim, mol)
    assert fused.shape == (10, 128)
    assert trim_view.shape == (10, 128)
    assert smiles_view.shape == (10, 128)


if __name__ == "__main__":
    test_pair_gate()
    test_ordinal_head_and_blend()
    test_molecular_fusion()
    print("SPO-DDI core component tests passed.")

