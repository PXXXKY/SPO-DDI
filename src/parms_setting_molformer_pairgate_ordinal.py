import argparse
import os
import random

import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_random_seed(2024, deterministic=True)


def settings_pairgate_ordinal():
    parser = argparse.ArgumentParser(
        description=(
            "SPO-DDI: MoLFormer enhancement + PairGate + ordinal learning"
        )
    )

    # Core MSFCL training arguments.
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--fastmode', action='store_true', default=False)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--out_file', default='spo_ddi_result.txt')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--network_ratio', type=float, default=1)
    parser.add_argument('--loss_ratio1', type=float, default=1)
    parser.add_argument('--loss_ratio3', type=float, default=0.5)
    parser.add_argument('--dimensions', type=int, default=128)
    parser.add_argument('--dimensions_sim', type=int, default=0)
    parser.add_argument('--hidden1', type=int, default=64)
    parser.add_argument('--hidden2', type=int, default=32)
    parser.add_argument('--decoder1', type=int, default=512)
    parser.add_argument('--zhongzi', type=int, default=0)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='ddinter')

    # New independent SMILES-language branch.
    parser.add_argument(
        '--molformer_feature_path',
        type=str,
        default=(
            'trimnet/data/ddinter/'
            'molformer_ibm_molformer_xl_both_10pct.npz'
        ),
        help=(
            'Precomputed frozen IBM MoLFormer drug embeddings. '
            'The npz must contain drug_ids and embeddings.'
        ),
    )

    parser.add_argument('--molecular_fusion_hidden_dim', type=int, default=256)
    parser.add_argument(
        '--molformer_scale_init',
        type=float,
        default=0.10,
        help=(
            'Initial residual contribution of the MoLFormer view.'
        ),
    )

    # Pair-conditioned molecular reliability gate.
    parser.add_argument('--pair_gate_hidden_dim', type=int, default=256)
    parser.add_argument('--pair_gate_dropout', type=float, default=0.10)
    parser.add_argument(
        '--pair_gate_strength_init',
        type=float,
        default=0.10,
        help=(
            'Initial residual strength of pair-conditioned fusion.'
        ),
    )

    # Severity-aware monotonic ordinal head: Minor < Moderate < Major.
    parser.add_argument('--ordinal_hidden_dim', type=int, default=128)
    parser.add_argument('--ordinal_dropout', type=float, default=0.10)
    parser.add_argument(
        '--ordinal_weight',
        type=float,
        default=0.20,
        help='Weight of the auxiliary monotonic ordinal BCE loss.',
    )
    parser.add_argument(
        '--ordinal_infer_blend',
        type=float,
        default=0.15,
        help=(
            'Validation/test probability blend between the 3-class head and '
            'ordinal head. 0 disables ordinal inference blending.'
        ),
    )
    parser.add_argument(
        '--ordinal_balance',
        type=str,
        choices=['none', 'global'],
        default='global',
        help=(
            'global balances the two cumulative ordinal thresholds using '
            'training-set positive/negative counts only.'
        ),
    )

    # Clean experiment isolation / recovery.
    parser.add_argument(
        '--checkpoint_root',
        type=str,
        default='checkpoints_pairgate_ordinal',
    )
    parser.add_argument(
        '--result_root',
        type=str,
        default='results_pairgate_ordinal',
    )
    parser.add_argument(
        '--auto_resume',
        action='store_true',
        default=False,
        help='Resume from the latest completed epoch if last_epoch.pt exists.',
    )

    return parser.parse_args()
