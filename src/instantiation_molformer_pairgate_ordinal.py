import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import random
import numpy as np
import torch
from torch.optim import Adam

from layer_molformer_pairgate_ordinal import MRCGNNPairGateOrdinal


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_random_seed(1, deterministic=True)


def Create_model_pairgate_ordinal(args):
    model = MRCGNNPairGateOrdinal(
        feature=int(args.dimensions),
        sim_feature=int(args.dimensions_sim),
        hidden1=int(args.hidden1),
        hidden2=int(args.hidden2),
        decoder1=int(args.decoder1),
        dropout=float(args.dropout),
        zhongzi=int(args.zhongzi),
        train_data1=args.train_data,
        dataset=str(args.dataset),
        molformer_feature_path=str(args.molformer_feature_path),
        molecular_fusion_hidden_dim=int(args.molecular_fusion_hidden_dim),
        molformer_scale_init=float(args.molformer_scale_init),
        pair_gate_hidden_dim=int(args.pair_gate_hidden_dim),
        pair_gate_dropout=float(args.pair_gate_dropout),
        pair_gate_strength_init=float(args.pair_gate_strength_init),
        ordinal_hidden_dim=int(args.ordinal_hidden_dim),
        ordinal_dropout=float(args.ordinal_dropout),
    )
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    return model, optimizer
