import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import random
import numpy as np
import torch

from parms_setting_molformer_pairgate_ordinal import settings_pairgate_ordinal
from data_preprocess import load_data, generate_data
from instantiation_molformer_pairgate_ordinal import Create_model_pairgate_ordinal
from train_molformer_pairgate_ordinal import train_model_pairgate_ordinal


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    """Parse arguments, build SPO-DDI, and train one predefined fold."""
    set_random_seed(2024, deterministic=False)

    args = settings_pairgate_ordinal()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.experiment_name = 'SPO-DDI'

    (
        data_o,
        data_o_sim,
        data_s,
        data_s_sim,
        data_a,
        data_a_sim,
        train_loader,
        val_loader,
        test_loader,
    ) = load_data(args)

    train_data, vaild_data, test_data = generate_data(args)
    args.test_data = test_data
    args.vaild_data = vaild_data
    args.train_data = train_data

    model, optimizer = Create_model_pairgate_ordinal(args)
    print('[MODEL] MoLFormer enhancement + PairGate + ordinal learning')
    print(model)

    train_model_pairgate_ordinal(
        model,
        optimizer,
        data_o,
        data_o_sim,
        data_s,
        data_s_sim,
        data_a,
        data_a_sim,
        train_loader,
        val_loader,
        test_loader,
        args,
        train_data,
    )


if __name__ == '__main__':
    main()
