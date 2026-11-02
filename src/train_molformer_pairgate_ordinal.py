from __future__ import annotations

import copy
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pair_conditioned_gate_ordinal import MonotonicOrdinalHead, blend_class_and_ordinal_logits


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_random_seed(2024, deterministic=True)


def gen_ran_output(data, model, vice_gnn, args, output, label, device):
    """Build the MSFCL adversarial target with one persistent auxiliary GNN.

    A CUDA GNN copy per mini-batch can accumulate allocator/device-guard state
    in a long run. ``vice_gnn`` is allocated once before training; its state is
    refreshed in place here, which preserves the original perturbation rule.
    """
    model.zero_grad(set_to_none=True)
    criterion = torch.nn.CrossEntropyLoss()
    temp_loss = criterion(output, label.long())
    temp_loss.backward(retain_graph=True)

    with torch.no_grad():
        original_gnn_params = dict(model.gnn.named_parameters())
        for name, vice_param in vice_gnn.named_parameters():
            param = original_gnn_params[name]
            if param.numel() <= 1:
                vice_param.copy_(param.detach())
                continue

            grad_direction = (
                param.grad.sign()
                if param.grad is not None
                else torch.zeros_like(param)
            )
            std = param.detach().std()
            if not torch.isfinite(std):
                std = torch.zeros((), device=param.device, dtype=param.dtype)
            random_noise = torch.normal(
                mean=0.0,
                std=torch.ones_like(param) * std,
            )
            loss_value = float(temp_loss.detach().cpu())
            loss_weight = loss_value / (loss_value + 1e-6)
            perturbed = param.detach() + loss_weight * args.eta * (
                args.alpha * grad_direction
                + (1.0 - args.alpha) * random_noise
            )
            vice_param.copy_(perturbed)

        # A per-batch deepcopy also copied GNN buffers (e.g. BatchNorm
        # statistics); refresh them explicitly for the persistent auxiliary.
        original_gnn_buffers = dict(model.gnn.named_buffers())
        for name, vice_buffer in vice_gnn.named_buffers():
            vice_buffer.copy_(original_gnn_buffers[name].detach())

        chemical_features, _, _ = model.molecular_node_features()
        vice_gnn.train(model.gnn.training)
        _, _, _, z2_o, _, z2_sim = vice_gnn(
            data,
            x_override=chemical_features.detach(),
        )

    model.zero_grad(set_to_none=True)
    return z2_o.detach(), z2_sim.detach()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class NormalizedCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits):
        logits = logits / self.temperature
        labels = torch.zeros(
            logits.size(0),
            dtype=torch.long,
            device=logits.device,
        )
        return self.cross_entropy(logits, labels)


def _labels_from_batch(inp, device):
    label = torch.as_tensor(
        np.asarray(inp[2], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    return label


def _warmup_factor(epoch_index, warmup_epochs):
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch_index + 1) / float(warmup_epochs))


def _rng_state():
    state = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'python' in state:
        random.setstate(state['python'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _checkpoint_paths(args):
    fold_dir = Path(args.checkpoint_root) / f'fold_{int(args.zhongzi)}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    return (
        fold_dir,
        fold_dir / 'best_val_macro_f1.pt',
        fold_dir / 'last_epoch.pt',
    )


def _save_last_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_f1,
    best_epoch,
):
    torch.save(
        {
            'epoch': int(epoch),
            'best_val_macro_f1': float(best_val_f1),
            'best_epoch': int(best_epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'rng_state': _rng_state(),
        },
        path,
    )


def _resume_if_requested(
    model,
    optimizer,
    scheduler,
    args,
    last_path,
):
    if not getattr(args, 'auto_resume', False):
        return 0, -1.0, 0
    if not last_path.exists():
        print('[RESUME] No last_epoch.pt found; starting from epoch 1.')
        return 0, -1.0, 0

    ckpt = torch.load(last_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    _restore_rng_state(ckpt.get('rng_state'))

    start_epoch = int(ckpt['epoch'])
    best_val_f1 = float(ckpt.get('best_val_macro_f1', -1.0))
    best_epoch = int(ckpt.get('best_epoch', 0))

    print(
        f'[RESUME] fold={int(args.zhongzi)} '
        f'next_epoch={start_epoch + 1} '
        f'best_epoch={best_epoch} '
        f'best_val_macro_f1={best_val_f1:.6f}'
    )
    return start_epoch, best_val_f1, best_epoch


def _classification_metrics(labels, logits):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32).reshape((-1, 3))
    preds = logits.argmax(axis=1)

    per_class_precision = precision_score(
        labels,
        preds,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    per_class_recall = recall_score(
        labels,
        preds,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    per_class_f1 = f1_score(
        labels,
        preds,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    ordinal_abs_error = np.abs(labels - preds)
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')

    return {
        'accuracy': accuracy_score(labels, preds),
        'balanced_accuracy': balanced_accuracy_score(labels, preds),
        'macro_f1': f1_score(
            labels, preds, average='macro', zero_division=0
        ),
        'macro_recall': recall_score(
            labels, preds, average='macro', zero_division=0
        ),
        'macro_precision': precision_score(
            labels, preds, average='macro', zero_division=0
        ),
        'mcc': matthews_corrcoef(labels, preds),
        'qwk': float(qwk),
        'ordinal_mae': float(ordinal_abs_error.mean()),
        'extreme_error_rate': float((ordinal_abs_error == 2).mean()),
        'minor_precision': float(per_class_precision[0]),
        'minor_recall': float(per_class_recall[0]),
        'minor_f1': float(per_class_f1[0]),
        'moderate_precision': float(per_class_precision[1]),
        'moderate_recall': float(per_class_recall[1]),
        'moderate_f1': float(per_class_f1[1]),
        'major_precision': float(per_class_precision[2]),
        'major_recall': float(per_class_recall[2]),
        'major_f1': float(per_class_f1[2]),
        'confusion_matrix': confusion_matrix(
            labels, preds, labels=[0, 1, 2]
        ).tolist(),
        'preds': preds,
        'logits': logits,
    }


def evaluate(
    model,
    loader,
    data_o,
    data_o_sim,
    data_s,
    data_s_sim,
    data_a,
    data_a_sim,
    args,
):
    model.eval()
    device = next(model.parameters()).device
    graph_data = args.train_data

    all_labels = []
    all_class_labels = []
    all_logits = []
    losses = []
    ordinal_losses = []
    gate_weights_all = []

    ce = nn.CrossEntropyLoss()
    pos_weight = None
    if getattr(args, 'ordinal_balance', 'global') == 'global':
        pos_weight = model.ordinal_pos_weight().to(device)
    ordinal_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        # Graph encodings are invariant across mini-batches during evaluation.
        x, x_sim, x1_o, x2_o, x1_sim, x2_sim = model.forward_cl(graph_data)

        for inp in loader:
            label = _labels_from_batch(inp, device)
            class_label = torch.as_tensor(
                np.asarray(inp[3], dtype=np.int64),
                dtype=torch.long,
                device=device,
            )

            class_logits, ordinal_logits, gate_weights = model.forward_pair_heads(
                x1_o,
                x2_o,
                x1_sim,
                x2_sim,
                inp,
            )
            class_logits = torch.squeeze(class_logits)
            ordinal_logits = torch.squeeze(ordinal_logits)
            if class_logits.ndim == 1:
                class_logits = class_logits.unsqueeze(0)
            if ordinal_logits.ndim == 1:
                ordinal_logits = ordinal_logits.unsqueeze(0)

            final_logits = blend_class_and_ordinal_logits(
                class_logits,
                ordinal_logits,
                getattr(args, 'ordinal_infer_blend', 0.0),
            )

            cls_loss = args.loss_ratio1 * ce(class_logits, label.long())
            ord_targets = MonotonicOrdinalHead.targets(label)
            ord_loss = ordinal_criterion(ordinal_logits, ord_targets)
            losses.append(float(cls_loss.detach().cpu()))
            ordinal_losses.append(float(ord_loss.detach().cpu()))

            all_labels.extend(label.detach().cpu().tolist())
            all_class_labels.extend(class_label.detach().cpu().tolist())
            all_logits.append(final_logits.detach().cpu().numpy())
            gate_weights_all.append(gate_weights.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    metrics = _classification_metrics(all_labels, all_logits)
    metrics['loss'] = float(np.mean(losses)) if losses else float('nan')
    metrics['ordinal_loss'] = (
        float(np.mean(ordinal_losses)) if ordinal_losses else float('nan')
    )
    metrics['class_labels'] = np.asarray(all_class_labels, dtype=np.int64)

    if gate_weights_all:
        gate_arr = np.concatenate(gate_weights_all, axis=0)
        metrics['gate_trim_mean'] = float(gate_arr[:, 0].mean())
        metrics['gate_smiles_mean'] = float(gate_arr[:, 1].mean())
        metrics['gate_smiles_std'] = float(gate_arr[:, 1].std())
    else:
        metrics['gate_trim_mean'] = float('nan')
        metrics['gate_smiles_mean'] = float('nan')
        metrics['gate_smiles_std'] = float('nan')

    # Baseline-compatible AUC/AUPR calculations on final inference logits.
    labels_arr = np.asarray(all_labels, dtype=np.int64)
    onehot = np.zeros((labels_arr.shape[0], 3), dtype=np.float32)
    onehot[np.arange(labels_arr.shape[0]), labels_arr] = 1.0

    auc_macro_sum = 0.0
    aupr_macro_sum = 0.0
    valid_classes = 0
    for cls in range(3):
        if onehot[:, cls].sum() < 1:
            continue
        auc_macro_sum += roc_auc_score(onehot[:, cls], all_logits[:, cls])
        precision, recall, _ = precision_recall_curve(
            onehot[:, cls], all_logits[:, cls]
        )
        aupr_macro_sum += auc(recall, precision)
        valid_classes += 1

    metrics['auc_macro'] = (
        auc_macro_sum / valid_classes if valid_classes else float('nan')
    )
    metrics['aupr_macro'] = (
        aupr_macro_sum / valid_classes if valid_classes else float('nan')
    )
    metrics['auc_micro'] = roc_auc_score(
        onehot.reshape(-1),
        all_logits.reshape(-1),
        average='micro',
    )
    precision, recall, _ = precision_recall_curve(
        onehot.reshape(-1), all_logits.reshape(-1)
    )
    metrics['aupr_micro'] = auc(recall, precision)
    return metrics

def train_model_pairgate_ordinal(
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
):
    device = torch.device(
        'cuda'
        if args.cuda
        else 'cpu'
    )

    criterion = LabelSmoothingLoss(classes=3, smoothing=0.1)
    nce_criterion = NormalizedCrossEntropyLoss(
        temperature=args.tau
    )

    ordinal_pos_weight = None
    if getattr(args, 'ordinal_balance', 'global') == 'global':
        ordinal_pos_weight = model.ordinal_pos_weight().to(device)
    ordinal_criterion = nn.BCEWithLogitsLoss(
        pos_weight=ordinal_pos_weight
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6,
    )
    # Allocate the auxiliary on CPU, then transfer it once. It is not optimized or checkpointed.
    vice_gnn = copy.deepcopy(model.gnn)

    if args.cuda:
        model.to(device)
        vice_gnn.to(device)
        data_o.to(device)
        data_s.to(device)
        data_a.to(device)
        data_o_sim.to(device)
        data_s_sim.to(device)
        data_a_sim.to(device)
        train_data.to(device)
        args.vaild_data.to(device)

    fold_dir, best_path, last_path = _checkpoint_paths(args)
    Path(args.result_root).mkdir(parents=True, exist_ok=True)

    start_epoch, best_val_f1, best_epoch = _resume_if_requested(
        model,
        optimizer,
        scheduler,
        args,
        last_path,
    )

    print(
        'Start '
        f'{getattr(args, "experiment_name", "SPO-DDI")} '
        'Training...'
    )
    print(
        '[SPO-DDI CONFIG] '
        f'molformer_scale_init={args.molformer_scale_init}, '
        f'pair_gate_strength_init={args.pair_gate_strength_init}, '
        f'ordinal_weight={args.ordinal_weight}, '
        f'ordinal_infer_blend={args.ordinal_infer_blend}, '
        f'ordinal_balance={args.ordinal_balance}'
    )

    t_total = time.time()
    epoch_rows = []

    for epoch in range(start_epoch, args.epochs):
        t = time.time()
        print(f'-------- Epoch {epoch + 1} --------')
        train_labels = []
        train_logits = []
        batch_total_losses = []
        batch_base_losses = []
        batch_ordinal_losses = []
        batch_gate_weights = []

        for i, inp in enumerate(train_loader):
            label = _labels_from_batch(inp, device)

            model.train()
            optimizer.zero_grad()

            x, x_sim, x1_o, x2_o, x1_sim, x2_sim = model.forward_cl(
                train_data
            )
            output, ordinal_logits, gate_weights = model.forward_pair_heads(
                x1_o,
                x2_o,
                x1_sim,
                x2_sim,
                inp,
            )
            output = torch.squeeze(output)
            ordinal_logits = torch.squeeze(ordinal_logits)
            if output.ndim == 1:
                output = output.unsqueeze(0)
            if ordinal_logits.ndim == 1:
                ordinal_logits = ordinal_logits.unsqueeze(0)

            z2_o, z2_sim = gen_ran_output(
                train_data,
                model,
                vice_gnn,
                args,
                output,
                label,
                device=device,
            )
            z2_o = Variable(
                z2_o.detach().data,
                requires_grad=False,
            )
            z2_sim = Variable(
                z2_sim.detach().data,
                requires_grad=False,
            )

            # Original DDInter MSFCL losses.
            loss1 = criterion(output, label)
            ordinal_targets = MonotonicOrdinalHead.targets(label)
            loss_ordinal = ordinal_criterion(
                ordinal_logits,
                ordinal_targets,
            )

            x1_x2_org = model.loss_cl(x, x2_o, z2_o)
            loss2_org = nce_criterion(x1_x2_org)
            x1_x2_sim = model.loss_cl(x_sim, x2_sim, z2_sim)
            loss2_sim = nce_criterion(x1_x2_sim)
            loss2 = loss2_org + loss2_sim

            loss_base = (
                args.loss_ratio1 * loss1
                + args.loss_ratio3 * loss2
            )

            loss_train = (
                loss_base
                + float(args.ordinal_weight) * loss_ordinal
            )

            loss_train.backward()
            optimizer.step()

            train_labels.extend(label.detach().cpu().tolist())
            train_final_logits = blend_class_and_ordinal_logits(
                output,
                ordinal_logits,
                getattr(args, 'ordinal_infer_blend', 0.0),
            )
            train_logits.append(train_final_logits.detach().cpu().numpy())
            batch_total_losses.append(float(loss_train.detach().cpu()))
            batch_base_losses.append(float(loss_base.detach().cpu()))
            batch_ordinal_losses.append(float(loss_ordinal.detach().cpu()))
            batch_gate_weights.append(gate_weights.detach().cpu().numpy())

            if i % 100 == 0:
                print(
                    f'epoch: {epoch + 1}/ '
                    f'iteration: {i + 1}/ '
                    f'loss_total: {loss_train.detach().cpu().item():.6f}/ '
                    f'loss_base: {loss_base.detach().cpu().item():.6f}/ '
                    f'ordinal: {loss_ordinal.detach().cpu().item():.6f}'
                )

        scheduler.step()

        train_logits_np = np.concatenate(train_logits, axis=0)
        train_metrics = _classification_metrics(
            train_labels,
            train_logits_np,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            data_o,
            data_o_sim,
            data_s,
            data_s_sim,
            data_a,
            data_a_sim,
            args,
        )

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = float(val_metrics['macro_f1'])
            best_epoch = int(epoch + 1)
            torch.save(
                {
                    'epoch': best_epoch,
                    'val_macro_f1': best_val_f1,
                    'val_accuracy': float(val_metrics['accuracy']),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                best_path,
            )
            print(
                f'[BEST SPO-DDI] epoch={best_epoch}, '
                f'val_macro_f1={best_val_f1:.6f}, '
                f'val_accuracy={val_metrics["accuracy"]:.6f}'
            )

        fusion_stats = model.molecular_fusion_stats()
        pair_stats = model.pairgate_ordinal_stats()
        if batch_gate_weights:
            gate_arr = np.concatenate(batch_gate_weights, axis=0)
            train_gate_trim_mean = float(gate_arr[:, 0].mean())
            train_gate_smiles_mean = float(gate_arr[:, 1].mean())
            train_gate_smiles_std = float(gate_arr[:, 1].std())
        else:
            train_gate_trim_mean = float('nan')
            train_gate_smiles_mean = float('nan')
            train_gate_smiles_std = float('nan')
        elapsed = time.time() - t

        epoch_row = {
            'epoch': int(epoch + 1),
            'loss_total': float(np.mean(batch_total_losses)),
            'loss_base': float(np.mean(batch_base_losses)),
            'ordinal_loss': float(np.mean(batch_ordinal_losses)),
            'train_accuracy': float(train_metrics['accuracy']),
            'train_macro_f1': float(train_metrics['macro_f1']),
            'val_loss': float(val_metrics['loss']),
            'val_accuracy': float(val_metrics['accuracy']),
            'val_macro_f1': float(val_metrics['macro_f1']),
            'val_minor_f1': float(val_metrics['minor_f1']),
            'val_moderate_f1': float(val_metrics['moderate_f1']),
            'val_major_f1': float(val_metrics['major_f1']),
            'molformer_residual_scale': float(
                fusion_stats['molformer_residual_scale']
            ),
            'pair_gate_strength': float(pair_stats['pair_gate_strength']),
            'train_gate_trim_mean': train_gate_trim_mean,
            'train_gate_smiles_mean': train_gate_smiles_mean,
            'train_gate_smiles_std': train_gate_smiles_std,
            'val_gate_trim_mean': float(val_metrics['gate_trim_mean']),
            'val_gate_smiles_mean': float(val_metrics['gate_smiles_mean']),
            'val_gate_smiles_std': float(val_metrics['gate_smiles_std']),
            'ordinal_threshold_1': float(
                pair_stats['ordinal_threshold_minor_moderate']
            ),
            'ordinal_threshold_2': float(
                pair_stats['ordinal_threshold_moderate_major']
            ),
            'time_sec': float(elapsed),
        }
        epoch_rows.append(epoch_row)

        print(
            f'epoch: {epoch + 1:04d} '
            f'loss_total: {epoch_row["loss_total"]:.4f} '
            f'loss_base: {epoch_row["loss_base"]:.4f} '
            f'acc_train: {train_metrics["accuracy"]:.4f} '
            f'f1_train: {train_metrics["macro_f1"]:.4f} '
            f'loss_val: {val_metrics["loss"]:.4f} '
            f'acc_val: {val_metrics["accuracy"]:.4f} '
            f'f1_val: {val_metrics["macro_f1"]:.4f} '
            f'minor_f1_val: {val_metrics["minor_f1"]:.4f} '
            f'ord: {epoch_row["ordinal_loss"]:.4f} '
            f'gate_s: {val_metrics["gate_smiles_mean"]:.3f}±{val_metrics["gate_smiles_std"]:.3f} '
            f'gate_strength: {pair_stats["pair_gate_strength"]:.3f} '
            f'mol_scale: {fusion_stats["molformer_residual_scale"]:.4f} '
            f'time: {elapsed:.2f}s'
        )

        _save_last_checkpoint(
            last_path,
            epoch + 1,
            model,
            optimizer,
            scheduler,
            best_val_f1,
            best_epoch,
        )

        # Persist epoch history after every completed epoch.
        history_path = fold_dir / 'epoch_metrics.csv'
        old = None
        if history_path.exists() and start_epoch > 0:
            try:
                old = pd.read_csv(history_path)
            except Exception:
                old = None
        current = pd.DataFrame(epoch_rows)
        if old is not None and not old.empty:
            current = pd.concat(
                [old[old['epoch'] <= start_epoch], current],
                ignore_index=True,
            )
        current = current.drop_duplicates(
            subset=['epoch'],
            keep='last',
        ).sort_values('epoch')
        current.to_csv(history_path, index=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('Optimization Finished!')
    print(f'Best validation Macro-F1 epoch: {best_epoch}')
    print(
        'Total time elapsed: '
        f'{time.time() - t_total:.4f}s'
    )

    if not best_path.exists():
        raise RuntimeError(
            f'No best validation checkpoint was created: {best_path}'
        )

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_metrics = evaluate(
        model,
        test_loader,
        data_o,
        data_o_sim,
        data_s,
        data_s_sim,
        data_a,
        data_a_sim,
        args,
    )

    result_root = Path(args.result_root)
    pred_path = (
        result_root
        / f'fold_{int(args.zhongzi)}_test_predictions.csv'
    )
    pd.DataFrame(
        {
            'true_label': np.asarray(
                args.test_labels_for_save
                if hasattr(args, 'test_labels_for_save')
                else [],
                dtype=np.int64,
            )
        }
    ) if False else None

    # evaluate already returns predictions but labels are reconstructed here
    # in loader order to make the output file self-describing.
    true_labels = []
    class_labels = []
    for inp in test_loader:
        true_labels.extend(
            np.asarray(inp[2], dtype=np.int64).reshape(-1).tolist()
        )
        class_labels.extend(
            np.asarray(inp[3], dtype=np.int64).reshape(-1).tolist()
        )

    test_probs = torch.softmax(
        torch.as_tensor(test_metrics['logits'], dtype=torch.float32),
        dim=1,
    ).cpu().numpy()

    pd.DataFrame(
        {
            'true_label': true_labels,
            'pred_label': test_metrics['preds'].astype(np.int64),
            'class_label': class_labels,
            'prob_minor': test_probs[:, 0],
            'prob_moderate': test_probs[:, 1],
            'prob_major': test_probs[:, 2],
        }
    ).to_csv(pred_path, index=False)

    logits_path = (
        result_root
        / f'fold_{int(args.zhongzi)}_test_logits.npy'
    )
    np.save(logits_path, test_metrics['logits'])

    metrics_row = {
        'fold': int(args.zhongzi),
        'best_epoch': int(best_epoch),
        'test_loss': float(test_metrics['loss']),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_macro_f1': float(test_metrics['macro_f1']),
        'test_macro_recall': float(test_metrics['macro_recall']),
        'test_macro_precision': float(test_metrics['macro_precision']),
        'balanced_accuracy': float(test_metrics['balanced_accuracy']),
        'mcc': float(test_metrics['mcc']),
        'qwk': float(test_metrics['qwk']),
        'ordinal_mae': float(test_metrics['ordinal_mae']),
        'extreme_error_rate': float(test_metrics['extreme_error_rate']),
        'minor_precision': float(test_metrics['minor_precision']),
        'minor_recall': float(test_metrics['minor_recall']),
        'minor_f1': float(test_metrics['minor_f1']),
        'moderate_precision': float(test_metrics['moderate_precision']),
        'moderate_recall': float(test_metrics['moderate_recall']),
        'moderate_f1': float(test_metrics['moderate_f1']),
        'major_precision': float(test_metrics['major_precision']),
        'major_recall': float(test_metrics['major_recall']),
        'major_f1': float(test_metrics['major_f1']),
        'confusion_matrix': str(test_metrics['confusion_matrix']),
        'auc_micro': float(test_metrics['auc_micro']),
        'aupr_micro': float(test_metrics['aupr_micro']),
        'auc_macro': float(test_metrics['auc_macro']),
        'aupr_macro': float(test_metrics['aupr_macro']),
        'molformer_residual_scale': float(
            model.molecular_fusion_stats()['molformer_residual_scale']
        ),
        'pair_gate_strength': float(
            model.pairgate_ordinal_stats()['pair_gate_strength']
        ),
        'gate_trim_mean': float(test_metrics['gate_trim_mean']),
        'gate_smiles_mean': float(test_metrics['gate_smiles_mean']),
        'gate_smiles_std': float(test_metrics['gate_smiles_std']),
        'ordinal_loss': float(test_metrics['ordinal_loss']),
        'ordinal_infer_blend': float(args.ordinal_infer_blend),
        'ordinal_weight': float(args.ordinal_weight),
    }

    metrics_path = (
        result_root
        / f'fold_{int(args.zhongzi)}_test_metrics.csv'
    )
    pd.DataFrame([metrics_row]).to_csv(
        metrics_path,
        index=False,
    )

    with open(args.out_file, 'a') as f:
        f.write(
            f'{int(args.zhongzi)}  '
            f'{metrics_row["test_accuracy"]}   '
            f'{metrics_row["test_macro_f1"]}   '
            f'{metrics_row["test_macro_recall"]}   '
            f'{metrics_row["test_macro_precision"]}   '
            f'{metrics_row["auc_micro"]}   '
            f'{metrics_row["aupr_micro"]}   '
            f'{metrics_row["auc_macro"]}   '
            f'{metrics_row["aupr_macro"]}\n'
        )

    print(
        '[PAIRGATE-ORD TEST] '
        f'loss={metrics_row["test_loss"]:.4f} '
        f'acc={metrics_row["test_accuracy"]:.4f} '
        f'balanced_acc={metrics_row["balanced_accuracy"]:.4f} '
        f'macro_f1={metrics_row["test_macro_f1"]:.4f} '
        f'macro_recall={metrics_row["test_macro_recall"]:.4f} '
        f'macro_precision={metrics_row["test_macro_precision"]:.4f} '
        f'mcc={metrics_row["mcc"]:.4f} '
        f'qwk={metrics_row["qwk"]:.4f} '
        f'ordinal_mae={metrics_row["ordinal_mae"]:.4f} '
        f'extreme_err={metrics_row["extreme_error_rate"]:.4f} '
        f'minor_p/r/f1={metrics_row["minor_precision"]:.4f}/'
        f'{metrics_row["minor_recall"]:.4f}/{metrics_row["minor_f1"]:.4f} '
        f'moderate_p/r/f1={metrics_row["moderate_precision"]:.4f}/'
        f'{metrics_row["moderate_recall"]:.4f}/{metrics_row["moderate_f1"]:.4f} '
        f'major_p/r/f1={metrics_row["major_precision"]:.4f}/'
        f'{metrics_row["major_recall"]:.4f}/{metrics_row["major_f1"]:.4f} '
        f'gate_smiles={metrics_row["gate_smiles_mean"]:.3f}±'
        f'{metrics_row["gate_smiles_std"]:.3f} '
        f'gate_strength={metrics_row["pair_gate_strength"]:.3f}'
    )
    print(
        '[PAIRGATE-ORD CONFUSION] '
        f'{metrics_row["confusion_matrix"]}'
    )

    return metrics_row
