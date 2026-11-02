# -*- coding: utf-8 -*-
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from datetime import datetime
import time 
import argparse
import copy
import shutil
from pathlib import Path
import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,precision_score
import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataset1,DrugDataLoader, TOTAL_ATOM_FEATS


######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=TOTAL_ATOM_FEATS, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=64, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=3, help='num of interaction types')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=300, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0,1])
parser.add_argument('--zhongzi', type=int, default=0)


args = parser.parse_args()
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size
zhongzi=args.zhongzi
weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)

###### Dataset
df_ddi_train = pd.read_csv('data/ddinter/'+str(zhongzi)+'/ddi_training1.csv')
df_ddi_val = pd.read_csv('data/ddinter/'+str(zhongzi)+'/ddi_validation1.csv')
df_ddi_test = pd.read_csv('data/ddinter/'+str(zhongzi)+'/ddi_test1.csv')
# df_ddi_train = pd.read_csv('data/drugbank_2/'+str(zhongzi)+'/ddi_training1.csv')
# df_ddi_val = pd.read_csv('data/drugbank_2/'+str(zhongzi)+'/ddi_validation1.csv')
# df_ddi_test = pd.read_csv('data/drugbank_2/'+str(zhongzi)+'/ddi_test1.csv')


train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['Drug1_ID'], df_ddi_train['Drug2_ID'], df_ddi_train['Y'])]
val_tup = [(h, t, r) for h, t, r in zip(df_ddi_val['Drug1_ID'], df_ddi_val['Drug2_ID'], df_ddi_val['Y'])]
test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['Drug1_ID'], df_ddi_test['Drug2_ID'], df_ddi_test['Y'])]


train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
test_data = DrugDataset(test_tup, disjoint_split=False)
druglist=DrugDataset1(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)

print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size, drop_last=False)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size ,drop_last=False)
druglist=DrugDataLoader(druglist)

def compute_weigth(batch, device, training=True):
    '''
        *batch: (pos_tri, neg_tri)
        *pos/neg_tri: (batch_h, batch_t, batch_r)
    '''
    probas_pred, ground_truth = [], []
    for batch in druglist:
        pos_tri = batch
        pos_tri = [tensor.to(device=device) for tensor in pos_tri]

        h_weight = model.get_weight(pos_tri,zhongzi)
        break

    return h_weight

def do_compute(batch, device, model,training=True):

        probas_pred, ground_truth = [], []
        pos_tri, neg_tri ,hlist= batch


        pos_tri = [tensor.to(device=device) for tensor in pos_tri]

        p_score,gt = model(pos_tri)

        return p_score, gt


def do_compute_metrics(probas_pred, target):


    y_pred_train1 = []
    y_label_train = np.array(target)

    y_label_train=y_label_train.reshape((-1))
    y_pred_train = np.array(probas_pred).reshape((-1, 3))
    for i in range(y_pred_train.shape[0]):
        a = np.max(y_pred_train[i])
        #print(y_pred_train[i])
        #print(a)
        for j in range(y_pred_train.shape[1]):
            if y_pred_train[i][j] == a:
                # print(y_pred_train[i][j])
                y_pred_train1.append(j)
                break


    acc = accuracy_score(y_label_train, y_pred_train1)
    f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
    recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
    precision1 = precision_score(y_label_train, y_pred_train1, average='macro')
    aaa=y_label_train
    bbb=y_pred_train

    return acc, f1_score1, recall1,precision1,aaa,bbb


def _save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    val_loss,
    val_acc,
    val_f1,
    val_recall,
    val_precision,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "scheduler_state_dict": (
                copy.deepcopy(scheduler.state_dict())
                if scheduler is not None
                else None
            ),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
            "val_macro_recall": float(val_recall),
            "val_macro_precision": float(val_precision),
            "fold": int(zhongzi),
            "args": vars(args),
        },
        path,
    )


def _evaluate_checkpoint(
    model,
    checkpoint_path,
    checkpoint_name,
    test_data_loader,
    loss_fn,
    device,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_probas_pred = []
    test_ground_truth = []
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_data_loader:
            probas_pred, gt = do_compute(
                batch,
                device,
                model,
                training=False,
            )
            batch_loss = loss_fn(probas_pred, gt)
            batch_size_now = int(gt.shape[0])
            test_loss += batch_loss.item() * batch_size_now

            test_probas_pred.append(
                probas_pred.detach().cpu().numpy()
            )
            test_ground_truth.append(gt.detach().cpu().numpy())

    test_loss /= len(test_data)
    test_probas_pred = np.concatenate(test_probas_pred)
    test_ground_truth = np.concatenate(test_ground_truth)

    (
        test_acc,
        test_f1,
        test_recall,
        test_precision,
        y_true,
        y_logits,
    ) = do_compute_metrics(test_probas_pred, test_ground_truth)

    y_pred = np.argmax(y_logits, axis=1)
    confusion = metrics.confusion_matrix(
        y_true.astype(int),
        y_pred.astype(int),
        labels=[0, 1, 2],
    )
    report = metrics.classification_report(
        y_true.astype(int),
        y_pred.astype(int),
        labels=[0, 1, 2],
        target_names=["Minor", "Moderate", "Major"],
        output_dict=True,
        zero_division=0,
    )

    print("\n" + "=" * 72)
    print(
        f"[TEST] {checkpoint_name} | "
        f"saved_epoch={checkpoint['epoch']}"
    )
    print(
        f"test_loss={test_loss:.4f}, "
        f"test_acc={test_acc:.4f}, "
        f"test_macro_f1={test_f1:.4f}, "
        f"test_macro_recall={test_recall:.4f}, "
        f"test_macro_precision={test_precision:.4f}"
    )
    print("True counts     :", np.bincount(y_true.astype(int), minlength=3))
    print("Predicted counts:", np.bincount(y_pred.astype(int), minlength=3))
    print("Confusion matrix:\n", confusion)

    return {
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "saved_epoch": int(checkpoint["epoch"]),
        "saved_val_loss": float(checkpoint["val_loss"]),
        "saved_val_macro_f1": float(checkpoint["val_macro_f1"]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_macro_recall": float(test_recall),
        "test_macro_precision": float(test_precision),
        "minor_f1": float(report["Minor"]["f1-score"]),
        "moderate_f1": float(report["Moderate"]["f1-score"]),
        "major_f1": float(report["Major"]["f1-score"]),
    }


def _export_current_embedding(model, checkpoint_name):
    """Export one drug embedding file for the currently loaded checkpoint."""
    model.eval()
    with torch.no_grad():
        compute_weigth(druglist, device)

    default_path = Path(
        f"data/ddinter/drug_emb_trimnet{zhongzi}.npy"
    )
    tagged_path = Path(
        f"data/ddinter/drug_emb_trimnet{zhongzi}_"
        f"{checkpoint_name}.npy"
    )
    if not default_path.exists():
        raise FileNotFoundError(
            f"Embedding was not generated: {default_path}"
        )
    shutil.copy2(default_path, tagged_path)
    print(f"[EMBEDDING] saved: {tagged_path}")


def train(
    model,
    train_data_loader,
    val_data_loader,
    loss_fn,
    optimizer,
    n_epochs,
    device,
    scheduler=None,
):
    print("Starting training at", datetime.today())

    checkpoint_root = Path(
        os.environ.get(
            "TRIMNET_CHECKPOINT_ROOT",
            "checkpoints",
        )
    )
    checkpoint_dir = checkpoint_root / f"fold_{zhongzi}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_loss_path = checkpoint_dir / "best_val_loss.pt"
    best_f1_path = checkpoint_dir / "best_val_macro_f1.pt"
    last_path = checkpoint_dir / "last_epoch.pt"

    best_val_loss = float("inf")
    best_val_f1 = float("-inf")
    best_loss_epoch = -1
    best_f1_epoch = -1
    start_epoch = 0

    auto_resume = (
        os.environ.get(
            "TRIMNET_AUTO_RESUME",
            "0",
        )
        == "1"
    )

    if auto_resume:
        if best_loss_path.exists():
            best_loss_checkpoint = torch.load(
                best_loss_path,
                map_location=device,
            )
            best_val_loss = float(
                best_loss_checkpoint.get(
                    "val_loss",
                    float("inf"),
                )
            )
            best_loss_epoch = int(
                best_loss_checkpoint.get(
                    "epoch",
                    -1,
                )
            )

        if best_f1_path.exists():
            best_f1_checkpoint = torch.load(
                best_f1_path,
                map_location=device,
            )
            best_val_f1 = float(
                best_f1_checkpoint.get(
                    "val_macro_f1",
                    float("-inf"),
                )
            )
            best_f1_epoch = int(
                best_f1_checkpoint.get(
                    "epoch",
                    -1,
                )
            )

        if last_path.exists():
            resume_checkpoint = torch.load(
                last_path,
                map_location=device,
            )

            checkpoint_fold = int(
                resume_checkpoint.get(
                    "fold",
                    zhongzi,
                )
            )
            if checkpoint_fold != int(zhongzi):
                raise RuntimeError(
                    f"Checkpoint fold={checkpoint_fold} "
                    f"does not match requested fold={zhongzi}"
                )

            model.load_state_dict(
                resume_checkpoint[
                    "model_state_dict"
                ]
            )
            model.to(device)

            optimizer.load_state_dict(
                resume_checkpoint[
                    "optimizer_state_dict"
                ]
            )

            scheduler_state = resume_checkpoint.get(
                "scheduler_state_dict"
            )
            if (
                scheduler is not None
                and scheduler_state is not None
            ):
                scheduler.load_state_dict(
                    scheduler_state
                )

                # 当前检查点是在scheduler.step()之前保存的，
                # 恢复后补执行一次以保持学习率进度连续。
                try:
                    scheduler.step()
                except TypeError:
                    pass

            start_epoch = int(
                resume_checkpoint["epoch"]
            )

            print(
                "[RESUME] "
                f"fold={zhongzi}, "
                f"start_epoch={start_epoch}, "
                f"target_epoch={n_epochs}, "
                f"best_loss_epoch={best_loss_epoch}, "
                f"best_f1_epoch={best_f1_epoch}"
            )

    for epoch in range(
        start_epoch + 1,
        n_epochs + 1,
    ):
        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []

        model.train()
        for batch in train_data_loader:
            p_score, gt = do_compute(batch, device, model)
            batch_loss = loss_fn(p_score, gt)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            batch_size_now = int(gt.shape[0])
            train_loss += batch_loss.item() * batch_size_now
            train_probas_pred.append(
                p_score.detach().cpu().numpy()
            )
            train_ground_truth.append(gt.detach().cpu().numpy())

        train_loss /= len(train_data)
        train_probas_pred = np.concatenate(train_probas_pred)
        train_ground_truth = np.concatenate(train_ground_truth)
        (
            train_acc,
            train_f1,
            train_recall,
            train_precision,
            _,
            _,
        ) = do_compute_metrics(
            train_probas_pred,
            train_ground_truth,
        )

        model.eval()
        with torch.no_grad():
            for batch in val_data_loader:
                probas_pred, gt = do_compute(
                    batch,
                    device,
                    model,
                    training=False,
                )
                batch_loss = loss_fn(probas_pred, gt)
                batch_size_now = int(gt.shape[0])
                val_loss += batch_loss.item() * batch_size_now

                val_probas_pred.append(
                    probas_pred.detach().cpu().numpy()
                )
                val_ground_truth.append(gt.detach().cpu().numpy())

        val_loss /= len(val_data)
        val_probas_pred = np.concatenate(val_probas_pred)
        val_ground_truth = np.concatenate(val_ground_truth)
        (
            val_acc,
            val_f1,
            val_recall,
            val_precision,
            _,
            _,
        ) = do_compute_metrics(
            val_probas_pred,
            val_ground_truth,
        )

        # Save checkpoint with minimum validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            _save_checkpoint(
                best_loss_path,
                model,
                optimizer,
                scheduler,
                epoch,
                train_loss,
                val_loss,
                val_acc,
                val_f1,
                val_recall,
                val_precision,
            )
            print(
                f"[SAVE best_val_loss] epoch={epoch}, "
                f"val_loss={val_loss:.6f}"
            )

        # Save checkpoint with maximum validation Macro-F1.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_f1_epoch = epoch
            _save_checkpoint(
                best_f1_path,
                model,
                optimizer,
                scheduler,
                epoch,
                train_loss,
                val_loss,
                val_acc,
                val_f1,
                val_recall,
                val_precision,
            )
            print(
                f"[SAVE best_val_macro_f1] epoch={epoch}, "
                f"val_macro_f1={val_f1:.6f}"
            )

        # Overwrite every epoch; at the end this is the final epoch.
        _save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            train_loss,
            val_loss,
            val_acc,
            val_f1,
            val_recall,
            val_precision,
        )

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch: {epoch} ({time.time() - start:.4f}s), "
            f"train_loss: {train_loss:.4f}, "
            f"val_loss: {val_loss:.4f}, "
            f"train_acc: {train_acc:.4f}, "
            f"val_acc: {val_acc:.4f}"
        )
        print(
            f"\t\ttrain_f1: {train_f1:.4f}, "
            f"val_f1: {val_f1:.4f}, "
            f"train_recall: {train_recall:.4f}, "
            f"val_recall: {val_recall:.4f}, "
            f"train_precision: {train_precision:.4f}, "
            f"val_precision: {val_precision:.4f}"
        )
        print(
            f"\t\tbest_loss_epoch={best_loss_epoch}, "
            f"best_val_loss={best_val_loss:.4f}; "
            f"best_f1_epoch={best_f1_epoch}, "
            f"best_val_macro_f1={best_val_f1:.4f}"
        )

    # Test all three only after training is complete.
    checkpoint_specs = [
        ("best_val_loss", best_loss_path),
        ("best_val_macro_f1", best_f1_path),
        ("last_epoch", last_path),
    ]

    results = []
    for checkpoint_name, checkpoint_path in checkpoint_specs:
        result = _evaluate_checkpoint(
            model,
            checkpoint_path,
            checkpoint_name,
            test_data_loader,
            loss_fn,
            device,
        )
        results.append(result)
        _export_current_embedding(model, checkpoint_name)

    result_file = checkpoint_dir / "three_checkpoint_test_metrics.csv"
    pd.DataFrame(results).to_csv(result_file, index=False)

    print("\n" + "=" * 72)
    print("Three-checkpoint test summary")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"[SAVE] {result_file}")

    # Restore the validation-Macro-F1 checkpoint as the default output.
    best_f1_checkpoint = torch.load(best_f1_path, map_location=device)
    model.load_state_dict(best_f1_checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    best_f1_embedding = Path(
        f"data/ddinter/drug_emb_trimnet{zhongzi}_"
        "best_val_macro_f1.npy"
    )
    default_embedding = Path(
        f"data/ddinter/drug_emb_trimnet{zhongzi}.npy"
    )
    shutil.copy2(best_f1_embedding, default_embedding)
    print(
        "[DEFAULT] Restored best validation Macro-F1 model and "
        f"embedding: {default_embedding}"
    )

    return results


model = models.TrimNet(55, 10, hidden_dim=64, depth=3,heads=4, dropout=0.2, outdim=1)

model.to(device=device)
loss=torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))



# if __name__ == '__main__':
train(model, train_data_loader, val_data_loader, loss, optimizer, n_epochs, device, scheduler)


