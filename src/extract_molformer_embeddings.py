#!/usr/bin/env python3
"""Precompute frozen IBM MoLFormer embeddings for DDInter drugs.

Adapted from the user's supplied LANTERN implementation:
  AutoModel/AutoTokenizer("ibm/MoLFormer-XL-both-10pct",
                          deterministic_eval=True, trust_remote_code=True)
  and outputs.pooler_output.

The MSFCL training stage never loads the large language model; it only loads the
small .npz generated here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def read_drug_smiles(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        raise RuntimeError(f"No drug rows found in {path}")

    header = [str(x).strip() for x in rows[0]]
    lower = [x.lower() for x in header]

    def find_col(candidates, fallback):
        for candidate in candidates:
            if candidate in lower:
                return lower.index(candidate)
        return fallback

    id_col = find_col(
        ["drug_id", "drugid", "drug", "id", "drug1_id"],
        0,
    )
    smiles_col = find_col(
        ["smiles", "canonical_smiles", "canonical smiles"],
        1,
    )

    if smiles_col >= len(header):
        raise RuntimeError(
            f"Could not infer a SMILES column from header: {header}"
        )

    drug_ids = []
    smiles = []
    seen = set()

    for line_no, row in enumerate(rows[1:], start=2):
        if max(id_col, smiles_col) >= len(row):
            raise RuntimeError(
                f"Malformed CSV row {line_no}: expected columns "
                f"{id_col} and {smiles_col}"
            )
        drug_id = str(row[id_col]).strip()
        smi = str(row[smiles_col]).strip()
        if not drug_id or not smi:
            raise RuntimeError(
                f"Empty drug ID or SMILES at row {line_no}: {row}"
            )
        if drug_id in seen:
            raise RuntimeError(
                f"Duplicate drug ID in smiles file: {drug_id}"
            )
        seen.add(drug_id)
        drug_ids.append(drug_id)
        smiles.append(smi)

    return drug_ids, smiles, header[id_col], header[smiles_col]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles_csv",
        default="trimnet/data/ddinter/ddinter_smiles.csv",
    )
    parser.add_argument(
        "--model",
        default="ibm/MoLFormer-XL-both-10pct",
        help=(
            "Hugging Face model ID or a local downloaded model directory. "
            "For an offline server, pass the local directory."
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "trimnet/data/ddinter/"
            "molformer_ibm_molformer_xl_both_10pct.npz"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--pooling",
        choices=["pooler", "mean"],
        default="pooler",
        help="LANTERN uses pooler_output; mean is provided only as fallback.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Never access the network; useful on the offline server.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers is required only for this one-time extraction step. "
            "The supplied LANTERN environment used transformers==4.46.3."
        ) from exc

    smiles_path = Path(args.smiles_csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    drug_ids, smiles, id_name, smiles_name = read_drug_smiles(smiles_path)
    print(
        f"[DATA] drugs={len(drug_ids)} id_col={id_name!r} "
        f"smiles_col={smiles_name!r}"
    )

    model_ref = args.model
    kwargs = dict(
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )

    print(f"[MODEL] loading {model_ref}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        **kwargs,
    )
    model = AutoModel.from_pretrained(
        model_ref,
        deterministic_eval=True,
        **kwargs,
    )

    device = torch.device(
        args.device
        if torch.cuda.is_available() and str(args.device).startswith("cuda")
        else "cpu"
    )
    model = model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(smiles), int(args.batch_size)):
            stop = min(start + int(args.batch_size), len(smiles))
            batch_smiles = smiles[start:stop]

            encoded = tokenizer(
                batch_smiles,
                padding=True,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }
            outputs = model(**encoded)

            if args.pooling == "pooler":
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is None:
                    raise RuntimeError(
                        "MoLFormer output has no pooler_output in this "
                        "transformers/model revision. Re-run with "
                        "--pooling mean only if you intentionally want a "
                        "different pooling protocol."
                    )
            else:
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    raise RuntimeError(
                        "Model output has no last_hidden_state."
                    )
                mask = encoded.get("attention_mask")
                if mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    mask = mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (
                        (hidden * mask).sum(dim=1)
                        / mask.sum(dim=1).clamp_min(1.0)
                    )

            embeddings.append(
                pooled.detach().float().cpu().numpy()
            )
            print(
                f"[EMBED] {stop}/{len(smiles)} "
                f"shape={tuple(pooled.shape)}"
            )

    matrix = np.concatenate(embeddings, axis=0).astype(np.float32)
    if matrix.shape[0] != len(drug_ids):
        raise RuntimeError(
            f"Embedding count mismatch: {matrix.shape[0]} vs {len(drug_ids)}"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("MoLFormer embeddings contain NaN or Inf.")

    np.savez_compressed(
        output_path,
        drug_ids=np.asarray(drug_ids, dtype=str),
        embeddings=matrix,
        model_ref=np.asarray([str(model_ref)], dtype=str),
        pooling=np.asarray([str(args.pooling)], dtype=str),
    )

    meta = {
        "smiles_csv": str(smiles_path),
        "model": str(model_ref),
        "pooling": str(args.pooling),
        "n_drugs": int(matrix.shape[0]),
        "embedding_dim": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] saved {output_path}")
    print(f"[OK] shape={matrix.shape} dtype={matrix.dtype}")
    print(f"[OK] metadata={meta_path}")


if __name__ == "__main__":
    main()
